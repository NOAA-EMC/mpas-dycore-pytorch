"""Utilities for optional torch.compile wrapping of dycore kernels."""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib
import inspect
import itertools
import json
import marshal
import os
import threading
import time
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

CompiledFn = Callable[..., Any]
FunctionCacheKey = tuple[Hashable, ...]

_COMPILED_CACHE: dict[tuple[object, ...], CompiledFn] = {}
_COMPILE_EVENT_SINK: Callable[[dict[str, Any]], None] | None = None
_COMPILE_EVENT_LOCK = threading.Lock()
_COMPILE_EVENT_COUNTER = itertools.count()
_COMPILE_RUNTIME_COUNTERS: dict[str, int] = {
    "wrapper_create": 0,
    "wrapper_cache_miss": 0,
    "eager_fallback": 0,
    "autodiff_bypass": 0,
    "compiled_call_error": 0,
    "specialization_revision": 0,
}
_COMPILE_SOURCE_HASH_CACHE: dict[tuple[str, int, int, int], str] = {}
_CUDA_INDUCTOR_CPU_PREQUALIFICATION_REQUIREMENTS = (
    "cpu_eager_vs_inductor_bitwise",
    "cpu_dynamo_one_graph_zero_breaks",
    "cpu_reverse_ad",
    "cpu_forward_ad_jvp",
    "cpu_ts1_split",
    "cpu_ts1_coupled",
    "cpu_ts5_split",
    "cpu_ts5_coupled",
)
_CUDA_INDUCTOR_GPU_PROMOTION_REQUIREMENTS = (
    "gpu_eager_vs_inductor_accuracy",
    "gpu_repeated_forward_no_crash",
    "gpu_guard_recompile_stability",
    "gpu_production_callsite_exercised",
    "gpu_section_timing_snapshot",
    "gpu_memory_snapshot",
)
_COMPILE_KERNEL_KINDS = frozenset(
    {
        "pure_numeric",
        "orchestration_blocked",
        "legacy_alias",
    }
)
_COMPILED_REVERSE_AD_POLICIES = frozenset(
    {"blocked", "cpu_experiment", "cpu_validated", "cuda_validated"}
)
_CUDA_INDUCTOR_STAGE_ORDER = (
    "diagnostic_solve_diagnostics",
    "explicit_scalar_transport",
    "monotonic_scalar_transport",
    "acoustic_step",
    "dynamics_tendency_normal_velocity_tendency",
    "dynamics_tendency_normal_velocity_mixing",
    "dynamics_tendency_vertical_velocity_horizontal_advection_theta_horizontal_advection",
    "large_step_recovery",
    "small_step_perturbation_divergence",
    "combined_hot",
)
_CUDA_INDUCTOR_STAGE_ALIASES = {
    "diagnostic_solve": "diagnostic_solve_diagnostics",
    "hot": "combined_hot",
    "combined": "combined_hot",
}
# Historical benchmark receipts are development records, not wheel content or
# runtime authority.  The installed policy starts closed and can only be opened
# by package-owned evidence in a future release.
_LEGACY_CUDA_INDUCTOR_GPU_VALIDATION_EVIDENCE: dict[
    tuple[str, str, str], list[str]
] = {}
_CPU_INDUCTOR_EVIDENCE = _CUDA_INDUCTOR_CPU_PREQUALIFICATION_REQUIREMENTS
_CPU_AD_EVIDENCE = ("cpu_reverse_ad", "cpu_forward_ad_jvp")
_COMPILE_AD_EVIDENCE_SCOPES = (*_CPU_AD_EVIDENCE, "compiled_reverse_ad")
_CPU_FORWARD_INDUCTOR_EVIDENCE = tuple(
    item for item in _CPU_INDUCTOR_EVIDENCE if item not in _CPU_AD_EVIDENCE
)
_INSTALLED_GPU_EVIDENCE: tuple[EvidenceInput, ...] = ()


@dataclass(frozen=True)
class CompileEvidenceRef:
    """Callable-scoped evidence; legacy labels are explicitly non-authorizing."""

    evidence_id: str
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    callable_path: str | None = None
    callable_source_sha256: str | None = None
    source_manifest_path: str | None = None
    source_manifest_sha256: str | None = None
    git_commit: str | None = None
    dirty_diff_sha256: str | None = None
    device_type: str | None = None
    dtype: str | None = None
    shapes: tuple[str, ...] = ()
    scenarios: tuple[str, ...] = ()
    rk_stages: tuple[int, ...] = ()
    torch_version: str | None = None
    cuda_version: str | None = None
    compute_capability: str | None = None
    compile_backend: str | None = None
    compile_mode: str | None = None
    cudagraph_policy: str | None = None
    validated_scopes: tuple[str, ...] = ()
    authorizing: bool = False

    @classmethod
    def legacy(cls, evidence_id: str) -> CompileEvidenceRef:
        return cls(evidence_id=str(evidence_id), authorizing=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "callable_path": self.callable_path,
            "callable_source_sha256": self.callable_source_sha256,
            "source_manifest_path": self.source_manifest_path,
            "source_manifest_sha256": self.source_manifest_sha256,
            "git_commit": self.git_commit,
            "dirty_diff_sha256": self.dirty_diff_sha256,
            "device_type": self.device_type,
            "dtype": self.dtype,
            "shapes": list(self.shapes),
            "scenarios": list(self.scenarios),
            "rk_stages": list(self.rk_stages),
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "compute_capability": self.compute_capability,
            "compile_backend": self.compile_backend,
            "compile_mode": self.compile_mode,
            "cudagraph_policy": self.cudagraph_policy,
            "validated_scopes": list(self.validated_scopes),
            "authorizing": self.authorizing,
        }


EvidenceInput = str | CompileEvidenceRef


_DYNAMICS_TENDENCY_WHOLE_STAGE_REVERSE_AOT_CPU_EVIDENCE = CompileEvidenceRef(
    evidence_id="cpu_compiled_reverse_ad_experimental",
    callable_path="mpas_pytorch.dynamics.dyn_tend.atm_compute_dyn_tend_work",
    device_type="cpu",
    validated_scopes=("compiled_reverse_ad",),
    authorizing=False,
)


def _evidence_refs(values: Iterable[EvidenceInput]) -> tuple[CompileEvidenceRef, ...]:
    return tuple(
        value
        if isinstance(value, CompileEvidenceRef)
        else CompileEvidenceRef.legacy(value)
        for value in values
    )


def _evidence_ids(values: Iterable[CompileEvidenceRef]) -> frozenset[str]:
    return frozenset(value.evidence_id for value in values)


def _has_authorizing_evidence(
    values: Iterable[CompileEvidenceRef],
    *,
    device_type: str,
) -> bool:
    return any(
        value.authorizing and value.device_type == device_type for value in values
    )


def _has_authorizing_scope_coverage(
    values: Iterable[CompileEvidenceRef],
    *,
    device_type: str,
    required_scopes: Iterable[str],
) -> bool:
    """Return whether validly scoped authority exists without combining legacy labels."""
    required = set(required_scopes)
    return any(
        value.authorizing
        and value.device_type == device_type
        and required.issubset(value.validated_scopes)
        for value in values
    )


def _spec_has_authorizing_cuda_promotion_evidence(spec: CompileKernelSpec) -> bool:
    """Require callable-specific CPU, AD, and GPU authority for safe CUDA admission."""
    return (
        spec.allow_staged_cuda
        and _has_current_authorizing_scope_coverage(
            spec,
            spec.cpu_evidence,
            evidence_group="cpu_evidence",
            device_type="cpu",
            required_scopes=_CPU_FORWARD_INDUCTOR_EVIDENCE,
        )
        and _has_current_authorizing_scope_coverage(
            spec,
            spec.ad_evidence,
            evidence_group="ad_evidence",
            device_type="cpu",
            required_scopes=_CPU_AD_EVIDENCE,
        )
        and _has_current_authorizing_scope_coverage(
            spec,
            spec.gpu_evidence,
            evidence_group="gpu_evidence",
            device_type="cuda",
            required_scopes=_CUDA_INDUCTOR_GPU_PROMOTION_REQUIREMENTS,
        )
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _source_manifest_current_problems(path: Path) -> tuple[str, ...]:
    """Validate an evidence source manifest against the current source tree."""
    problems: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return (f"source manifest is not valid JSON: {exc}",)
    if not isinstance(payload, Mapping):
        return ("source manifest is not an object",)
    if payload.get("schema") != "mpas-source-provenance.v1":
        problems.append("source manifest schema is not mpas-source-provenance.v1")
    if payload.get("generation_clean") is not True:
        problems.append("source manifest was not generated from clean source")
    if payload.get("dirty_source_paths") not in ([], ()):
        problems.append("source manifest records dirty source paths")
    entries = payload.get("source_manifest")
    if not isinstance(entries, list) or not entries:
        problems.append("source manifest entries are missing")
        return tuple(problems)
    if payload.get("source_manifest_sha256") != _canonical_json_sha256(entries):
        problems.append("source manifest entry digest mismatch")

    root = path.parent.resolve()
    seen: set[str] = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            problems.append(f"source manifest entry {index} is not an object")
            continue
        relative = str(raw_entry.get("path", ""))
        if not relative or relative in seen:
            problems.append(
                f"source manifest entry {index} has a missing or duplicate path"
            )
            continue
        seen.add(relative)
        source_path = (root / relative).resolve()
        try:
            source_path.relative_to(root)
        except ValueError:
            problems.append(f"source manifest path escapes repository: {relative}")
            continue
        if not source_path.is_file():
            problems.append(f"source manifest path is missing: {relative}")
            continue
        if raw_entry.get("sha256") != _sha256_path(source_path):
            problems.append(f"source manifest path is stale: {relative}")
        if raw_entry.get("size_bytes") != source_path.stat().st_size:
            problems.append(f"source manifest size is stale: {relative}")
    return tuple(problems)


def _callable_source_path(callable_path: str) -> Path:
    module_name, attribute = callable_path.rsplit(".", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    source_path = inspect.getsourcefile(value)
    if source_path is None:
        raise ValueError(f"callable {callable_path!r} has no Python source file")
    return Path(source_path).resolve()


def _artifact_has_passing_helper_result(
    payload: Mapping[str, Any],
    *,
    helper: str,
) -> bool:
    """Accept a direct helper result or a production preflight helper receipt."""
    results = payload.get("results")
    if not isinstance(results, list):
        return False
    for result in results:
        if not isinstance(result, Mapping):
            continue
        if result.get("helper") == helper and result.get("pass") is True:
            return True
        preflight = result.get("compile_preflight")
        if not isinstance(preflight, Mapping):
            continue
        helpers = preflight.get("helpers")
        if not isinstance(helpers, list):
            continue
        if any(
            isinstance(item, Mapping)
            and item.get("helper") == helper
            and item.get("status") == "pass"
            for item in helpers
        ):
            return True
    return False


def _artifact_has_production_callsite_receipt(
    payload: Mapping[str, Any],
    *,
    helper: str,
) -> bool:
    """Require a production preflight cold exercise and a stable same-key warm call."""

    def integer_field(record: Mapping[str, Any], key: str) -> int | None:
        value = record.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    results = payload.get("results")
    if not isinstance(results, list):
        return False
    for result in results:
        if not isinstance(result, Mapping):
            continue
        preflight = result.get("compile_preflight")
        if not isinstance(preflight, Mapping) or preflight.get("status") != "pass":
            continue
        helpers = preflight.get("helpers")
        if not isinstance(helpers, list):
            continue
        for item in helpers:
            if not isinstance(item, Mapping):
                continue
            if item.get("helper") != helper or item.get("status") != "pass":
                continue
            runs = item.get("runs")
            if not isinstance(runs, list):
                continue
            cold_exercised = any(
                isinstance(run, Mapping)
                and run.get("kind") == "cold_compile"
                and (integer_field(run, "event_count") or 0) > 0
                for run in runs
            )
            warm_stable = any(
                isinstance(run, Mapping)
                and run.get("kind") == "same_key_stability"
                and (integer_field(run, "event_count") or 0) > 0
                and integer_field(run, "recompile_indicator_total") == 0
                for run in runs
            )
            if cold_exercised and warm_stable:
                return True
    return False


def _authorizing_evidence_problems(
    spec: CompileKernelSpec,
    ref: CompileEvidenceRef,
    *,
    evidence_group: str,
) -> tuple[str, ...]:
    prefix = f"{spec.name}: authorizing {evidence_group} {ref.evidence_id!r}"
    problems: list[str] = []
    required = {
        "artifact_path": ref.artifact_path,
        "artifact_sha256": ref.artifact_sha256,
        "callable_path": ref.callable_path,
        "callable_source_sha256": ref.callable_source_sha256,
        "source_manifest_path": ref.source_manifest_path,
        "source_manifest_sha256": ref.source_manifest_sha256,
        "git_commit": ref.git_commit,
        "dirty_diff_sha256": ref.dirty_diff_sha256,
        "device_type": ref.device_type,
        "dtype": ref.dtype,
        "shapes": ref.shapes,
        "scenarios": ref.scenarios,
        "rk_stages": ref.rk_stages,
        "torch_version": ref.torch_version,
        "compile_backend": ref.compile_backend,
        "compile_mode": ref.compile_mode,
        "validated_scopes": ref.validated_scopes,
    }
    if ref.device_type == "cuda":
        required.update(
            {
                "cuda_version": ref.cuda_version,
                "compute_capability": ref.compute_capability,
                "cudagraph_policy": ref.cudagraph_policy,
            }
        )
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        problems.append(f"{prefix} missing {missing}")
        return tuple(problems)
    if ref.callable_path != spec.callable:
        problems.append(f"{prefix} callable mismatch")
        return tuple(problems)
    if len(str(ref.git_commit)) != 40:
        problems.append(f"{prefix} git_commit must be a full SHA-1")
    for key, digest in (
        ("artifact_sha256", ref.artifact_sha256),
        ("callable_source_sha256", ref.callable_source_sha256),
        ("source_manifest_sha256", ref.source_manifest_sha256),
        ("dirty_diff_sha256", ref.dirty_diff_sha256),
    ):
        value = str(digest)
        if len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value.lower()
        ):
            problems.append(f"{prefix} {key} must be SHA-256")
    for path_key, path_value, expected_hash in (
        ("artifact", ref.artifact_path, ref.artifact_sha256),
        ("source manifest", ref.source_manifest_path, ref.source_manifest_sha256),
    ):
        path = Path(str(path_value)).expanduser()
        if not path.is_absolute():
            problems.append(f"{prefix} {path_key} path must be absolute")
            continue
        if not path.is_file():
            problems.append(f"{prefix} {path_key} is missing: {path}")
        elif _sha256_path(path) != expected_hash:
            problems.append(f"{prefix} {path_key} hash mismatch")
        elif path_key == "source manifest":
            problems.extend(
                f"{prefix} {problem}"
                for problem in _source_manifest_current_problems(path)
            )
        elif path_key == "artifact":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                problems.append(f"{prefix} artifact is not valid JSON: {exc}")
            else:
                if not isinstance(
                    payload, Mapping
                ) or not _artifact_has_passing_helper_result(
                    payload,
                    helper=spec.name,
                ):
                    problems.append(f"{prefix} artifact lacks a passing helper result")
                has_callsite_receipt = isinstance(
                    payload, Mapping
                ) and _artifact_has_production_callsite_receipt(
                    payload,
                    helper=spec.name,
                )
                if evidence_group == "gpu_evidence" and not has_callsite_receipt:
                    problems.append(
                        f"{prefix} artifact lacks a stable production call-site receipt"
                    )
    try:
        callable_source = _callable_source_path(str(ref.callable_path))
    except (ImportError, AttributeError, ValueError) as exc:
        problems.append(f"{prefix} callable cannot be resolved: {exc}")
    else:
        if _sha256_path(callable_source) != ref.callable_source_sha256:
            problems.append(f"{prefix} callable source hash mismatch")
    if evidence_group == "gpu_evidence":
        required_scopes = set(_CUDA_INDUCTOR_GPU_PROMOTION_REQUIREMENTS)
    elif evidence_group == "ad_evidence":
        required_scopes = set()
        unknown_scopes = sorted(
            set(ref.validated_scopes) - set(_COMPILE_AD_EVIDENCE_SCOPES)
        )
        if unknown_scopes:
            problems.append(f"{prefix} has unknown validated scopes {unknown_scopes}")
    else:
        required_scopes = set(_CPU_FORWARD_INDUCTOR_EVIDENCE)
    missing_scopes = sorted(required_scopes - set(ref.validated_scopes))
    if missing_scopes:
        problems.append(f"{prefix} missing validated scopes {missing_scopes}")
    return tuple(problems)


def _has_current_authorizing_scope_coverage(
    spec: CompileKernelSpec,
    values: Iterable[CompileEvidenceRef],
    *,
    evidence_group: str,
    device_type: str,
    required_scopes: Iterable[str],
) -> bool:
    """Admit authority only while its artifact, callable, and sources remain current."""
    required = set(required_scopes)
    return any(
        ref.authorizing
        and ref.device_type == device_type
        and required.issubset(ref.validated_scopes)
        and not _authorizing_evidence_problems(spec, ref, evidence_group=evidence_group)
        for ref in values
    )


@dataclass(frozen=True)
class CompileKernelSpec:
    """Single source of truth for dycore torch.compile policy."""

    name: str
    callable: str | None
    owning_module: str
    kind: str
    stage: str | None
    cpu_evidence: tuple[CompileEvidenceRef, ...]
    gpu_evidence: tuple[CompileEvidenceRef, ...]
    ad_evidence: tuple[CompileEvidenceRef, ...]
    expected_callsite: str
    numeric_callable: str | None = None
    allow_core: bool = False
    allow_hot: bool = False
    allow_full: bool = False
    allow_staged_cuda: bool = False
    compiled_reverse_ad_policy: str = "blocked"
    policy_visible: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "callable": self.callable,
            "owning_module": self.owning_module,
            "kind": self.kind,
            "stage": self.stage,
            "cpu_evidence": sorted(_evidence_ids(self.cpu_evidence)),
            "gpu_evidence": sorted(_evidence_ids(self.gpu_evidence)),
            "ad_evidence": sorted(_evidence_ids(self.ad_evidence)),
            "cpu_evidence_refs": [item.as_dict() for item in self.cpu_evidence],
            "gpu_evidence_refs": [item.as_dict() for item in self.gpu_evidence],
            "ad_evidence_refs": [item.as_dict() for item in self.ad_evidence],
            "expected_callsite": self.expected_callsite,
            "numeric_callable": self.numeric_callable,
            "allow_core": self.allow_core,
            "allow_hot": self.allow_hot,
            "allow_full": self.allow_full,
            "allow_staged_cuda": self.allow_staged_cuda,
            "compiled_reverse_ad_policy": self.compiled_reverse_ad_policy,
            "policy_visible": self.policy_visible,
        }


def _compile_kernel_spec(
    name: str,
    *,
    callable: str | None,
    owning_module: str,
    kind: str,
    expected_callsite: str,
    numeric_callable: str | None = None,
    stage: str | None = None,
    cpu_evidence: tuple[EvidenceInput, ...] = (),
    gpu_evidence: tuple[EvidenceInput, ...] = (),
    ad_evidence: tuple[EvidenceInput, ...] = (),
    allow_core: bool = False,
    allow_hot: bool = False,
    allow_full: bool = False,
    allow_staged_cuda: bool = False,
    compiled_reverse_ad_policy: str = "blocked",
    policy_visible: bool = True,
) -> CompileKernelSpec:
    return CompileKernelSpec(
        name=name,
        callable=callable,
        owning_module=owning_module,
        kind=kind,
        stage=stage,
        cpu_evidence=_evidence_refs(cpu_evidence),
        gpu_evidence=_evidence_refs(gpu_evidence),
        ad_evidence=_evidence_refs(ad_evidence),
        expected_callsite=expected_callsite,
        numeric_callable=numeric_callable or callable,
        allow_core=allow_core,
        allow_hot=allow_hot,
        allow_full=allow_full,
        allow_staged_cuda=allow_staged_cuda,
        compiled_reverse_ad_policy=compiled_reverse_ad_policy,
        policy_visible=policy_visible,
    )


def _candidate_helper_spec(
    name: str,
    *,
    callable: str,
    owning_module: str,
    expected_callsite: str,
    stage: str,
    cpu_evidence: tuple[EvidenceInput, ...] = _CPU_INDUCTOR_EVIDENCE,
    gpu_evidence: tuple[EvidenceInput, ...] = _INSTALLED_GPU_EVIDENCE,
    ad_evidence: tuple[EvidenceInput, ...] = (),
    allow_hot: bool = True,
    allow_staged_cuda: bool = True,
    policy_visible: bool = True,
) -> CompileKernelSpec:
    return _compile_kernel_spec(
        name,
        callable=callable,
        owning_module=owning_module,
        kind="pure_numeric",
        stage=stage,
        cpu_evidence=cpu_evidence,
        gpu_evidence=gpu_evidence,
        ad_evidence=ad_evidence,
        expected_callsite=expected_callsite,
        allow_hot=allow_hot,
        allow_full=True,
        allow_staged_cuda=allow_staged_cuda,
        policy_visible=policy_visible,
    )


_COMPILE_KERNEL_SPECS: tuple[CompileKernelSpec, ...] = (
    _compile_kernel_spec(
        "vertical_implicit_coefficients_vert_imp_coefs",
        callable="mpas_pytorch.dynamics.vert_imp_coefs.atm_compute_vert_imp_coefs_work",
        owning_module="vert_imp_coefs",
        kind="pure_numeric",
        cpu_evidence=(
            "cpu_compact_forward_consistency",
            "cpu_compact_compiled_consistency",
            "cpu_compact_scan_consistency",
            "cpu_compact_reverse_ad",
            "cpu_compact_forward_ad",
        ),
        expected_callsite=(
            "time_integration._atm_srk3_impl:pre-dynamics vertical implicit coefficients. "
            "The compact-active-patch coefficient path is the no-grad CPU compile candidate; "
            "public padded packing is outside the compiled numeric region and delayed "
            "until after acoustic-step consumes compact patches. Thomas factorization is the "
            "default; scan factorization remains opt-in because it is not bitwise "
            "equivalent. AD stays eager because compiled JVP is blocked by a PyTorch "
            "AOT donated-buffer limitation."
        ),
        allow_hot=False,
        allow_full=True,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_divergence_damping",
        callable="mpas_pytorch.dynamics.divergence_damping.atm_divergence_damping_3d",
        owning_module="divergence_damping",
        kind="orchestration_blocked",
        expected_callsite=(
            "time_integration._atm_srk3_impl:divergence damping wrapper. The wrapper "
            "owns orchestration; the pure active-edge region is the compile candidate"
        ),
        policy_visible=False,
    ),
    _candidate_helper_spec(
        "dynamics_tendency_divergence_damping_region",
        callable="mpas_pytorch.dynamics.divergence_damping._divergence_damping_region",
        owning_module="divergence_damping",
        expected_callsite="divergence_damping.atm_divergence_damping_3d:active-edge damping region",
        stage="small_step_perturbation_divergence",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
        allow_hot=False,
    ),
    _compile_kernel_spec(
        "small_step_perturbation_moist_coefficients",
        callable="mpas_pytorch.dynamics.moist_coefficients.atm_compute_moist_coefficients",
        owning_module="moist_coefficients",
        kind="orchestration_blocked",
        expected_callsite=(
            "time_integration._atm_srk3_impl:moist coefficient wrapper. It remains "
            "eager because it owns qtot/cqw/cqu allocation"
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "small_step_perturbation_smlstep_pert",
        callable="mpas_pytorch.dynamics.smlstep_pert.atm_set_smlstep_pert_variables_work",
        owning_module="smlstep_pert",
        kind="orchestration_blocked",
        expected_callsite=(
            "time_integration._atm_srk3_impl:small-step perturbation wrapper; "
            "the wrapper stays orchestration-blocked and the pure w-tendency region "
            "is the compile candidate"
        ),
        policy_visible=False,
    ),
    _candidate_helper_spec(
        "small_step_perturbation_smlstep_pert_w_tendency_region",
        callable="mpas_pytorch.dynamics.smlstep_pert._small_step_perturbation_smlstep_pert_w_tendency_region",
        owning_module="smlstep_pert",
        expected_callsite="smlstep_pert.atm_set_smlstep_pert_variables_work:active-cell w-tendency region",
        stage="small_step_perturbation_divergence",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
        allow_hot=False,
    ),
    _candidate_helper_spec(
        "diagnostic_solve_solve_diagnostics",
        callable="mpas_pytorch.dynamics.solve_diagnostics._diagnostic_solve_numeric_composition",
        owning_module="solve_diagnostics",
        expected_callsite=(
            "solve_diagnostics.DiagnosticSolveNumericPlan:prebound diagnostics composition; "
            "scratch stays compact and public padding is packed once inside the "
            "pure compiled boundary; workspace ownership stays outside"
        ),
        stage="diagnostic_solve_diagnostics",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
        allow_hot=True,
    ),
    _candidate_helper_spec(
        "explicit_scalar_transport_pattern_c_flux_region",
        callable="mpas_pytorch.dynamics.advance_scalars._explicit_scalar_transport_pattern_c_flux_region",
        owning_module="advance_scalars",
        expected_callsite="advance_scalars.atm_advance_scalars_work:pattern-c flux region",
        stage="explicit_scalar_transport",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
        allow_staged_cuda=False,
        policy_visible=False,
    ),
    _candidate_helper_spec(
        "explicit_scalar_transport_horizontal_flux_divergence",
        callable="mpas_pytorch.dynamics.advance_scalars._explicit_scalar_transport_horizontal_flux_divergence_region",
        owning_module="advance_scalars",
        expected_callsite="advance_scalars.atm_advance_scalars_work:horizontal flux divergence region",
        stage="explicit_scalar_transport",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
        allow_staged_cuda=False,
        policy_visible=False,
    ),
    _candidate_helper_spec(
        "explicit_scalar_transport_scalar_tend_column_region",
        callable="mpas_pytorch.dynamics.advance_scalars._explicit_scalar_transport_scalar_tend_column_region",
        owning_module="advance_scalars",
        expected_callsite="advance_scalars.atm_advance_scalars_work:horizontal tendency column scaling region",
        stage="explicit_scalar_transport",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
        allow_staged_cuda=False,
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "explicit_scalar_transport_vertical_flux_divergence_region",
        callable="mpas_pytorch.dynamics.advance_scalars._explicit_scalar_transport_vertical_flux_divergence_region",
        owning_module="advance_scalars",
        kind="pure_numeric",
        expected_callsite=(
            "advance_scalars.atm_advance_scalars_work:legacy vertical flux "
            "divergence wrapper; kept eager because compilation changes arithmetic "
            "association. Production uses boundary-flux and divergence-assembly "
            "split helpers plus eager interior third-order flux"
        ),
        policy_visible=False,
    ),
    _candidate_helper_spec(
        "explicit_scalar_transport_vertical_boundary_flux_region",
        callable="mpas_pytorch.dynamics.advance_scalars._explicit_scalar_transport_vertical_boundary_flux_region",
        owning_module="advance_scalars",
        expected_callsite="advance_scalars.atm_advance_scalars_work:bottom/top vertical interface fluxes",
        stage="explicit_scalar_transport",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
        allow_staged_cuda=False,
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "explicit_scalar_transport_vertical_interior_flux_region",
        callable="mpas_pytorch.dynamics.advance_scalars._explicit_scalar_transport_vertical_interior_flux_region",
        owning_module="advance_scalars",
        kind="pure_numeric",
        cpu_evidence=_CPU_AD_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        expected_callsite=(
            "advance_scalars.atm_advance_scalars_work:interior third-order "
            "vertical flux. Its compiled form is not bitwise equivalent, so the "
            "helper remains policy-hidden"
        ),
        policy_visible=False,
    ),
    _candidate_helper_spec(
        "explicit_scalar_transport_vertical_flux_divergence_from_fluxes_region",
        callable="mpas_pytorch.dynamics.advance_scalars._explicit_scalar_transport_vertical_flux_divergence_from_fluxes_region",
        owning_module="advance_scalars",
        expected_callsite="advance_scalars.atm_advance_scalars_work:full-level vertical divergence assembly from precomputed interface fluxes",
        stage="explicit_scalar_transport",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
        allow_staged_cuda=False,
        policy_visible=False,
    ),
    _candidate_helper_spec(
        "explicit_scalar_transport_final_scalar_update_region",
        callable="mpas_pytorch.dynamics.advance_scalars._explicit_scalar_transport_final_scalar_update_region",
        owning_module="advance_scalars",
        expected_callsite="advance_scalars.atm_advance_scalars_work:final scalar update region",
        stage="explicit_scalar_transport",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
        allow_staged_cuda=False,
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "explicit_scalar_transport_numeric_composition",
        callable="mpas_pytorch.dynamics.advance_scalars._explicit_scalar_transport_numeric_composition",
        owning_module="advance_scalars",
        kind="pure_numeric",
        expected_callsite=(
            "canonical explicit-scalar-transport composition; whole-region compilation changes the "
            "interior third-order vertical-flux association, so production compiles "
            "the horizontal and vertical-finish boundaries around the eager interior "
            "flux"
        ),
        cpu_evidence=_CPU_AD_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        policy_visible=False,
    ),
    _candidate_helper_spec(
        "explicit_scalar_transport_horizontal_transport_composition",
        callable="mpas_pytorch.dynamics.advance_scalars._explicit_scalar_transport_horizontal_transport_composition",
        owning_module="advance_scalars",
        expected_callsite=(
            "advance_scalars.ExplicitScalarTransportNumericPlan:runner-bound Pattern-C flux plus "
            "ordered horizontal divergence and tendency"
        ),
        stage="explicit_scalar_transport",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
    ),
    _candidate_helper_spec(
        "explicit_scalar_transport_vertical_finish_region",
        callable="mpas_pytorch.dynamics.advance_scalars._explicit_scalar_transport_vertical_finish_region",
        owning_module="advance_scalars",
        expected_callsite=(
            "advance_scalars.ExplicitScalarTransportNumericPlan:runner-bound boundary flux, vertical "
            "divergence, and scalar update after eager ordered interior flux"
        ),
        stage="explicit_scalar_transport",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
    ),
    _compile_kernel_spec(
        "explicit_scalar_transport_scalar_update_region",
        callable="mpas_pytorch.dynamics.advance_scalars._explicit_scalar_transport_scalar_update_region",
        owning_module="advance_scalars",
        kind="pure_numeric",
        expected_callsite=(
            "advance_scalars.atm_advance_scalars_work:legacy monolithic scalar "
            "update wrapper; blocked because compilation is not numerically "
            "equivalent. Production uses the split scalar_tend_column, "
            "vertical_flux_divergence, and final_scalar_update regions instead"
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "explicit_scalar_transport_advance_scalars",
        callable="mpas_pytorch.dynamics.advance_scalars.atm_advance_scalars_work",
        owning_module="advance_scalars",
        kind="orchestration_blocked",
        expected_callsite="time_integration._atm_srk3_impl:coupled explicit scalar transport wrapper",
    ),
    _compile_kernel_spec(
        "explicit_scalar_transport_advance_scalars_transport",
        callable="mpas_pytorch.dynamics.advance_scalars.atm_advance_scalars_work",
        owning_module="advance_scalars",
        kind="orchestration_blocked",
        expected_callsite="time_integration._atm_srk3_impl:split explicit scalar transport wrapper",
    ),
    _compile_kernel_spec(
        "monotonic_scalar_transport_pattern_c_flux_region",
        callable="mpas_pytorch.dynamics.advance_scalars_mono._monotonic_scalar_transport_pattern_c_flux_region",
        owning_module="advance_scalars_mono",
        kind="pure_numeric",
        expected_callsite=(
            "validation-only decomposition of the former sectioned monotonic-scalar-transport path; "
            "production forward and AD use the streaming tile region"
        ),
        stage="monotonic_scalar_transport",
        cpu_evidence=_CPU_INDUCTOR_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        gpu_evidence=(),
        policy_visible=False,
    ),
    _candidate_helper_spec(
        "monotonic_scalar_transport_streaming_scalar_transport_region",
        callable="mpas_pytorch.dynamics.advance_scalars_mono._monotonic_scalar_transport_streaming_scalar_transport_tile_numeric",
        owning_module="advance_scalars_mono",
        expected_callsite=(
            "advance_scalars_mono._atm_advance_scalars_mono_work_no_chunk:"
            "per-tile streaming scalar transport streaming scalar transport numeric call; scalar "
            "chunk/tile loops and concatenation remain outside compile"
        ),
        stage="monotonic_scalar_transport",
        cpu_evidence=_CPU_INDUCTOR_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        gpu_evidence=(),
    ),
    _compile_kernel_spec(
        "monotonic_scalar_transport_upwind_scale_prep_region",
        callable="mpas_pytorch.dynamics.advance_scalars_mono._monotonic_scalar_transport_upwind_scale_prep_region",
        owning_module="advance_scalars_mono",
        kind="pure_numeric",
        expected_callsite=(
            "validation-only decomposition of the former sectioned monotonic-scalar-transport path; "
            "production forward and AD use the streaming tile region"
        ),
        stage="monotonic_scalar_transport",
        cpu_evidence=_CPU_INDUCTOR_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        gpu_evidence=(),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "monotonic_scalar_transport_limiter_flux_region",
        callable="mpas_pytorch.dynamics.advance_scalars_mono._monotonic_scalar_transport_limiter_flux_region",
        owning_module="advance_scalars_mono",
        kind="pure_numeric",
        expected_callsite=(
            "validation-only decomposition of the former sectioned monotonic-scalar-transport path; "
            "it cannot stand in for production streaming-tile evidence"
        ),
        stage="monotonic_scalar_transport",
        cpu_evidence=_CPU_INDUCTOR_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "monotonic_scalar_transport_final_update_region",
        callable="mpas_pytorch.dynamics.advance_scalars_mono._monotonic_scalar_transport_final_update_region",
        owning_module="advance_scalars_mono",
        kind="pure_numeric",
        expected_callsite=(
            "validation-only decomposition of the former sectioned monotonic-scalar-transport path; "
            "production forward and AD use the streaming tile region"
        ),
        stage="monotonic_scalar_transport",
        cpu_evidence=_CPU_INDUCTOR_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        gpu_evidence=(),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "monotonic_scalar_transport_advance_scalars_mono",
        callable="mpas_pytorch.dynamics.advance_scalars_mono.atm_advance_scalars_mono_work",
        owning_module="advance_scalars_mono",
        kind="orchestration_blocked",
        expected_callsite="time_integration._atm_srk3_impl:coupled monotonic scalar wrapper",
    ),
    _compile_kernel_spec(
        "monotonic_scalar_transport_advance_scalars_mono_transport",
        callable="mpas_pytorch.dynamics.advance_scalars_mono.atm_advance_scalars_mono_work",
        owning_module="advance_scalars_mono",
        kind="orchestration_blocked",
        expected_callsite="time_integration._atm_srk3_impl:split monotonic scalar wrapper",
    ),
    _compile_kernel_spec(
        "acoustic_step_acoustic_cell_flux_slots",
        callable="mpas_pytorch.dynamics.acoustic_step._sum_acoustic_cell_flux_slots",
        owning_module="acoustic_step",
        kind="pure_numeric",
        cpu_evidence=_CPU_INDUCTOR_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        expected_callsite=(
            "nested/conditional acoustic-step cell-flux reducer used inside "
            "acoustic_step._acoustic_step_acoustic_flux_solve_region and direct helper "
            "validation; not a standalone staged CUDA callsite"
        ),
        policy_visible=True,
    ),
    _candidate_helper_spec(
        "acoustic_step_acoustic_flux_solve_region",
        callable="mpas_pytorch.dynamics.acoustic_step._acoustic_step_acoustic_flux_solve_numeric",
        owning_module="acoustic_step",
        expected_callsite=(
            "time_integration.execute_rk_step builds a runner-owned "
            "AcousticSolvePlan that flattens topology once and invokes the "
            "dynamic-tensor-only acoustic flux/RHS/solve/post-solve region from "
            "acoustic_step.atm_advance_acoustic_step_work. The default production "
            "policy keeps the Thomas solve for "
            "strict CPU gates; scan is opt-in until a proven-drift budget artifact "
            "authorizes it."
        ),
        stage="acoustic_step",
        ad_evidence=_CPU_AD_EVIDENCE,
        allow_hot=False,
    ),
    _candidate_helper_spec(
        "acoustic_step_horizontal_momentum_update",
        callable="mpas_pytorch.dynamics.acoustic_step._acoustic_step_horizontal_momentum_update",
        owning_module="acoustic_step",
        expected_callsite="acoustic_step.atm_advance_acoustic_step_work:horizontal momentum update",
        stage="acoustic_step",
        ad_evidence=_CPU_AD_EVIDENCE,
    ),
    _compile_kernel_spec(
        "acoustic_step_acoustic_step",
        callable="mpas_pytorch.dynamics.acoustic_step.atm_advance_acoustic_step_work",
        owning_module="acoustic_step",
        kind="orchestration_blocked",
        expected_callsite="time_integration._atm_srk3_impl:acoustic substep wrapper",
    ),
    _compile_kernel_spec(
        "large_step_recovery_recover_large_step",
        callable="mpas_pytorch.dynamics.recover_large_step.atm_recover_large_step_variables_work",
        owning_module="recover_large_step",
        kind="orchestration_blocked",
        expected_callsite="time_integration._atm_srk3_impl:large-step recovery wrapper",
    ),
    _candidate_helper_spec(
        "large_step_recovery_recovery_region",
        callable="mpas_pytorch.dynamics.recover_large_step._large_step_recovery_recovery_numeric",
        owning_module="recover_large_step",
        expected_callsite="recover_large_step.atm_recover_large_step_variables_work:runner-owned compact parent recovery region",
        stage="large_step_recovery",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
        allow_hot=False,
    ),
    _compile_kernel_spec(
        "large_step_recovery_density_theta_recovery_region",
        callable=None,
        owning_module="recover_large_step",
        kind="legacy_alias",
        expected_callsite="direct compatibility helper; production compile uses large_step_recovery_recovery_region",
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "large_step_recovery_edge_momentum_recovery_region",
        callable=None,
        owning_module="recover_large_step",
        kind="legacy_alias",
        expected_callsite="direct compatibility helper; production compile uses large_step_recovery_recovery_region",
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "large_step_recovery_vertical_velocity_recovery_region",
        callable=None,
        owning_module="recover_large_step",
        kind="legacy_alias",
        expected_callsite="direct compatibility helper; production compile uses large_step_recovery_recovery_region",
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "large_step_recovery_pack_recovery_region",
        callable=None,
        owning_module="recover_large_step",
        kind="legacy_alias",
        expected_callsite="public packing boundary outside large_step_recovery_recovery_region",
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_normal_velocity_tendency_base_tendency",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "eager proof reference; production invokes "
            "dynamics_tendency_normal_velocity_tendency_normal_velocity_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_normal_velocity_tendency_vertical_tendency",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "eager proof reference; production invokes "
            "dynamics_tendency_normal_velocity_tendency_normal_velocity_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_normal_velocity_tendency_base_coupling",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "eager proof reference; production invokes "
            "dynamics_tendency_normal_velocity_tendency_normal_velocity_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_normal_velocity_tendency_pressure_curvature",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "eager proof reference; production invokes "
            "dynamics_tendency_normal_velocity_tendency_normal_velocity_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_normal_velocity_tendency_coupling_pressure_section",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "eager proof reference; production invokes "
            "dynamics_tendency_normal_velocity_tendency_normal_velocity_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_normal_velocity_tendency_normal_velocity_section",
        callable="mpas_pytorch.dynamics.dyn_tend._normal_velocity_tendency_normal_velocity_owned_section_region",
        owning_module="dyn_tend",
        kind="pure_numeric",
        expected_callsite=(
            "dyn_tend.atm_compute_dyn_tend_work:sole normal-velocity-tendency production region. It "
            "owns ordered vertical transport, PV-neighbor reduction, KE and "
            "divergence coupling, pressure/Euler updates, and optional "
            "curvature. Subhelpers are eager reference implementations. CPU "
            "compilation is allowed by the registered policy; GPU compilation "
            "remains fail-closed."
        ),
        stage="dynamics_tendency_normal_velocity_tendency",
        cpu_evidence=_CPU_INDUCTOR_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        gpu_evidence=(),
        allow_hot=True,
        allow_full=True,
        allow_staged_cuda=True,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_normal_velocity_mixing_normal_velocity_finalization_u_tendency_region",
        callable="mpas_pytorch.dynamics.dyn_tend._normal_velocity_mixing_normal_velocity_finalization_u_tendency_region",
        owning_module="dyn_tend",
        kind="pure_numeric",
        expected_callsite=(
            "dyn_tend._dynamics_tendency_canonical_composition:RK1 production parent for normal-velocity-mixing "
            "diffusion/mixing and normal-velocity-finalization momentum finalization. The required public "
            "Euler tendency is materialized once; normal-velocity-mixing scratch stays internal and "
            "the final normal-velocity-finalization tendency is produced in the same compile island. "
            "Individual normal-velocity-mixing/normal-velocity-finalization helpers remain equation references, not production "
            "compile targets. CUDA promotion requires exact-source full-mesh proof."
        ),
        stage="dynamics_tendency_normal_velocity_mixing",
        cpu_evidence=_CPU_INDUCTOR_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        gpu_evidence=(),
        allow_hot=True,
        allow_full=True,
        allow_staged_cuda=True,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_normal_velocity_mixing_u_mixing_section",
        callable="mpas_pytorch.dynamics.dyn_tend._normal_velocity_mixing_u_mixing_section_region",
        owning_module="dyn_tend",
        kind="pure_numeric",
        expected_callsite=(
            "Equation reference nested by dynamics_tendency_normal_velocity_mixing_normal_velocity_finalization_u_tendency_region in normal "
            "production. Direct execution remains available for differential, AD, "
            "and debug-tap validation; it is not a production compile target."
        ),
        cpu_evidence=_CPU_INDUCTOR_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_normal_velocity_mixing_u_del2_diffusion",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "independent eager reference test, but production invokes and "
            "compiles dynamics_tendency_normal_velocity_mixing_u_mixing_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_normal_velocity_mixing_u_del4_diffusion",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "independent eager reference test, but production invokes and "
            "compiles dynamics_tendency_normal_velocity_mixing_u_mixing_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_normal_velocity_mixing_u_vertical_mix_update",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "independent eager reference test, but production invokes and "
            "compiles dynamics_tendency_normal_velocity_mixing_u_mixing_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_normal_velocity_finalization_u_final_section",
        callable="mpas_pytorch.dynamics.dyn_tend._normal_velocity_finalization_u_final_section_region",
        owning_module="dyn_tend",
        kind="pure_numeric",
        expected_callsite=(
            "Equation reference nested by dynamics_tendency_normal_velocity_mixing_normal_velocity_finalization_u_tendency_region on RK1 "
            "and called directly on RK2/RK3. It remains the production compile "
            "target for the RK2/RK3 tail while the parent owns the hot RK1 handoff."
        ),
        stage="dynamics_tendency_normal_velocity_mixing",
        cpu_evidence=_CPU_INDUCTOR_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        gpu_evidence=(),
        allow_hot=True,
        allow_full=True,
        allow_staged_cuda=True,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_normal_velocity_finalization_u_rayleigh_damp",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired conditional compile name. The equation helper remains an "
            "eager reference; production invokes dynamics_tendency_normal_velocity_finalization_u_final_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_vertical_velocity_horizontal_advection_advective_cell_tendency",
        callable="mpas_pytorch.dynamics.dyn_tend._vertical_velocity_horizontal_advection_w_advective_cell_tendency_region",
        owning_module="dyn_tend",
        kind="pure_numeric",
        expected_callsite=(
            "nested helper called from _vertical_velocity_horizontal_advection_w_hadv_section_region; production "
            "no-grad forward compiles dynamics_tendency_vertical_velocity_horizontal_advection_w_hadv_section unless this helper "
            "receives a standalone runtime callsite"
        ),
        stage="dynamics_tendency_vertical_velocity_horizontal_advection_theta_horizontal_advection",
        cpu_evidence=_CPU_INDUCTOR_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        policy_visible=False,
    ),
    _candidate_helper_spec(
        "dynamics_tendency_vertical_velocity_horizontal_advection_w_hadv_section",
        callable="mpas_pytorch.dynamics.dyn_tend._vertical_velocity_horizontal_advection_w_hadv_owned_section_region",
        owning_module="dyn_tend",
        expected_callsite="dyn_tend.atm_compute_dyn_tend_work:vertical-velocity-horizontal-advection w horizontal advection section",
        stage="dynamics_tendency_vertical_velocity_horizontal_advection_theta_horizontal_advection",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_theta_horizontal_advection_advective_cell_tendency",
        callable="mpas_pytorch.dynamics.dyn_tend._theta_horizontal_advection_theta_advective_cell_tendency_region",
        owning_module="dyn_tend",
        kind="pure_numeric",
        expected_callsite=(
            "nested helper called from _theta_horizontal_advection_theta_hadv_section_region; production "
            "no-grad forward compiles dynamics_tendency_theta_horizontal_advection_theta_hadv_section unless this "
            "helper receives a standalone runtime callsite"
        ),
        stage="dynamics_tendency_vertical_velocity_horizontal_advection_theta_horizontal_advection",
        cpu_evidence=_CPU_INDUCTOR_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        policy_visible=False,
    ),
    _candidate_helper_spec(
        "dynamics_tendency_theta_horizontal_advection_theta_hadv_section",
        callable="mpas_pytorch.dynamics.dyn_tend._theta_horizontal_advection_theta_hadv_owned_section_region",
        owning_module="dyn_tend",
        expected_callsite="dyn_tend.atm_compute_dyn_tend_work:theta-horizontal-advection theta horizontal advection section",
        stage="dynamics_tendency_vertical_velocity_horizontal_advection_theta_horizontal_advection",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
    ),
    _candidate_helper_spec(
        "dynamics_tendency_theta_horizontal_advection_theta_finalization_theta_tendency_region",
        callable="mpas_pytorch.dynamics.dyn_tend._theta_horizontal_advection_theta_finalization_theta_tendency_region",
        owning_module="dyn_tend",
        expected_callsite=(
            "dyn_tend.atm_compute_dyn_tend_work:contiguous theta-horizontal-advection horizontal "
            "advection through theta-finalization final theta tendency parent. Production "
            "returns only final solve-row theta/rthdynten and the optional "
            "active-cell Euler patch. CUDA staging remains fail-closed until "
            "full-mesh production evidence exists."
        ),
        ad_evidence=_CPU_AD_EVIDENCE,
        stage="combined_hot",
        gpu_evidence=(),
    ),
    _compile_kernel_spec(
        "dynamics_tendency_vertical_velocity_horizontal_mixing_w_horizontal_mix_section",
        callable=(
            "mpas_pytorch.dynamics.dyn_tend._vertical_velocity_horizontal_mixing_w_horizontal_mix_section_region"
        ),
        owning_module="dyn_tend",
        kind="pure_numeric",
        expected_callsite=(
            "Equation/reference vertical-velocity-horizontal-mixing helper and explicit isolation target. Normal "
            "RK1 production invokes dynamics_tendency_vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency_region instead."
        ),
        cpu_evidence=_CPU_INDUCTOR_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        allow_full=True,
    ),
    _candidate_helper_spec(
        "dynamics_tendency_vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency_region",
        callable="mpas_pytorch.dynamics.dyn_tend._vertical_velocity_horizontal_mixing_vertical_velocity_vertical_transport_w_tendency_region",
        owning_module="dyn_tend",
        expected_callsite=(
            "dyn_tend.atm_compute_dyn_tend_work:contiguous RK1 vertical-velocity-horizontal-mixing horizontal "
            "mixing through vertical-velocity-vertical-transport vertical/final w tendency parent; production "
            "returns only final solve-row w and final active-cell Euler state"
        ),
        stage="combined_hot",
        gpu_evidence=(),
        ad_evidence=_CPU_AD_EVIDENCE,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_vertical_velocity_vertical_transport_w_vertical_boundary_fluxes",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "eager proof reference; production invokes dynamics_tendency_vertical_velocity_vertical_transport_w_vertical_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_vertical_velocity_vertical_transport_w_vertical_flux_delta",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "eager proof reference; production invokes dynamics_tendency_vertical_velocity_vertical_transport_w_vertical_section."
        ),
        policy_visible=False,
    ),
    _candidate_helper_spec(
        "dynamics_tendency_theta_horizontal_mixing_theta_horizontal_mix_section",
        callable=(
            "mpas_pytorch.dynamics.dyn_tend._theta_horizontal_mixing_theta_horizontal_mix_section_region"
        ),
        owning_module="dyn_tend",
        expected_callsite=(
            "dyn_tend.atm_compute_dyn_tend_work:theta-horizontal-mixing complete theta "
            "horizontal-mixing section; owns ordered del2 reduction, turbulent "
            "tendency, optional del4 correction, and compact patch assembly"
        ),
        stage="combined_hot",
        ad_evidence=_CPU_AD_EVIDENCE,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_vertical_velocity_vertical_transport_w_pointwise_update",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "eager proof reference; production invokes dynamics_tendency_vertical_velocity_vertical_transport_w_vertical_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_vertical_velocity_vertical_transport_w_vertical_section",
        callable="mpas_pytorch.dynamics.dyn_tend._vertical_velocity_vertical_transport_w_vertical_section_region",
        owning_module="dyn_tend",
        kind="pure_numeric",
        expected_callsite=(
            "dyn_tend.atm_compute_dyn_tend_work:sole vertical-velocity-vertical-transport production region. It "
            "owns boundary/interior vertical fluxes and pointwise w updates; "
            "the subhelpers are eager proof references only. Eager "
            "regionalization and AD/JVP are supported. CPU Inductor is available "
            "only under explicit full scope; CUDA remains fail-closed."
        ),
        stage=None,
        cpu_evidence=(
            "cpu_eager_formula_consistency",
            "cpu_compiled_helper_consistency",
            "cpu_production_scope_drift_unproven",
            "cpu_multistep_scope_drift_unproven",
        ),
        ad_evidence=_CPU_AD_EVIDENCE,
        allow_full=True,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_theta_vertical_advection_diabatic_theta_pointwise_update",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "eager proof reference; production invokes "
            "dynamics_tendency_theta_vertical_advection_diabatic_theta_vertical_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_theta_vertical_advection_diabatic_theta_vertical_flux_delta",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "eager proof reference; production invokes "
            "dynamics_tendency_theta_vertical_advection_diabatic_theta_vertical_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_theta_vertical_advection_diabatic_theta_vertical_section",
        callable="mpas_pytorch.dynamics.dyn_tend._theta_vertical_advection_diabatic_theta_vertical_section_region",
        owning_module="dyn_tend",
        kind="pure_numeric",
        expected_callsite=(
            "dyn_tend.atm_compute_dyn_tend_work:sole theta-vertical-advection-diabatic production region. It "
            "owns boundary/interior vertical fluxes, diabatic diagnostics, and "
            "pointwise theta updates; the subhelpers are eager proof references "
            "only. Eager regionalization and AD/JVP are supported. CPU Inductor "
            "is available only under explicit full scope; CUDA remains fail-closed."
        ),
        stage="combined_hot",
        cpu_evidence=(
            "cpu_eager_formula_consistency",
            "cpu_compiled_helper_consistency",
            "cpu_production_scope_drift_unproven",
            "cpu_multistep_scope_drift_unproven",
        ),
        ad_evidence=_CPU_AD_EVIDENCE,
        allow_full=True,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_theta_vertical_advection_diabatic_theta_vertical_boundary_fluxes",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "eager proof reference; production invokes "
            "dynamics_tendency_theta_vertical_advection_diabatic_theta_vertical_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_theta_vertical_mixing_theta_finalization_theta_final_section",
        callable="mpas_pytorch.dynamics.dyn_tend._theta_vertical_mixing_theta_finalization_theta_final_section_region",
        owning_module="dyn_tend",
        kind="pure_numeric",
        expected_callsite=(
            "dyn_tend.atm_compute_dyn_tend_work:sole theta-vertical-mixing/theta-finalization production region. "
            "It owns optional vertical theta mixing, compact Euler-patch "
            "composition, final Euler accumulation, and optional physics. "
            "CPU eager/Inductor, reverse AD, and JVP are supported. CUDA remains "
            "fail-closed."
        ),
        stage="combined_hot",
        cpu_evidence=_CPU_INDUCTOR_EVIDENCE,
        ad_evidence=_CPU_AD_EVIDENCE,
        gpu_evidence=(),
        allow_hot=True,
        allow_full=True,
        allow_staged_cuda=True,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_theta_vertical_mixing_theta_vertical_mix_update",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "eager reference; production invokes dynamics_tendency_theta_vertical_mixing_theta_finalization_theta_final_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_theta_finalization_theta_final_section",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "eager reference; production invokes dynamics_tendency_theta_vertical_mixing_theta_finalization_theta_final_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_normal_velocity_tendency_edge_neighbor_pv",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite=(
            "Retired standalone compile name. The equation helper remains an "
            "eager proof reference; production invokes "
            "dynamics_tendency_normal_velocity_tendency_normal_velocity_section."
        ),
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_vertical_velocity_horizontal_advection_edge_advective_slots",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite="legacy policy name; no current regional helper callsite",
    ),
    _compile_kernel_spec(
        "dynamics_tendency_theta_horizontal_advection_local_cell_edge_flux",
        callable=None,
        owning_module="dyn_tend",
        kind="legacy_alias",
        expected_callsite="legacy policy name; no current regional helper callsite",
    ),
    _compile_kernel_spec(
        "dynamics_tendency_adjoint_normal_velocity_tendency_edge_neighbor_pv_backward",
        callable=(
            "mpas_pytorch.dynamics.dyn_tend._normal_velocity_tendency_edge_neighbor_pv_ordered_backward"
        ),
        owning_module="dyn_tend",
        kind="pure_numeric",
        ad_evidence=("cpu_ordered_edge_neighbor_reverse_ad",),
        expected_callsite=(
            "DynamicsTendencyADCompilePlan: _NormalVelocityTendencyEdgeNeighborPVFunction.backward; "
            "AD-only and never a forward compile target"
        ),
        compiled_reverse_ad_policy="cpu_experiment",
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_adjoint_normal_velocity_tendency_vertical_backward",
        callable="mpas_pytorch.dynamics.dyn_tend._normal_velocity_tendency_vertical_ordered_backward",
        owning_module="dyn_tend",
        kind="pure_numeric",
        ad_evidence=("cpu_ordered_vertical_reverse_ad",),
        expected_callsite=(
            "DynamicsTendencyADCompilePlan: _NormalVelocityTendencyVerticalTendencyFunction.backward; "
            "AD-only and never a forward compile target"
        ),
        compiled_reverse_ad_policy="cpu_experiment",
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_adjoint_edge_advective_cell_tendency_backward",
        callable=(
            "mpas_pytorch.dynamics.dyn_tend."
            "_edge_advective_cell_tendency_ordered_backward"
        ),
        owning_module="dyn_tend",
        kind="pure_numeric",
        ad_evidence=("cpu_ordered_advective_reverse_ad",),
        expected_callsite=(
            "DynamicsTendencyADCompilePlan: _EdgeAdvectiveCellTendencyFunction.backward; "
            "AD-only and never a forward compile target"
        ),
        compiled_reverse_ad_policy="cpu_experiment",
        policy_visible=False,
    ),
    _compile_kernel_spec(
        "dynamics_tendency_dyn_tend_whole_stage",
        callable="mpas_pytorch.dynamics.dyn_tend.atm_compute_dyn_tend_work",
        numeric_callable=(
            "mpas_pytorch.dynamics.dyn_tend._dynamics_tendency_canonical_composition"
        ),
        owning_module="dyn_tend",
        kind="pure_numeric",
        expected_callsite=(
            "time_integration.execute_rk_step: runner-owned DynamicsTendencyNumericPlan binds "
            "97 static dimensions/topology/metric/namelist/control arguments and "
            "passes 41 evolving tensor arguments positionally through a generated "
            "equation-free explicit binder to the same atm_compute_dyn_tend_work "
            "entry-to-exit body. The numeric body returns ten compact active-row "
            "patches; the caller validates and packs public padded arrays after the "
            "compiled call. No functools.partial, keyword dictionary, or public "
            "full-field output pack crosses the compiled boundary. The caller fixes debug taps, "
            "section timers, workspace mutation, and nested helper compilation "
            "outside the graph. "
            "This target is CPU/full-scope and explicit-target only. Staged CUDA "
            "execution remains disabled."
        ),
        cpu_evidence=(
            "cpu_entry_exit_forward_consistency",
            "cpu_entry_exit_error_budget",
            "cpu_multistep_consistency",
        ),
        ad_evidence=(
            "cpu_entry_exit_adjoint_identity",
            _DYNAMICS_TENDENCY_WHOLE_STAGE_REVERSE_AOT_CPU_EVIDENCE,
        ),
        allow_full=True,
        allow_hot=False,
        allow_staged_cuda=False,
        compiled_reverse_ad_policy="cpu_experiment",
    ),
    _compile_kernel_spec(
        "dynamics_tendency_dyn_tend",
        callable="mpas_pytorch.dynamics.dyn_tend.atm_compute_dyn_tend_work",
        owning_module="dyn_tend",
        kind="orchestration_blocked",
        expected_callsite=(
            "time_integration._atm_srk3_impl:full dynamics tendency wrapper. "
            "Compilation changes numerical results, so the wrapper remains "
            "fail-closed; registered pure subsections provide the compiled path"
        ),
    ),
)


def _kernel_names_where(
    predicate: Callable[[CompileKernelSpec], bool],
) -> frozenset[str]:
    return frozenset(spec.name for spec in _COMPILE_KERNEL_SPECS if predicate(spec))


_COMPILE_KERNEL_REGISTRY = {spec.name: spec for spec in _COMPILE_KERNEL_SPECS}
_CORE_HOT_KERNELS = _kernel_names_where(
    lambda spec: spec.policy_visible and spec.allow_core
)
_HOT_KERNELS = _kernel_names_where(lambda spec: spec.policy_visible and spec.allow_hot)
_COMPILE_POLICY_KERNELS = _kernel_names_where(lambda spec: spec.policy_visible)
_ORCHESTRATION_COMPILE_BLOCKED_KERNELS = _kernel_names_where(
    lambda spec: spec.kind == "orchestration_blocked"
)
_NUMERIC_COMPILE_POLICY_KERNELS = _kernel_names_where(
    lambda spec: (
        spec.policy_visible
        and spec.kind != "orchestration_blocked"
        and (
            spec.allow_core
            or spec.allow_hot
            or spec.allow_full
            or spec.allow_staged_cuda
        )
    )
)
_CUDA_INDUCTOR_CANDIDATE_KERNELS = _kernel_names_where(
    lambda spec: spec.policy_visible and spec.allow_staged_cuda
)
_CUDA_INDUCTOR_CPU_PREQUALIFIED_KERNELS = _kernel_names_where(
    lambda spec: (
        spec.policy_visible
        and spec.allow_staged_cuda
        and set(_CPU_FORWARD_INDUCTOR_EVIDENCE).issubset(
            _evidence_ids(spec.cpu_evidence)
        )
        and set(_CPU_AD_EVIDENCE).issubset(_evidence_ids(spec.ad_evidence))
    )
)
_CUDA_INDUCTOR_GPU_VALIDATED_KERNELS = _kernel_names_where(
    _spec_has_authorizing_cuda_promotion_evidence
)
_CUDA_INDUCTOR_SIMPLE_KERNELS = _CUDA_INDUCTOR_GPU_VALIDATED_KERNELS
_CUDA_INDUCTOR_VALIDATED_ENVIRONMENTS: frozenset[tuple[str, str, str | None]] = (
    frozenset(
        (
            str(ref.torch_version),
            str(ref.cuda_version),
            str(ref.compute_capability),
        )
        for spec in _COMPILE_KERNEL_SPECS
        if spec.name in _CUDA_INDUCTOR_GPU_VALIDATED_KERNELS
        for ref in spec.gpu_evidence
        if ref.authorizing and ref.device_type == "cuda"
    )
)
_CUDA_INDUCTOR_GPU_VALIDATION_EVIDENCE: dict[tuple[str, str, str | None], list[str]] = {
    environment: sorted(
        {
            str(ref.artifact_path)
            for spec in _COMPILE_KERNEL_SPECS
            if spec.name in _CUDA_INDUCTOR_GPU_VALIDATED_KERNELS
            for ref in spec.gpu_evidence
            if ref.authorizing
            and ref.device_type == "cuda"
            and (
                str(ref.torch_version),
                str(ref.cuda_version),
                str(ref.compute_capability),
            )
            == environment
        }
    )
    for environment in _CUDA_INDUCTOR_VALIDATED_ENVIRONMENTS
}
_CUDA_INDUCTOR_STAGE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (
        stage,
        tuple(
            spec.name
            for spec in _COMPILE_KERNEL_SPECS
            if spec.stage == stage and spec.policy_visible and spec.allow_staged_cuda
        ),
    )
    for stage in _CUDA_INDUCTOR_STAGE_ORDER
)
_CUDA_INDUCTOR_POLICY_ALIASES = {
    "": "safe",
    "default": "safe",
    "1": "force",
    "true": "force",
    "yes": "force",
    "on": "force",
    "all": "force",
    "full": "force",
    "0": "off",
    "false": "off",
    "no": "off",
    "none": "off",
}


def set_compile_event_sink(sink: Callable[[dict[str, Any]], None] | None) -> None:
    """Install a process-local sink for torch.compile telemetry events."""
    global _COMPILE_EVENT_SINK
    with _COMPILE_EVENT_LOCK:
        _COMPILE_EVENT_SINK = sink


def clear_compile_event_sink() -> None:
    """Clear the process-local torch.compile telemetry sink."""
    set_compile_event_sink(None)


def compile_event_sink_enabled() -> bool:
    """Return whether compile call telemetry is currently being recorded."""
    return _COMPILE_EVENT_SINK is not None


def _cache_key_digest(key: object) -> str:
    text = repr(key).encode("utf-8", errors="replace")
    return hashlib.sha256(text).hexdigest()[:16]


def _json_safe(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return _tensor_signature(value)
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def _emit_compile_event(fn_name: str, phase: str, **payload: object) -> None:
    with _COMPILE_EVENT_LOCK:
        sink = _COMPILE_EVENT_SINK
    if sink is None:
        return

    event: dict[str, Any] = {
        "event": "torch_compile",
        "event_id": next(_COMPILE_EVENT_COUNTER),
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        "fn_name": fn_name,
        "phase": phase,
    }
    event.update({key: _json_safe(value) for key, value in payload.items()})
    try:
        sink(event)
    except (
        Exception  # noqa: BLE001
    ) as exc:  # pragma: no cover - observability must not change math.
        warnings.warn(
            f"torch.compile telemetry sink failed for {fn_name} ({exc})",
            RuntimeWarning,
            stacklevel=2,
        )


def _tensor_signature(value: torch.Tensor) -> dict[str, Any]:
    return {
        "kind": "tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
        "is_contiguous": bool(value.is_contiguous()),
    }


def _value_signature(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return _tensor_signature(value)
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "len": len(value),
            "keys": sorted(str(key) for key in value)[:48],
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "len": len(value),
            "items": [_value_signature(item) for item in list(value)[:16]],
            "omitted": max(0, len(value) - 16),
        }
    if isinstance(value, (str, int, float, bool, type(None))):
        return {"kind": type(value).__name__, "value": value}
    return {"kind": type(value).__name__}


def _argument_signature(
    args: tuple[object, ...],
    kwargs: dict[str, Any],
    *,
    limit: int = 64,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, value in enumerate(args):
        if len(items) >= limit:
            break
        items.append(
            {
                "name": f"arg{index}",
                "signature": _value_signature(value),
            }
        )
    if len(items) < limit:
        for name in sorted(kwargs):
            if len(items) >= limit:
                break
            items.append(
                {
                    "name": str(name),
                    "signature": _value_signature(kwargs[name]),
                }
            )
    omitted = len(args) + len(kwargs) - len(items)
    if omitted > 0:
        items.append({"name": "__omitted__", "count": omitted})
    return items


def _dynamo_counter_snapshot() -> dict[str, Any]:
    try:
        from torch._dynamo.utils import counters
    except Exception:  # noqa: BLE001
        return {}

    groups: dict[str, dict[str, int]] = {}
    for group, counter in counters.items():
        try:
            group_items = {
                str(key): int(value)
                for key, value in counter.items()
                if isinstance(value, (int, float))
            }
        except Exception:  # noqa: BLE001, S112
            continue
        if group_items:
            groups[str(group)] = group_items
    stats = groups.get("stats", {})
    graph_breaks = groups.get("graph_break", {})
    inductor = groups.get("inductor", {})
    return {
        "groups": groups,
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "graph_break_count": int(sum(graph_breaks.values())),
        "fxgraph_cache_hit": int(inductor.get("fxgraph_cache_hit", 0)),
        "fxgraph_cache_miss": int(inductor.get("fxgraph_cache_miss", 0)),
    }


def compile_timing_counter_snapshot() -> dict[str, Any]:
    """Take fixed-width counters suitable for a timed-region boundary.

    No event sink is installed and no cache is traversed. Dynamo/Inductor
    counters are read by fixed key, while MPAS counters advance only on wrapper
    creation, fallback, error, or specialization admission.
    """
    try:
        from torch._dynamo.utils import counters
    except Exception:  # noqa: BLE001
        counters = {}  # type: ignore[assignment]
    stats: Mapping[str, Any] = counters.get("stats", {})
    inductor: Mapping[str, Any] = counters.get("inductor", {})
    frames: Mapping[str, Any] = counters.get("frames", {})
    return {
        **{name: int(value) for name, value in _COMPILE_RUNTIME_COUNTERS.items()},
        "dynamo_unique_graphs": int(stats.get("unique_graphs", 0)),
        "dynamo_calls_captured": int(stats.get("calls_captured", 0)),
        "dynamo_frames_total": int(frames.get("total", 0)),
        "dynamo_frames_ok": int(frames.get("ok", 0)),
        "inductor_fxgraph_cache_hit": int(inductor.get("fxgraph_cache_hit", 0)),
        "inductor_fxgraph_cache_miss": int(inductor.get("fxgraph_cache_miss", 0)),
        "event_sink_enabled": compile_event_sink_enabled(),
    }


def compile_timing_counter_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare timed-boundary snapshots and fail closed on compile activity."""
    numeric_names = sorted(
        name for name in set(before) | set(after) if name != "event_sink_enabled"
    )
    delta = {
        name: int(after.get(name, 0)) - int(before.get(name, 0))
        for name in numeric_names
    }
    forbidden_names = {
        "wrapper_create",
        "wrapper_cache_miss",
        "eager_fallback",
        "autodiff_bypass",
        "compiled_call_error",
        "specialization_revision",
        "dynamo_unique_graphs",
        "dynamo_frames_total",
        "inductor_fxgraph_cache_miss",
    }
    forbidden = {
        name: value
        for name, value in delta.items()
        if name in forbidden_names and value != 0
    }
    failures: list[str] = []
    if bool(before.get("event_sink_enabled")) or bool(after.get("event_sink_enabled")):
        failures.append("per-call compile event sink was enabled at a timing boundary")
    if forbidden:
        failures.append(f"compile activity occurred inside timed compute: {forbidden}")
    return {
        "status": "pass" if not failures else "fail",
        "pass": not failures,
        "before": dict(before),
        "after": dict(after),
        "delta": delta,
        "forbidden_delta": forbidden,
        "failures": failures,
    }


def _counter_delta(
    after: dict[str, Any],
    before: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "unique_graphs",
        "graph_break_count",
        "fxgraph_cache_hit",
        "fxgraph_cache_miss",
    ):
        after_value = after.get(key, 0)
        before_value = before.get(key, 0)
        if isinstance(after_value, int) and isinstance(before_value, int):
            out[key] = after_value - before_value

    after_groups = after.get("groups", {})
    before_groups = before.get("groups", {})
    if isinstance(after_groups, dict) and isinstance(before_groups, dict):
        group_delta: dict[str, dict[str, int]] = {}
        for group in sorted(set(after_groups) | set(before_groups)):
            after_group = after_groups.get(group, {})
            before_group = before_groups.get(group, {})
            if not isinstance(after_group, dict) or not isinstance(before_group, dict):
                continue
            delta = {
                str(name): int(after_group.get(name, 0))
                - int(before_group.get(name, 0))
                for name in sorted(set(after_group) | set(before_group))
                if int(after_group.get(name, 0)) - int(before_group.get(name, 0)) != 0
            }
            if delta:
                group_delta[str(group)] = delta
        out["groups"] = group_delta
    return out


def _stringify_sequence(values: object) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    return [str(value) for value in values]


def _ops_per_graph_signature(values: object) -> list[list[str]]:
    if not isinstance(values, (list, tuple)):
        return []
    out: list[list[str]] = []
    for graph_ops in values:
        if isinstance(graph_ops, (list, tuple)):
            out.append([str(op) for op in graph_ops])
        else:
            out.append([str(graph_ops)])
    return out


def _guard_summary(values: object, *, limit: int = 48) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for guard in values[:limit]:
        out.append(
            {
                "name": str(getattr(guard, "name", "")),
                "source": str(getattr(guard, "source", "")),
                "create_fn_name": str(getattr(guard, "create_fn_name", "")),
                "guard_types": _json_safe(getattr(guard, "guard_types", None)),
                "code_list": _json_safe(getattr(guard, "code_list", None)),
            }
        )
    omitted = len(values) - len(out)
    if omitted > 0:
        out.append({"omitted_guard_count": int(omitted)})
    return out


def _counter_nested_value(value: object, keys: tuple[str, ...]) -> int:
    if isinstance(value, dict):
        total = 0
        for key, item in value.items():
            if str(key) in keys and isinstance(item, (int, float)):
                total += int(item)
            total += _counter_nested_value(item, keys)
        return total
    if isinstance(value, list):
        return sum(_counter_nested_value(item, keys) for item in value)
    return 0


def _recompile_indicator_count(delta: dict[str, Any]) -> int:
    """Return nonzero only for uncached compile work on this call.

    ``unique_graphs`` and stable graph-break counters are structural telemetry:
    they show how much Dynamo split the callable, but they do not prove the same
    specialization key recompiled. Same-key instability is represented by
    explicit recompiles or cache misses.
    """
    return _counter_nested_value(
        delta,
        (
            "fxgraph_cache_miss",
            "recompiles",
        ),
    )


def _compile_logging_snapshot() -> dict[str, Any]:
    return {
        "TORCH_LOGS": os.environ.get("TORCH_LOGS"),
        "MPAS_TORCH_COMPILE_EXPLAIN_CALLS": os.environ.get(
            "MPAS_TORCH_COMPILE_EXPLAIN_CALLS"
        ),
    }


def _compile_call_explain_payload(
    fn: CompiledFn,
    args: tuple[object, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    enabled = str(os.environ.get("MPAS_TORCH_COMPILE_EXPLAIN_CALLS", "")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return None
    try:
        from torch._dynamo import explain

        explained = explain(fn)(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    out_guards = getattr(explained, "out_guards", []) or []
    return {
        "status": "ok",
        "graph_count": int(getattr(explained, "graph_count", 0) or 0),
        "graph_break_count": int(getattr(explained, "graph_break_count", 0) or 0),
        "guard_count": len(out_guards),
        "guard_summary": _guard_summary(out_guards),
        "op_count": int(getattr(explained, "op_count", 0) or 0),
        "break_reasons": _stringify_sequence(getattr(explained, "break_reasons", [])),
    }


def explain_compile_kernel(
    fn_name: str,
    fn: CompiledFn,
    *args: object,
    mode: str,
    backend: str,
    specialization_key: object | None = None,
    **kwargs: object,
) -> dict[str, Any]:
    """
    Return offline Dynamo explain metadata for a helper preflight.

    This intentionally executes ``fn`` under ``torch._dynamo.explain`` and is
    therefore for controlled preflight probes, not live performance timing.
    """
    key = (
        fn_name,
        _compile_function_cache_key(fn),
        mode,
        backend,
        _normalize_specialization_key(specialization_key),
    )
    key_digest = _cache_key_digest(key)
    start = time.perf_counter()
    _emit_compile_event(
        fn_name,
        "preflight_explain_start",
        cache_key=key_digest,
        mode=mode,
        backend=backend,
        specialization_key=_normalize_specialization_key(specialization_key),
        argument_signature=_argument_signature(args, kwargs),
    )
    try:
        from torch._dynamo import explain

        explained = explain(fn)(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        result = {
            "event": "torch_compile_preflight_explain",
            "fn_name": fn_name,
            "status": "failed",
            "cache_key": key_digest,
            "mode": mode,
            "backend": backend,
            "specialization_key": list(
                _normalize_specialization_key(specialization_key)
            ),
            "explain_seconds": time.perf_counter() - start,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _emit_compile_event(
            fn_name,
            "preflight_explain_error",
            **{
                key: value
                for key, value in result.items()
                if key not in {"event", "fn_name"}
            },
        )
        return result

    out_guards = getattr(explained, "out_guards", []) or []
    result = {
        "event": "torch_compile_preflight_explain",
        "fn_name": fn_name,
        "status": "ok",
        "cache_key": key_digest,
        "mode": mode,
        "backend": backend,
        "specialization_key": list(_normalize_specialization_key(specialization_key)),
        "explain_seconds": time.perf_counter() - start,
        "graph_count": int(getattr(explained, "graph_count", 0) or 0),
        "graph_break_count": int(getattr(explained, "graph_break_count", 0) or 0),
        "guard_count": len(out_guards),
        "guard_summary": _guard_summary(out_guards),
        "op_count": int(getattr(explained, "op_count", 0) or 0),
        "break_reasons": _stringify_sequence(getattr(explained, "break_reasons", [])),
        "ops_per_graph": _ops_per_graph_signature(
            getattr(explained, "ops_per_graph", [])
        ),
        "compile_times": str(getattr(explained, "compile_times", "")),
        "argument_signature": _argument_signature(args, kwargs),
    }
    _emit_compile_event(
        fn_name,
        "preflight_explain_success",
        **{
            key: value
            for key, value in result.items()
            if key not in {"event", "fn_name"}
        },
    )
    return result


def compile_kernel_specs() -> tuple[CompileKernelSpec, ...]:
    """Return the ordered dycore compile-kernel policy registry."""
    return _COMPILE_KERNEL_SPECS


def compile_kernel_spec(name: str) -> CompileKernelSpec | None:
    """Return one compile-kernel spec, or None for an unknown policy name."""
    return _COMPILE_KERNEL_REGISTRY.get(str(name))


def _qualified_callable_path(fn: CompiledFn) -> str:
    return (
        f"{getattr(fn, '__module__', type(fn).__module__)}."
        f"{getattr(fn, '__qualname__', getattr(fn, '__name__', type(fn).__qualname__))}"
    )


def _resolve_registered_callable(path: str) -> CompiledFn:
    module_name, _, qualname = path.rpartition(".")
    if not module_name or not qualname:
        raise ValueError(f"invalid registered callable path: {path!r}")
    value: object = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise TypeError(f"registered compile target is not callable: {path!r}")
    return value


def _bound_payload_component(value: object) -> object:
    """Return a bounded identity summary for a runner-owned static closure."""
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "data_ptr": int(value.data_ptr()),
            "version": int(value._version),
        }
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "items": {
                str(key): _bound_payload_component(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            },
        }
    if isinstance(value, (tuple, list)):
        return {
            "kind": type(value).__name__,
            "items": [_bound_payload_component(item) for item in value],
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": type(value).__name__, "value": value}
    return {
        "kind": "object",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "identity": id(value),
    }


def _static_plan_binder_identity(
    fn: CompiledFn,
    *,
    registered_path: str,
    registered_callable: CompiledFn,
) -> dict[str, Any] | None:
    """Authenticate a fixed-signature closure emitted by a numeric-plan builder."""
    actual_module = getattr(fn, "__module__", None)
    registered_module = getattr(registered_callable, "__module__", None)
    if actual_module != registered_module:
        return None
    source_path = inspect.getsourcefile(fn)
    if source_path is None:
        return None
    first_line = int(fn.__code__.co_firstlineno)
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    enclosing_builders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("build_")
        and int(node.lineno) <= first_line <= int(node.end_lineno or node.lineno)
    ]
    if not enclosing_builders:
        return None
    owner = max(enclosing_builders, key=lambda node: int(node.lineno)).name
    signature = inspect.signature(fn)
    if any(
        parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    ):
        return None

    closure = inspect.getclosurevars(fn)
    referenced = {**closure.globals, **closure.nonlocals}
    canonical_names = sorted(
        name for name, value in referenced.items() if value is registered_callable
    )
    code_names = set(fn.__code__.co_names) | set(fn.__code__.co_freevars)
    if not canonical_names or not any(name in code_names for name in canonical_names):
        return None
    payload = {
        name: _bound_payload_component(value)
        for name, value in sorted(closure.nonlocals.items())
        if value is not registered_callable
    }
    if not payload:
        return None
    source = inspect.getsource(fn)
    return {
        "binder_kind": "runner_owned_static_plan",
        "builder": owner,
        "registered_path": registered_path,
        "canonical_reference_names": canonical_names,
        "binder_code_sha256": hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest(),
        "binder_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "captured_payload": payload,
    }


def _partial_binder_identity(
    fn: CompiledFn,
    *,
    registered_path: str,
    registered_callable: CompiledFn,
) -> dict[str, Any] | None:
    """Authenticate a partial that binds arguments to the exact registered target."""
    if not isinstance(fn, functools.partial) or fn.func is not registered_callable:
        return None
    return {
        "binder_kind": "exact_callable_partial",
        "registered_path": registered_path,
        "bound_args": [_bound_payload_component(value) for value in fn.args],
        "bound_kwargs": {
            str(key): _bound_payload_component(value)
            for key, value in sorted((fn.keywords or {}).items())
        },
    }


def _cached_compile_source_sha256(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _COMPILE_SOURCE_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = _sha256_path(resolved)
    after = resolved.stat()
    after_key = (
        str(resolved),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    )
    if after_key != key:
        raise RuntimeError(f"compile callable source changed while hashing: {resolved}")
    _COMPILE_SOURCE_HASH_CACHE[key] = digest
    return digest


def compile_callable_identity(fn_name: str, fn: CompiledFn) -> dict[str, Any]:
    """Resolve and validate the callable covered by a compile-policy name.

    Generated numeric binders must carry an explicit origin and generated-source
    digest. This prevents a closure with different equations from borrowing an
    existing registry label/evidence record.
    """
    spec = compile_kernel_spec(fn_name)
    if spec is None:
        return {
            "kernel_name": str(fn_name),
            "registered": False,
            "actual_callable": _qualified_callable_path(fn),
        }
    if spec.kind != "pure_numeric" or spec.callable is None:
        raise ValueError(
            f"torch.compile policy {fn_name!r} is {spec.kind!r}, not a pure numeric callable"
        )
    actual_path = _qualified_callable_path(fn)
    generated_kernel = getattr(fn, "__mpas_compile_kernel_name__", None)
    generated_origin = getattr(fn, "__mpas_compile_origin_callable__", None)
    generated_body = getattr(fn, "__mpas_compile_numeric_body__", None)
    generated_digest = getattr(fn, "__mpas_compile_generated_source_sha256__", None)
    static_plan_binder = None
    partial_binder = None
    generated = generated_kernel is not None or generated_origin is not None
    if (
        not generated
        and spec.name == "dynamics_tendency_dyn_tend_whole_stage"
        and actual_path
        == "mpas_pytorch.dynamics.dyn_tend._dynamics_tendency_bound_numeric_body"
    ):
        bound_body = getattr(fn, "__globals__", {}).get("_dynamics_tendency_body")
        body_path = (
            _qualified_callable_path(bound_body) if callable(bound_body) else None
        )
        if body_path != spec.numeric_callable:
            raise ValueError(
                "whole-dynamics-tendency generated binder is not bound to the canonical numeric body "
                "registered by policy"
            )
        generated = True
        generated_kernel = spec.name
        generated_origin = spec.callable
        generated_body = body_path
        generated_digest = hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest()
    if generated:
        if generated_kernel != fn_name or generated_origin != spec.callable:
            raise ValueError(
                f"generated callable identity mismatch for {fn_name!r}: "
                f"kernel={generated_kernel!r}, origin={generated_origin!r}, "
                f"registered={spec.callable!r}"
            )
        if not isinstance(generated_body, str) or not generated_body:
            raise ValueError(
                f"generated callable for {fn_name!r} lacks numeric-body identity"
            )
        if not isinstance(generated_digest, str) or len(generated_digest) != 64:
            raise ValueError(f"generated callable for {fn_name!r} lacks source SHA-256")
        try:
            int(generated_digest, 16)
        except ValueError as exc:
            raise ValueError(
                f"generated callable for {fn_name!r} has invalid source SHA-256"
            ) from exc
    elif actual_path != spec.callable:
        registered_callable = _resolve_registered_callable(spec.callable)
        partial_binder = _partial_binder_identity(
            fn,
            registered_path=spec.callable,
            registered_callable=registered_callable,
        )
        if partial_binder is None:
            static_plan_binder = _static_plan_binder_identity(
                fn,
                registered_path=spec.callable,
                registered_callable=registered_callable,
            )
        if partial_binder is None and static_plan_binder is None:
            raise ValueError(
                f"callable identity mismatch for {fn_name!r}: "
                f"actual={actual_path!r}, registered={spec.callable!r}"
            )
    source_owner = (
        _resolve_registered_callable(spec.callable)
        if partial_binder is not None
        else fn
    )
    source = inspect.getsourcefile(source_owner)
    source_sha256 = (
        _cached_compile_source_sha256(Path(source)) if source is not None else None
    )
    return {
        "kernel_name": spec.name,
        "registered": True,
        "registered_callable": spec.callable,
        "actual_callable": actual_path,
        "generated": generated,
        "generated_numeric_body": generated_body,
        "generated_source_sha256": generated_digest,
        "static_plan_binder": static_plan_binder,
        "partial_binder": partial_binder,
        "source_path": str(Path(source).resolve()) if source is not None else None,
        "source_sha256": source_sha256,
    }


def compile_kernel_registry() -> dict[str, dict[str, Any]]:
    """Return JSON-safe compile-kernel registry metadata keyed by kernel name."""
    return {spec.name: spec.as_dict() for spec in _COMPILE_KERNEL_SPECS}


def cuda_inductor_candidate_kernels() -> tuple[str, ...]:
    """Return helpers eligible for the CUDA Inductor promotion ladder."""
    return tuple(sorted(_CUDA_INDUCTOR_CANDIDATE_KERNELS))


def cuda_inductor_stage_names() -> tuple[str, ...]:
    """Return the ordered staged CUDA compile ladder names."""
    return tuple(name for name, _helpers in _CUDA_INDUCTOR_STAGE_GROUPS)


def normalize_cuda_inductor_stage(stage: object | None) -> str | None:
    """Normalize a staged CUDA compile ladder name."""
    if stage is None:
        return None
    raw = str(stage).strip().lower()
    if not raw or raw in {"none", "off", "false", "0"}:
        return None
    key = raw.replace(" ", "_").replace("-", "_")
    normalized = _CUDA_INDUCTOR_STAGE_ALIASES.get(key, key)
    if normalized not in cuda_inductor_stage_names():
        allowed = ", ".join(cuda_inductor_stage_names())
        raise ValueError(
            f"Unknown CUDA Inductor compile stage {stage!r}; allowed: {allowed}"
        )
    return normalized


def cuda_inductor_stage_groups() -> dict[str, list[str]]:
    """Return non-cumulative helper groups for each CUDA compile ladder stage."""
    return {name: list(helpers) for name, helpers in _CUDA_INDUCTOR_STAGE_GROUPS}


def cuda_inductor_stage_kernels(stage: object | None) -> tuple[str, ...]:
    """Return cumulative helpers enabled through a CUDA compile ladder stage."""
    normalized = normalize_cuda_inductor_stage(stage)
    if normalized is None:
        return ()
    helpers: list[str] = []
    for name, group in _CUDA_INDUCTOR_STAGE_GROUPS:
        helpers.extend(group)
        if name == normalized:
            break
    return tuple(sorted(dict.fromkeys(helpers)))


def _cuda_inductor_stage_policy_entry(
    *,
    stage: str,
    index: int,
    group: tuple[str, ...],
    config: dict | None,
    device: torch.device | str | None,
    prior_effective: set[str],
) -> tuple[dict[str, Any], set[str]]:
    """Return one ladder row with requested and policy-effective helper sets."""
    requested_cumulative = list(cuda_inductor_stage_kernels(stage))
    effective_cumulative = [
        helper
        for helper in requested_cumulative
        if helper in _CUDA_INDUCTOR_GPU_VALIDATED_KERNELS
    ]
    effective_set = set(effective_cumulative)
    effective_new = [helper for helper in group if helper in effective_set]
    blocked_new = [helper for helper in group if helper not in effective_set]
    blocked_no_gpu_evidence_new = [
        helper
        for helper in group
        if helper in _CUDA_INDUCTOR_CANDIDATE_KERNELS
        and helper not in _CUDA_INDUCTOR_GPU_VALIDATED_KERNELS
    ]
    effective_delta = sorted(effective_set - prior_effective)
    entry: dict[str, Any] = {
        "index": index,
        "name": stage,
        "new_helpers": list(group),
        "cumulative_helpers": requested_cumulative,
        "requested_new_helpers": list(group),
        "requested_cumulative_helpers": requested_cumulative,
        "effective_new_helpers": effective_new,
        "effective_cumulative_helpers": effective_cumulative,
        "effective_delta_helpers": effective_delta,
        "blocked_new_helpers": blocked_new,
        "blocked_no_gpu_evidence_new_helpers": blocked_no_gpu_evidence_new,
        "no_effective_delta": not effective_delta,
        "performance_claim_status": (
            "no_effective_delta" if not effective_delta else "effective_delta"
        ),
    }
    if config is None and device is None:
        return entry, effective_set

    stage_config = dict(config or {})
    stage_config["config_torch_compile_stage"] = stage
    stage_config.setdefault("config_use_torch_compile", True)
    runtime_enabled_cumulative = [
        helper
        for helper in requested_cumulative
        if kernel_compile_enabled_for_device(helper, stage_config, device=device)
    ]
    runtime_enabled_set = set(runtime_enabled_cumulative)
    entry.update(
        {
            "runtime_enabled_new_helpers": [
                helper for helper in group if helper in runtime_enabled_set
            ],
            "runtime_enabled_cumulative_helpers": runtime_enabled_cumulative,
        }
    )
    return entry, effective_set


def cuda_inductor_stage_ladder(
    config: dict | None = None,
    *,
    device: torch.device | str | None = None,
) -> list[dict[str, Any]]:
    """Return staged compile ladder metadata with requested/effective helper sets."""
    ladder = []
    prior_effective: set[str] = set()
    for index, (name, group) in enumerate(_CUDA_INDUCTOR_STAGE_GROUPS, start=1):
        entry, prior_effective = _cuda_inductor_stage_policy_entry(
            stage=name,
            index=index,
            group=group,
            config=config,
            device=device,
            prior_effective=prior_effective,
        )
        ladder.append(entry)
    return ladder


def cuda_inductor_stage_policy_report(
    stage: object | None,
    config: dict,
    *,
    device: torch.device | str | None,
) -> dict[str, Any] | None:
    """Return requested/effective/blocked helper metadata for one stage."""
    normalized = normalize_cuda_inductor_stage(stage)
    if normalized is None:
        return None
    ladder = cuda_inductor_stage_ladder(config, device=device)
    for entry in ladder:
        if entry["name"] == normalized:
            return entry
    return None


def orchestration_compile_blocked_kernels() -> tuple[str, ...]:
    """Return orchestration entry points that are never numeric compile regions."""
    return tuple(sorted(_ORCHESTRATION_COMPILE_BLOCKED_KERNELS))


def numeric_compile_policy_kernels() -> tuple[str, ...]:
    """Return compile-policy names admitted as numeric regions."""
    return tuple(sorted(_NUMERIC_COMPILE_POLICY_KERNELS))


def cuda_inductor_cpu_prequalified_kernels() -> tuple[str, ...]:
    """Return helpers with CPU eager/compile/AD/TS evidence."""
    return tuple(sorted(_CUDA_INDUCTOR_CPU_PREQUALIFIED_KERNELS))


def cuda_inductor_gpu_validated_kernel_registry() -> tuple[str, ...]:
    """Return helpers with recorded GPU validation evidence."""
    return tuple(sorted(_CUDA_INDUCTOR_GPU_VALIDATED_KERNELS))


def _cuda_inductor_runtime_environment(
    device: torch.device | str | None,
) -> tuple[str, str, str | None]:
    cuda_version = torch.version.cuda or ""
    if _device_type(device) != "cuda" or not cuda_version:
        return (torch.__version__, cuda_version, None)
    if device is None:
        return (torch.__version__, cuda_version, None)
    try:
        if not torch.cuda.is_available():
            return (torch.__version__, cuda_version, None)
    except Exception:  # noqa: BLE001
        return (torch.__version__, cuda_version, None)

    try:
        torch_device = torch.device(device)
        index = torch_device.index
        if index is None:
            index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
    except Exception:  # noqa: BLE001
        return (torch.__version__, cuda_version, None)
    return (torch.__version__, cuda_version, f"sm{major}{minor}")


def cuda_inductor_gpu_environment_validated(
    device: torch.device | str | None = None,
) -> bool:
    """Return whether the current CUDA software/device class has evidence."""
    return (
        _cuda_inductor_runtime_environment(device)
        in _CUDA_INDUCTOR_VALIDATED_ENVIRONMENTS
    )


def cuda_inductor_gpu_validation_environment(
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Return current CUDA validation-environment metadata."""
    key = _cuda_inductor_runtime_environment(device)
    return {
        "torch_version": key[0],
        "cuda_version": key[1],
        "device_capability": key[2],
        "validated": key in _CUDA_INDUCTOR_VALIDATED_ENVIRONMENTS,
        "evidence": _CUDA_INDUCTOR_GPU_VALIDATION_EVIDENCE.get(key),
    }


def cuda_inductor_gpu_validated_kernels(
    device: torch.device | str | None = None,
) -> tuple[str, ...]:
    """Return helpers admitted by default CUDA Inductor safe mode now."""
    if device is not None and not cuda_inductor_gpu_environment_validated(device):
        return ()
    return cuda_inductor_gpu_validated_kernel_registry()


def cuda_inductor_promotion_requirements() -> dict[str, list[str]]:
    """Return the evidence required before CPU and CUDA-safe promotion."""
    return {
        "cpu_prequalification": list(_CUDA_INDUCTOR_CPU_PREQUALIFICATION_REQUIREMENTS),
        "cuda_safe_promotion": list(_CUDA_INDUCTOR_GPU_PROMOTION_REQUIREMENTS),
    }


def validate_cuda_inductor_promotion_registry() -> tuple[str, ...]:
    """Check that CUDA-safe compile policy cannot outrun validation evidence."""
    problems: list[str] = []
    names = [spec.name for spec in _COMPILE_KERNEL_SPECS]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        problems.append(f"Duplicate compile-kernel registry entries: {duplicate_names}")
    for spec in _COMPILE_KERNEL_SPECS:
        if spec.kind not in _COMPILE_KERNEL_KINDS:
            problems.append(f"{spec.name}: unknown compile kernel kind {spec.kind!r}")
        if spec.compiled_reverse_ad_policy not in _COMPILED_REVERSE_AD_POLICIES:
            problems.append(
                f"{spec.name}: unknown compiled reverse-AD policy "
                f"{spec.compiled_reverse_ad_policy!r}"
            )
        if spec.compiled_reverse_ad_policy != "blocked":
            if spec.kind != "pure_numeric" or not spec.callable:
                problems.append(
                    f"{spec.name}: compiled autodiff requires a pure numeric callable"
                )
            if not spec.ad_evidence:
                problems.append(
                    f"{spec.name}: compiled autodiff requires recorded AD evidence"
                )
        if spec.compiled_reverse_ad_policy == "cpu_validated" and not (
            _has_current_authorizing_scope_coverage(
                spec,
                spec.ad_evidence,
                evidence_group="ad_evidence",
                device_type="cpu",
                required_scopes=("compiled_reverse_ad",),
            )
        ):
            problems.append(
                f"{spec.name}: validated CPU reverse autodiff requires callable-scoped "
                "authorizing reverse-AD evidence"
            )
        if spec.compiled_reverse_ad_policy == "cuda_validated" and not (
            _has_current_authorizing_scope_coverage(
                spec,
                spec.ad_evidence,
                evidence_group="ad_evidence",
                device_type="cuda",
                required_scopes=("compiled_reverse_ad",),
            )
        ):
            problems.append(
                f"{spec.name}: validated CUDA reverse autodiff requires callable-scoped "
                "authorizing CUDA reverse-AD evidence"
            )
        if spec.stage is not None and spec.stage not in _CUDA_INDUCTOR_STAGE_ORDER:
            problems.append(f"{spec.name}: unknown staged CUDA group {spec.stage!r}")
        if spec.allow_staged_cuda and spec.stage is None:
            problems.append(f"{spec.name}: staged CUDA helper requires a stage")
        if spec.kind == "orchestration_blocked" and (
            spec.allow_core
            or spec.allow_hot
            or spec.allow_full
            or spec.allow_staged_cuda
        ):
            problems.append(f"{spec.name}: blocked orchestration cannot be allowed")
        if spec.kind == "legacy_alias":
            if spec.callable is not None:
                problems.append(f"{spec.name}: legacy alias should not name a callable")
            if (
                spec.allow_core
                or spec.allow_hot
                or spec.allow_full
                or spec.allow_staged_cuda
            ):
                problems.append(f"{spec.name}: legacy alias cannot be compile-enabled")
        if spec.kind == "pure_numeric" and not spec.callable:
            problems.append(f"{spec.name}: pure numeric entry requires a callable")
        if spec.gpu_evidence and not spec.allow_staged_cuda:
            problems.append(
                f"{spec.name}: GPU evidence is only meaningful for staged CUDA helpers"
            )
        if _has_authorizing_evidence(spec.gpu_evidence, device_type="cuda"):
            if not _has_authorizing_scope_coverage(
                spec.cpu_evidence,
                device_type="cpu",
                required_scopes=_CPU_FORWARD_INDUCTOR_EVIDENCE,
            ):
                problems.append(
                    f"{spec.name}: authorizing GPU evidence requires authorizing CPU "
                    "forward evidence"
                )
            if not _has_authorizing_scope_coverage(
                spec.ad_evidence,
                device_type="cpu",
                required_scopes=_CPU_AD_EVIDENCE,
            ):
                problems.append(
                    f"{spec.name}: authorizing GPU evidence requires authorizing CPU AD evidence"
                )
        for evidence_group, refs in (
            ("cpu_evidence", spec.cpu_evidence),
            ("gpu_evidence", spec.gpu_evidence),
            ("ad_evidence", spec.ad_evidence),
        ):
            for ref in refs:
                if not ref.evidence_id:
                    problems.append(
                        f"{spec.name}: {evidence_group} has an empty evidence ID"
                    )
                if not ref.authorizing:
                    continue
                problems.extend(
                    _authorizing_evidence_problems(
                        spec,
                        ref,
                        evidence_group=evidence_group,
                    )
                )
    if not _CUDA_INDUCTOR_CANDIDATE_KERNELS <= _COMPILE_POLICY_KERNELS:
        extra = sorted(_CUDA_INDUCTOR_CANDIDATE_KERNELS - _COMPILE_POLICY_KERNELS)
        problems.append(f"CUDA Inductor candidates outside compile policy: {extra}")
    if not _CUDA_INDUCTOR_CPU_PREQUALIFIED_KERNELS <= _CUDA_INDUCTOR_CANDIDATE_KERNELS:
        extra = sorted(
            _CUDA_INDUCTOR_CPU_PREQUALIFIED_KERNELS - _CUDA_INDUCTOR_CANDIDATE_KERNELS
        )
        problems.append(f"CPU-prequalified kernels outside CUDA candidate set: {extra}")
    if _CUDA_INDUCTOR_CANDIDATE_KERNELS & _ORCHESTRATION_COMPILE_BLOCKED_KERNELS:
        bad = sorted(
            _CUDA_INDUCTOR_CANDIDATE_KERNELS & _ORCHESTRATION_COMPILE_BLOCKED_KERNELS
        )
        problems.append(
            f"CUDA Inductor candidates include orchestration kernels: {bad}"
        )
    if (
        not _CUDA_INDUCTOR_GPU_VALIDATED_KERNELS
        <= _CUDA_INDUCTOR_CPU_PREQUALIFIED_KERNELS
    ):
        extra = sorted(
            _CUDA_INDUCTOR_GPU_VALIDATED_KERNELS
            - _CUDA_INDUCTOR_CPU_PREQUALIFIED_KERNELS
        )
        problems.append(f"GPU-validated kernels without CPU prequalification: {extra}")
    if _CUDA_INDUCTOR_SIMPLE_KERNELS != _CUDA_INDUCTOR_GPU_VALIDATED_KERNELS:
        problems.append(
            "CUDA Inductor safe list must be exactly the GPU-validated kernel set"
        )
    expected_environments = {
        (
            str(ref.torch_version),
            str(ref.cuda_version),
            str(ref.compute_capability),
        )
        for spec in _COMPILE_KERNEL_SPECS
        if spec.name in _CUDA_INDUCTOR_GPU_VALIDATED_KERNELS
        for ref in spec.gpu_evidence
        if ref.authorizing and ref.device_type == "cuda"
    }
    if set(_CUDA_INDUCTOR_VALIDATED_ENVIRONMENTS) != expected_environments:
        problems.append(
            "CUDA validated environments must derive from authorizing GPU evidence"
        )
    return tuple(problems)


def _resolve_config_scalar(
    config: Mapping[str, Any], key: str, default: object
) -> object:
    """Resolve scalar config values and cache tensor conversions."""
    if key in config:
        value = config[key]
        in_config = True
    else:
        value = default
        in_config = False

    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(
                f"{key} must be a Python scalar or CPU scalar tensor; "
                f"got {value.device.type} tensor"
            )
        scalar = value.item()
        if in_config and isinstance(config, dict):
            config[key] = scalar
        return scalar

    return value


def compile_enabled(config: dict) -> bool:
    """Return whether compile wrapping is enabled for this run."""
    if os.environ.get("MPAS_DISABLE_TORCH_COMPILE", "0") == "1":
        return False
    value = _resolve_config_scalar(config, "config_use_torch_compile", False)
    return bool(value)


def compile_mode(config: dict) -> str:
    """Resolve torch.compile mode from config."""
    mode = _resolve_config_scalar(
        config, "config_torch_compile_mode", "reduce-overhead"
    )
    mode = str(mode)
    if mode not in {"default", "reduce-overhead", "max-autotune"}:
        return "reduce-overhead"
    return mode


def compile_target_kernels(config: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Return an explicit compile target allowlist, when one is configured.

    The runner uses this only for preflight warmup: one helper is enabled at a
    time on the target case while every other numeric region stays eager.
    """
    if "config_torch_compile_target_kernels" in config:
        raw = config["config_torch_compile_target_kernels"]
    elif "config_torch_compile_target_kernel" in config:
        raw = config["config_torch_compile_target_kernel"]
    else:
        return None

    if raw is None:
        return ()
    if isinstance(raw, torch.Tensor):
        if raw.device.type != "cpu":
            raise ValueError(
                "config_torch_compile_target_kernels must be Python values or "
                f"CPU tensors; got {raw.device.type} tensor"
            )
        raw = raw.detach().cpu().reshape(-1).tolist()
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (tuple, list, frozenset, set)):
        values = [str(item).strip() for item in raw if str(item).strip()]
    else:
        values = [str(raw).strip()] if str(raw).strip() else []
    targets = tuple(sorted(set(values)))
    unknown = tuple(name for name in targets if name not in _COMPILE_KERNEL_REGISTRY)
    if unknown:
        raise ValueError("unknown torch.compile target kernels: " + ", ".join(unknown))
    return targets


def cuda_inductor_stage(config: Mapping[str, Any]) -> str | None:
    """Return the configured CUDA Inductor compile ladder stage, if any."""
    if "config_torch_compile_stage" in config:
        raw = _resolve_config_scalar(config, "config_torch_compile_stage", None)
    elif "config_torch_compile_cuda_inductor_stage" in config:
        raw = _resolve_config_scalar(
            config,
            "config_torch_compile_cuda_inductor_stage",
            None,
        )
    else:
        raw = None
    return normalize_cuda_inductor_stage(raw)


def kernel_compile_enabled(fn_name: str, config: Mapping[str, Any]) -> bool:
    """Resolve per-kernel compile policy from config scope."""
    scope = _resolve_config_scalar(config, "config_torch_compile_scope", "hot")
    scope = str(scope).strip().lower()
    if fn_name in _ORCHESTRATION_COMPILE_BLOCKED_KERNELS:
        return False
    target_kernels = compile_target_kernels(config)
    if target_kernels is not None:
        return (
            fn_name in set(target_kernels)
            and fn_name in _NUMERIC_COMPILE_POLICY_KERNELS
        )
    stage = cuda_inductor_stage(config)
    if stage is not None:
        return fn_name in set(cuda_inductor_stage_kernels(stage))
    if scope in {"all", "full"}:
        return fn_name in _NUMERIC_COMPILE_POLICY_KERNELS
    if scope in {"none", "off", "false", "0"}:
        return False
    if scope in {"core", "minimal"}:
        return fn_name in _CORE_HOT_KERNELS
    # default "hot": the CPU-validated dycore compile set.
    return fn_name in _HOT_KERNELS


def cuda_inductor_policy(config: Mapping[str, Any]) -> str:
    """Resolve CUDA Inductor policy for dycore kernel wrapping."""
    policy = _resolve_config_scalar(
        config,
        "config_torch_compile_cuda_inductor_policy",
        "safe",
    )
    normalized = str(policy).strip().lower().replace("_", "-")
    normalized = _CUDA_INDUCTOR_POLICY_ALIASES.get(normalized, normalized)
    if normalized not in {"safe", "force", "off"}:
        return "safe"
    return normalized


def _device_type(device: torch.device | str | None) -> str:
    if device is None:
        return "cpu"
    if isinstance(device, torch.device):
        return device.type
    try:
        return torch.device(device).type
    except (TypeError, RuntimeError):
        return str(device).split(":", 1)[0].strip().lower()


def kernel_compile_enabled_for_device(
    fn_name: str,
    config: Mapping[str, Any],
    *,
    device: torch.device | str | None,
) -> bool:
    """Resolve kernel compile policy including CUDA Inductor safety guards."""
    if not kernel_compile_enabled(fn_name, config):
        return False
    if _device_type(device) != "cuda":
        return True
    if compile_backend(config) != "inductor":
        return True
    if (
        fn_name in _CUDA_INDUCTOR_CANDIDATE_KERNELS
        and compile_target_kernels(config) is None
        and cuda_inductor_stage(config) is None
    ):
        return False

    policy = cuda_inductor_policy(config)
    if policy == "off":
        return False
    if policy == "force":
        return True
    return fn_name in cuda_inductor_gpu_validated_kernels(device)


def kernel_compiled_reverse_ad_enabled_for_device(
    fn_name: str,
    config: dict,
    *,
    device: torch.device | str | None,
) -> bool:
    """Admit explicitly targeted compiled AD without widening normal policy.

    CPU experiments are opt-in until a measured production artifact promotes the
    registry entry to ``cpu_validated``. CUDA remains fail-closed unless a future
    entry is explicitly promoted to ``cuda_validated`` after CUDA AD evidence.
    """
    if not kernel_compile_enabled_for_device(fn_name, config, device=device):
        return False
    targets = compile_target_kernels(config)
    if targets is None or fn_name not in targets:
        return False
    spec = _COMPILE_KERNEL_REGISTRY.get(fn_name)
    if spec is None or spec.kind != "pure_numeric" or not spec.ad_evidence:
        return False
    policy = spec.compiled_reverse_ad_policy
    device_type = _device_type(device)
    if device_type == "cuda":
        return policy == "cuda_validated" and _has_current_authorizing_scope_coverage(
            spec,
            spec.ad_evidence,
            evidence_group="ad_evidence",
            device_type="cuda",
            required_scopes=("compiled_reverse_ad",),
        )
    requested = (
        str(
            _resolve_config_scalar(
                config,
                "config_torch_compile_differentiable_policy",
                "blocked",
            )
        )
        .strip()
        .lower()
    )
    evidence_generation = bool(
        _resolve_config_scalar(
            config,
            "config_torch_compile_differentiable_evidence_generation",
            False,
        )
    )
    if requested == "experiment" and evidence_generation:
        # A stale authorizing artifact must block production admission, but it
        # cannot also block the explicit CPU experiment that replaces it.  This
        # capability is deliberately unavailable on CUDA and is never inferred
        # from the registry policy.
        return True
    if policy == "cpu_validated":
        return _has_current_authorizing_scope_coverage(
            spec,
            spec.ad_evidence,
            evidence_group="ad_evidence",
            device_type="cpu",
            required_scopes=("compiled_reverse_ad",),
        )
    if policy != "cpu_experiment":
        return False
    return requested == "experiment"


def validate_differentiable_compile_request(
    config: dict,
    *,
    device: torch.device | str | None,
    values: object | None = None,
) -> tuple[str, ...]:
    """Validate exact reverse-AD compile targets; forward-mode stays blocked."""
    if not compile_enabled(config):
        return ()
    if values is not None and _value_carries_forward_ad(values):
        raise ValueError(
            "differentiable torch.compile forward AD/JVP is not qualified; "
            "use the eager differentiable path"
        )
    targets = compile_target_kernels(config)
    if targets is None:
        raise ValueError(
            "differentiable torch.compile requires an explicit "
            "config_torch_compile_target_kernels allowlist"
        )
    if not targets:
        raise ValueError(
            "differentiable torch.compile requires at least one explicit target kernel"
        )
    blocked = [
        name
        for name in targets
        if not kernel_compiled_reverse_ad_enabled_for_device(
            name,
            config,
            device=device,
        )
    ]
    if blocked:
        raise ValueError(
            "differentiable torch.compile targets are not AD-qualified for "
            f"device {_device_type(device)!r}: {blocked}"
        )
    return targets


def compile_backend(config: Mapping[str, Any]) -> str:
    """Resolve torch.compile backend from config."""
    backend = _resolve_config_scalar(config, "config_torch_compile_backend", "inductor")
    backend = str(backend).strip()
    if not backend:
        return "inductor"
    return backend


def compile_policy_snapshot(
    config: dict,
    *,
    device: torch.device | str | None,
    kernel_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return the effective compile policy for known dycore kernels."""
    cfg = dict(config)
    compile_on = compile_enabled(cfg)
    backend = compile_backend(cfg)
    mode = compile_mode(cfg)
    scope = (
        str(_resolve_config_scalar(cfg, "config_torch_compile_scope", "hot"))
        .strip()
        .lower()
    )
    target_kernels = compile_target_kernels(cfg)
    stage = cuda_inductor_stage(cfg)
    device_type = _device_type(device)
    policy = cuda_inductor_policy(cfg)
    candidates = tuple(sorted(kernel_names or _COMPILE_POLICY_KERNELS))
    enabled = (
        [
            name
            for name in candidates
            if kernel_compile_enabled_for_device(name, cfg, device=device)
        ]
        if compile_on
        else []
    )
    return {
        "compile_enabled": bool(compile_on),
        "device_type": device_type,
        "backend": backend,
        "mode": mode,
        "scope": scope,
        "target_kernels": list(target_kernels) if target_kernels is not None else None,
        "compile_kernel_registry": compile_kernel_registry(),
        "cuda_inductor_stage": stage,
        "cuda_inductor_stage_ladder": cuda_inductor_stage_ladder(
            cfg,
            device=device,
        ),
        "cuda_inductor_stage_report": cuda_inductor_stage_policy_report(
            stage,
            cfg,
            device=device,
        ),
        "cuda_inductor_policy": policy,
        "cuda_inductor_candidate_kernels": list(cuda_inductor_candidate_kernels()),
        "cuda_inductor_cpu_prequalified_kernels": list(
            cuda_inductor_cpu_prequalified_kernels()
        ),
        "cuda_inductor_gpu_validated_kernel_registry": list(
            cuda_inductor_gpu_validated_kernel_registry()
        ),
        "cuda_inductor_gpu_validated_kernels": list(
            cuda_inductor_gpu_validated_kernels(device)
        ),
        "cuda_inductor_gpu_validation_environment": cuda_inductor_gpu_validation_environment(
            device
        ),
        "cuda_inductor_promotion_requirements": cuda_inductor_promotion_requirements(),
        "orchestration_compile_blocked_kernels": list(
            orchestration_compile_blocked_kernels()
        ),
        "numeric_compile_policy_kernels": list(numeric_compile_policy_kernels()),
        "candidate_kernels": list(candidates),
        "enabled_kernels": enabled,
        "disabled_kernels": [name for name in candidates if name not in enabled],
    }


def validate_effective_compile_request(
    config: dict,
    *,
    device: torch.device | str | None,
) -> None:
    """Fail closed when an explicit compile request enables no numeric helper.

    Target-kernel and staged compile runs are used as evidence for numerical and
    performance claims.  If the requested names are unknown, legacy aliases,
    orchestration-blocked wrappers, or evidence-blocked helpers, the run must not
    proceed as a misleading "compiled" gate with an empty effective kernel set.
    """
    if not compile_enabled(config):
        return

    target_kernels = compile_target_kernels(config)
    stage = cuda_inductor_stage(config)
    if target_kernels is None and stage is None:
        return

    snapshot = compile_policy_snapshot(config, device=device)
    enabled = tuple(str(name) for name in snapshot.get("enabled_kernels", ()))

    if target_kernels is not None:
        requested = tuple(str(name) for name in target_kernels)
        unknown = tuple(
            name for name in requested if name not in _COMPILE_POLICY_KERNELS
        )
        not_numeric = tuple(
            name
            for name in requested
            if name in _COMPILE_POLICY_KERNELS
            and name not in _NUMERIC_COMPILE_POLICY_KERNELS
        )
        disabled = tuple(
            name
            for name in requested
            if name in _NUMERIC_COMPILE_POLICY_KERNELS
            and not kernel_compile_enabled_for_device(name, config, device=device)
        )
        if unknown or not_numeric or disabled or not enabled:
            details: list[str] = []
            if unknown:
                details.append(f"unknown={list(unknown)}")
            if not_numeric:
                details.append(f"not_numeric_or_not_promoted={list(not_numeric)}")
            if disabled:
                details.append(f"disabled_by_device_policy={list(disabled)}")
            if not enabled:
                details.append("effective_enabled_kernels=[]")
            raise ValueError(
                "torch.compile target request does not resolve to an effective "
                "numeric kernel set; " + ", ".join(details)
            )

    if stage is not None and not enabled:
        raise ValueError(
            "torch.compile staged request does not resolve to any effective "
            f"numeric kernels for stage {stage!r} on device {_device_type(device)!r}"
        )


def _freeze_specialization_item(value: object) -> Hashable:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            if value.device.type != "cpu":
                raise ValueError(
                    "torch.compile specialization keys must use Python scalars "
                    f"or CPU scalar tensors; got {value.device.type} tensor"
                )
            return _freeze_specialization_item(value.detach().reshape(()).tolist())
        return ("tensor", tuple(value.shape), str(value.dtype), str(value.device))
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value  # type: ignore[return-value]


def _normalize_specialization_key(value: object | None) -> tuple[Hashable, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return tuple(_freeze_specialization_item(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_specialization_item(item) for item in value)
    return (_freeze_specialization_item(value),)


def _primitive_default_key(value: object) -> Hashable | None:
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    return None


def _compile_function_cache_key(fn: CompiledFn) -> FunctionCacheKey:
    """
    Return a stable cache key for safe function objects.

    Top-level functions and non-closure local functions can be recreated across
    calls without changing semantics. Closure functions may capture changing
    tensors or scalars, so those retain identity-based keys.
    """
    code = getattr(fn, "__code__", None)
    closure = getattr(fn, "__closure__", None)
    defaults = getattr(fn, "__defaults__", None) or ()
    kwdefaults = getattr(fn, "__kwdefaults__", None) or {}
    if code is None or closure:
        return ("callable_id", id(fn))

    default_keys = []
    for value in defaults:
        default_key = _primitive_default_key(value)
        if default_key is None:
            return ("callable_id", id(fn))
        default_keys.append(default_key)
    kwdefault_keys = []
    for key, value in sorted(kwdefaults.items()):
        default_key = _primitive_default_key(value)
        if default_key is None:
            return ("callable_id", id(fn))
        kwdefault_keys.append((key, default_key))

    return (
        "function",
        getattr(fn, "__module__", type(fn).__module__),
        getattr(fn, "__qualname__", getattr(fn, "__name__", type(fn).__qualname__)),
        code.co_filename,
        code.co_firstlineno,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        tuple(default_keys),
        tuple(kwdefault_keys),
    )


def _compile_backend_environment_key(backend: str) -> tuple[Hashable, ...]:
    """Return backend settings that materially affect generated wrapper code."""
    if str(backend) != "inductor":
        return ()
    try:
        import torch._inductor.config as inductor_config

        triton_config = getattr(inductor_config, "triton", None)
        cudagraphs = getattr(triton_config, "cudagraphs", None)
        cudagraph_trees = getattr(triton_config, "cudagraph_trees", None)
    except Exception:  # noqa: BLE001
        return ()
    return (
        ("triton.cudagraphs", bool(cudagraphs) if cudagraphs is not None else None),
        (
            "triton.cudagraph_trees",
            bool(cudagraph_trees) if cudagraph_trees is not None else None,
        ),
    )


def _tensor_carries_autodiff(tensor: torch.Tensor) -> bool:
    """Return whether a tensor participates in reverse- or forward-mode AD."""

    if getattr(tensor, "requires_grad", False):
        return True
    try:
        from torch.autograd import forward_ad as fwAD

        _primal, tangent = fwAD.unpack_dual(tensor)
    except Exception:  # noqa: BLE001
        return False
    return tangent is not None


def _tensor_carries_forward_ad(tensor: torch.Tensor) -> bool:
    try:
        from torch.autograd import forward_ad as fwAD

        _primal, tangent = fwAD.unpack_dual(tensor)
    except Exception:  # noqa: BLE001
        return False
    return tangent is not None


def _value_carries_autodiff(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return _tensor_carries_autodiff(value)
    if isinstance(value, Mapping):
        try:
            values: Iterable[Any] = value.values()
        except TypeError:
            values = (value[key] for key in value)
        return any(_value_carries_autodiff(item) for item in values)
    if isinstance(value, (tuple, list)):
        return any(_value_carries_autodiff(item) for item in value)
    return False


def _value_carries_forward_ad(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return _tensor_carries_forward_ad(value)
    if isinstance(value, Mapping):
        try:
            values: Iterable[Any] = value.values()
        except TypeError:
            values = (value[key] for key in value)
        return any(_value_carries_forward_ad(item) for item in values)
    if isinstance(value, (tuple, list)):
        return any(_value_carries_forward_ad(item) for item in value)
    return False


def _call_carries_autodiff(args: tuple[object, ...], kwargs: Mapping[str, Any]) -> bool:
    return any(_value_carries_autodiff(arg) for arg in args) or any(
        _value_carries_autodiff(value) for value in kwargs.values()
    )


def maybe_compile_kernel(
    fn_name: str,
    fn: CompiledFn,
    *,
    enabled: bool,
    mode: str,
    backend: str,
    specialization_key: object | None = None,
    allow_reverse_autodiff: bool = False,
) -> CompiledFn:
    """
    Optionally wrap a kernel with torch.compile using a memoized cache.

    Falls back to eager function when compile is unavailable or raises.
    """
    if not enabled:
        return fn

    if not hasattr(torch, "compile"):
        _COMPILE_RUNTIME_COUNTERS["eager_fallback"] += 1
        return fn

    callable_identity = compile_callable_identity(fn_name, fn)
    callable_identity_digest = hashlib.sha256(
        json.dumps(callable_identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    normalized_specialization_key = _normalize_specialization_key(specialization_key)
    key = (
        fn_name,
        callable_identity_digest,
        _compile_function_cache_key(fn),
        mode,
        backend,
        _compile_backend_environment_key(backend),
        normalized_specialization_key,
        bool(allow_reverse_autodiff),
    )
    key_digest: str | None = None

    def _telemetry_key_digest() -> str:
        nonlocal key_digest
        if key_digest is None:
            key_digest = _cache_key_digest(key)
        return key_digest

    cached = _COMPILED_CACHE.get(key)
    if cached is not None:
        if compile_event_sink_enabled():
            _emit_compile_event(
                fn_name,
                "wrapper_cache_hit",
                cache_hit=True,
                cache_key=_telemetry_key_digest(),
                mode=mode,
                backend=backend,
                specialization_key=normalized_specialization_key,
                callable_identity=callable_identity,
                compile_backend_environment_key=_compile_backend_environment_key(
                    backend
                ),
                compile_logging=_compile_logging_snapshot(),
            )
        return cached

    _COMPILE_RUNTIME_COUNTERS["wrapper_create"] += 1
    _COMPILE_RUNTIME_COUNTERS["wrapper_cache_miss"] += 1
    _COMPILE_RUNTIME_COUNTERS["specialization_revision"] += 1
    create_start = time.perf_counter()
    if compile_event_sink_enabled():
        _emit_compile_event(
            fn_name,
            "wrapper_create_start",
            cache_hit=False,
            cache_key=_telemetry_key_digest(),
            mode=mode,
            backend=backend,
            specialization_key=normalized_specialization_key,
            callable_identity=callable_identity,
            compile_backend_environment_key=_compile_backend_environment_key(backend),
            compile_logging=_compile_logging_snapshot(),
        )
    try:
        try:
            compiled = torch.compile(fn, mode=mode, backend=backend)
        except TypeError as exc:
            if "mode" not in str(exc):
                raise
            compiled = torch.compile(fn, backend=backend)
    except Exception as exc:  # pragma: no cover - fallback path  # noqa: BLE001
        _COMPILE_RUNTIME_COUNTERS["eager_fallback"] += 1
        if compile_event_sink_enabled():
            _emit_compile_event(
                fn_name,
                "wrapper_create_error",
                cache_hit=False,
                cache_key=_telemetry_key_digest(),
                mode=mode,
                backend=backend,
                specialization_key=normalized_specialization_key,
                compile_backend_environment_key=_compile_backend_environment_key(
                    backend
                ),
                compile_logging=_compile_logging_snapshot(),
                wrapper_create_seconds=time.perf_counter() - create_start,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        warnings.warn(
            f"torch.compile failed for {fn_name} ({exc}); falling back to eager path.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fn

    @functools.wraps(fn)
    def _compiled_with_telemetry(*args: object, **kwargs: object) -> object:
        carries_forward_ad = _value_carries_forward_ad(
            args
        ) or _value_carries_forward_ad(kwargs)
        if carries_forward_ad or (
            _call_carries_autodiff(args, kwargs) and not allow_reverse_autodiff
        ):
            _COMPILE_RUNTIME_COUNTERS["autodiff_bypass"] += 1
            if compile_event_sink_enabled():
                _emit_compile_event(
                    fn_name,
                    "compiled_call_autodiff_bypass",
                    cache_hit=True,
                    cache_key=_telemetry_key_digest(),
                    mode=mode,
                    backend=backend,
                    specialization_key=normalized_specialization_key,
                    compile_backend_environment_key=_compile_backend_environment_key(
                        backend
                    ),
                    compile_logging=_compile_logging_snapshot(),
                    argument_signature=_argument_signature(args, kwargs),
                    autodiff_bypass_reason=(
                        "forward_ad_unqualified"
                        if carries_forward_ad
                        else "reverse_ad_not_enabled"
                    ),
                )
            return fn(*args, **kwargs)

        if not compile_event_sink_enabled():
            return compiled(*args, **kwargs)

        call_id = next(_COMPILE_EVENT_COUNTER)
        before = _dynamo_counter_snapshot()
        call_start = time.perf_counter()
        argument_signature = _argument_signature(args, kwargs)
        _emit_compile_event(
            fn_name,
            "compiled_call_start",
            call_id=call_id,
            cache_key=_telemetry_key_digest(),
            mode=mode,
            backend=backend,
            specialization_key=normalized_specialization_key,
            compile_backend_environment_key=_compile_backend_environment_key(backend),
            compile_logging=_compile_logging_snapshot(),
            argument_signature=argument_signature,
            dynamo_counters_before=before,
        )
        try:
            result = compiled(*args, **kwargs)
        except Exception as exc:
            _COMPILE_RUNTIME_COUNTERS["compiled_call_error"] += 1
            after = _dynamo_counter_snapshot()
            delta = _counter_delta(after, before)
            _emit_compile_event(
                fn_name,
                "compiled_call_error",
                call_id=call_id,
                cache_key=_telemetry_key_digest(),
                mode=mode,
                backend=backend,
                specialization_key=normalized_specialization_key,
                compile_backend_environment_key=_compile_backend_environment_key(
                    backend
                ),
                compile_logging=_compile_logging_snapshot(),
                compile_or_call_seconds=time.perf_counter() - call_start,
                dynamo_counters_after=after,
                dynamo_counters_delta=delta,
                recompile_indicator_count=_recompile_indicator_count(delta),
                explain=_compile_call_explain_payload(fn, args, kwargs),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise

        after = _dynamo_counter_snapshot()
        delta = _counter_delta(after, before)
        _emit_compile_event(
            fn_name,
            "compiled_call_success",
            call_id=call_id,
            cache_key=_telemetry_key_digest(),
            mode=mode,
            backend=backend,
            specialization_key=normalized_specialization_key,
            compile_backend_environment_key=_compile_backend_environment_key(backend),
            compile_logging=_compile_logging_snapshot(),
            compile_or_call_seconds=time.perf_counter() - call_start,
            dynamo_counters_after=after,
            dynamo_counters_delta=delta,
            recompile_indicator_count=_recompile_indicator_count(delta),
            explain=_compile_call_explain_payload(fn, args, kwargs),
            output_signature=_value_signature(result),
        )
        return result

    _COMPILED_CACHE[key] = _compiled_with_telemetry
    _compiled_with_telemetry.__mpas_compile_callable_identity__ = (  # type: ignore[attr-defined]
        callable_identity
    )
    if compile_event_sink_enabled():
        _emit_compile_event(
            fn_name,
            "wrapper_created",
            cache_hit=False,
            cache_key=_telemetry_key_digest(),
            mode=mode,
            backend=backend,
            specialization_key=normalized_specialization_key,
            compile_backend_environment_key=_compile_backend_environment_key(backend),
            compile_logging=_compile_logging_snapshot(),
            wrapper_create_seconds=time.perf_counter() - create_start,
        )
    return _compiled_with_telemetry
