"""
Dycore state dataclass for MPAS-Atmosphere PyTorch port.

Encapsulates all prognostic and diagnostic variables needed for
the RK3 time integration loop.

Entity arrays use zero-based active rows followed by one padding row:
- Cell arrays: shape (nCells+1, nVertLevels)
- Edge arrays: shape (nEdges+1, nVertLevels)
- W-level arrays: shape (nCells+1, nVertLevels+1)

The loader translates upstream connectivity into this convention.
"""

import hashlib
import json
import warnings
from dataclasses import dataclass, fields
from typing import Any, cast

import numpy as np
import torch

from mpas_pytorch.constants import DTYPE
from mpas_pytorch.exceptions import MpasConfigError
from mpas_pytorch.io_schema import continuation_configuration_digest, mesh_digest


def _check(condition: bool, message: str) -> None:
    """
    Validate a condition and raise a typed config error if false.

    Unlike assert, this check is NOT disabled with python -O.
    """
    if not condition:
        raise MpasConfigError(message)


def dict_to_device(data: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """
    Return a new dict with tensor values moved to ``device``.

    Non-tensor values are copied through unchanged.
    """
    target_device = torch.device(device)
    return {
        key: value.to(target_device) if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }


def clone_tree(value: Any) -> Any:
    """Recursively clone tensors while preserving container structure."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    return value


_BOUNDARY_PROGNOSTICS = ("u", "w", "rho_zz", "theta_m", "scalars")
_BOUNDARY_PRIMARY_ALIASES = {"t": "theta_m"}
_BOUNDARY_MARKER = "_restart_skip_rederivation"
_BOUNDARY_BINDING = "mpas-boundary-bytes.v1"
_NumericalSnapshot = dict[str, tuple[str, tuple[int, ...], bytes]]


def _boundary_prognostic_digest(state: dict[str, Any]) -> str:
    """Bind a complete step snapshot to the exact prognostic tensor values.

    A value digest survives a deep clone and catches both tensor replacement and
    in-place edits. A pointer/version receipt would incorrectly reject a clone.
    """
    digest = hashlib.sha256()
    for name in _BOUNDARY_PROGNOSTICS:
        value = state.get(name)
        if not isinstance(value, torch.Tensor):
            raise MpasConfigError(f"complete step boundary lacks tensor {name!r}")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.data.cast("B"))
    return digest.hexdigest()


def _boundary_state_snapshot(
    state: dict[str, Any], *, support_only: bool = False
) -> _NumericalSnapshot:
    """Copy immutable bytes; comparisons bind values without version-counter gaps.

    A call-local cache shares bytes for identical tensor views. The cache never
    survives a call, so numpy/.data edits and cloned tensors are checked afresh.
    """
    snapshot: _NumericalSnapshot = {}
    copies: dict[tuple[Any, ...], bytes] = {}
    identity: tuple[Any, ...]
    for key, value in state.items():
        if key.startswith("_") and key not in {
            "_edge_ordering_mismatch",
            "_mesh_connectivity_preferred",
        }:
            continue
        if support_only and (
            key in _BOUNDARY_PROGNOSTICS or key in _BOUNDARY_PRIMARY_ALIASES
        ):
            continue
        if isinstance(value, torch.Tensor):
            identity = (
                "tensor",
                str(value.device),
                value.data_ptr(),
                str(value.dtype),
                tuple(value.shape),
                tuple(value.stride()),
            )
            array = value.detach().cpu().contiguous().numpy()
        elif isinstance(value, np.ndarray):
            identity = (
                "array",
                value.__array_interface__["data"][0],
                str(value.dtype),
                value.shape,
                value.strides,
            )
            array = np.ascontiguousarray(value)
        elif isinstance(value, (bool, int, float, str, np.generic)):
            snapshot[key] = (
                "scalar",
                (),
                json.dumps(value, sort_keys=True, default=str).encode(),
            )
            continue
        else:
            continue
        if identity not in copies:
            copies[identity] = array.tobytes()
        snapshot[key] = (str(array.dtype), tuple(array.shape), copies[identity])
    return snapshot


def _boundary_support_snapshot(snapshot: _NumericalSnapshot) -> _NumericalSnapshot:
    return {
        key: value
        for key, value in snapshot.items()
        if key not in _BOUNDARY_PROGNOSTICS and key not in _BOUNDARY_PRIMARY_ALIASES
    }


def require_unedited_boundary_support(state: dict[str, Any]) -> None:
    """Primary edits may be rederived; diagnostic/support edits are invalid."""
    marker = state.get(_BOUNDARY_MARKER)
    if isinstance(marker, dict) and marker.get("binding") != _BOUNDARY_BINDING:
        raise MpasConfigError(
            "complete boundary binding is unsupported; reload a current restart"
        )
    if isinstance(marker, dict) and marker.get("support") != _boundary_state_snapshot(
        state, support_only=True
    ):
        raise MpasConfigError(
            "complete boundary diagnostics or support inputs were edited"
        )
    if isinstance(marker, dict):
        saved_aliases = marker.get("primary_aliases", {})
        for alias, primary in _BOUNDARY_PRIMARY_ALIASES.items():
            if (alias in saved_aliases) != (alias in state):
                raise MpasConfigError(
                    "complete boundary diagnostic alias inventory changed"
                )
            if alias in saved_aliases:
                value = state[alias]
                # RK's t alias can share theta_m storage or be cloned separately.
                # Retain unchanged aliases or aliases following a primary edit.
                if not isinstance(value, torch.Tensor) or (
                    _boundary_state_snapshot({alias: value}).get(alias)
                    != saved_aliases[alias]
                    and not (
                        isinstance(state.get(primary), torch.Tensor)
                        and value.dtype == state[primary].dtype
                        and value.device == state[primary].device
                        and value.shape == state[primary].shape
                        and torch.equal(value, state[primary])
                    )
                ):
                    raise MpasConfigError(
                        "complete boundary diagnostic alias was edited"
                    )


def mark_complete_step_boundary(
    state: dict[str, Any],
    *,
    mesh: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    dt: float | None = None,
) -> None:
    """Bind a reusable boundary to its values and continuation identity."""
    for name in _BOUNDARY_PROGNOSTICS:
        if not isinstance(state.get(name), torch.Tensor):
            raise MpasConfigError(f"complete step boundary lacks tensor {name!r}")
    snapshot = _boundary_state_snapshot(state)
    state[_BOUNDARY_MARKER] = {
        "binding": _BOUNDARY_BINDING,
        "state": snapshot,
        "support": _boundary_support_snapshot(snapshot),
        "primary_aliases": {
            alias: snapshot[alias]
            for alias in _BOUNDARY_PRIMARY_ALIASES
            if alias in state
        },
        "mesh": mesh_digest(mesh) if mesh is not None else None,
        "config": continuation_configuration_digest(config)
        if config is not None
        else None,
        "dt": float(dt) if dt is not None else None,
    }


def require_boundary_continuation_identity(
    state: dict[str, Any], *, mesh: dict[str, Any], config: dict[str, Any], dt: float
) -> None:
    """Primary edits require rederivation, never a different continuation identity."""
    marker = state.get(_BOUNDARY_MARKER)
    if not isinstance(marker, dict):
        return
    if (
        marker.get("binding") != _BOUNDARY_BINDING
        or marker.get("mesh") != mesh_digest(mesh)
        or marker.get("config") != continuation_configuration_digest(config)
        or marker.get("dt") != float(dt)
    ):
        raise MpasConfigError("complete boundary continuation identity mismatch")


def has_complete_step_boundary(
    state: dict[str, Any],
    *,
    mesh: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    dt: float | None = None,
) -> bool:
    """Reject stale or legacy unbound continuation markers."""
    marker = state.get(_BOUNDARY_MARKER)
    if not isinstance(marker, dict) or marker.get("binding") != _BOUNDARY_BINDING:
        return False
    try:
        return (
            marker.get("state") == _boundary_state_snapshot(state)
            and (mesh is None or marker.get("mesh") == mesh_digest(mesh))
            and (
                config is None
                or marker.get("config") == continuation_configuration_digest(config)
            )
            and (dt is None or marker.get("dt") == float(dt))
        )
    except (MpasConfigError, ValueError):
        return False


def _to_python_scalar_if_0d(value: Any) -> Any:
    """Convert 0-d tensors to Python scalars; leave everything else unchanged."""
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        return value
    if value.dtype == torch.bool:
        return bool(value.item())
    if value.dtype.is_floating_point:
        return float(value.item())
    if value.dtype.is_complex:
        return complex(value.item())
    return int(value.item())


# Required prognostic variables that must be present when strict=True.
# A model with zeros for these fields will run but produce wrong physics.
REQUIRED_PROGNOSTIC = ["u", "w", "rho_zz", "theta_m", "scalars"]

# Required for ANY dycore run (no zero-fill allowed)
CORE_REQUIRED_FIELDS = [
    "u",
    "w",
    "theta_m",
    "rho_zz",
    "pressure_p",
]

# Required for regional runs only
REGIONAL_REQUIRED_FIELDS = [
    "bdyMaskCell",
    "bdyMaskEdge",
    "scalars_driving",
    "ru_driving",
    "rtheta_m_driving",
    "rho_zz_driving",
    "rw_driving",
    "meshScalingRegionalCell",
    "meshScalingRegionalEdge",
]

# Truly optional (can be computed or are legitimately zero)
OPTIONAL_FIELDS = [
    "divergence",
    "vorticity",
    "ke",
    "pv_edge",  # Can be computed
    "rt_diabatic_tend",  # Zero for adiabatic runs
]


@dataclass
class DycoreState:
    """
    State container for MPAS dynamics core.

    Attributes:
        Dimensions:
            nCells: Number of cells
            nEdges: Number of edges
            nVertLevels: Number of vertical levels

        Prognostic variables (required):
            u: Edge-normal velocity (nEdges+1, nVertLevels)
            w: Vertical velocity (nCells+1, nVertLevels+1)
            rho_zz: Jacobian-weighted density (nCells+1, nVertLevels)
            theta_m: Modified potential temperature (nCells+1, nVertLevels)
            scalars: Scalar tracers (nCells+1, nVertLevels, num_scalars)

        Derived/diagnostic variables (optional, computed during timestep):
            ru: rho*u (nEdges+1, nVertLevels)
            rw: rho*w (nCells+1, nVertLevels+1)
            exner: Exner function (nCells+1, nVertLevels)
            pressure_p: Pressure perturbation (nCells+1, nVertLevels)
            rho_p: Density perturbation (nCells+1, nVertLevels)
            rtheta_p: rho*theta perturbation (nCells+1, nVertLevels)

        Perturbation variables for acoustic substeps:
            ru_p: ru perturbation (nEdges+1, nVertLevels)
            rw_p: rw perturbation (nCells+1, nVertLevels+1)
            rho_pp: Density double perturbation (nCells+1, nVertLevels)
            rtheta_pp: rho*theta double perturbation (nCells+1, nVertLevels)

        Time-averaged fluxes for scalar transport:
            ruAvg: Time-averaged ru (nEdges+1, nVertLevels)
            wwAvg: Time-averaged vertical flux (nCells+1, nVertLevels+1)

        Reconstruction support arrays (optional):
            coeffs_reconstruct: Reconstruction coefficients for mpas_reconstruct_2d
            latCell: Cell latitude (radians)
            lonCell: Cell longitude (radians)
    """

    # Dimensions (required)
    nCells: int
    nEdges: int
    nVertLevels: int

    # Prognostic variables (required)
    u: torch.Tensor
    w: torch.Tensor
    rho_zz: torch.Tensor
    theta_m: torch.Tensor
    scalars: torch.Tensor

    # Derived variables (optional, computed during timestep)
    ru: torch.Tensor | None = None
    rw: torch.Tensor | None = None
    exner: torch.Tensor | None = None
    pressure_p: torch.Tensor | None = None
    rho_p: torch.Tensor | None = None
    rtheta_p: torch.Tensor | None = None

    # Perturbation variables for acoustic substeps
    ru_p: torch.Tensor | None = None
    rw_p: torch.Tensor | None = None
    rho_pp: torch.Tensor | None = None
    rtheta_pp: torch.Tensor | None = None

    # Time-averaged fluxes for scalar transport
    ruAvg: torch.Tensor | None = None
    wwAvg: torch.Tensor | None = None

    # Reconstruction support arrays
    coeffs_reconstruct: torch.Tensor | None = None
    latCell: torch.Tensor | None = None
    lonCell: torch.Tensor | None = None

    def clone(self) -> "DycoreState":
        """
        Create a deep copy of the state.

        All tensors are cloned to ensure independence from the original.
        Modifying the cloned state will not affect the original.

        Returns:
            New DycoreState with cloned tensors.
        """

        def _clone_optional(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.clone() if t is not None else None

        return DycoreState(
            nCells=self.nCells,
            nEdges=self.nEdges,
            nVertLevels=self.nVertLevels,
            u=self.u.clone(),
            w=self.w.clone(),
            rho_zz=self.rho_zz.clone(),
            theta_m=self.theta_m.clone(),
            scalars=self.scalars.clone(),
            ru=_clone_optional(self.ru),
            rw=_clone_optional(self.rw),
            exner=_clone_optional(self.exner),
            pressure_p=_clone_optional(self.pressure_p),
            rho_p=_clone_optional(self.rho_p),
            rtheta_p=_clone_optional(self.rtheta_p),
            ru_p=_clone_optional(self.ru_p),
            rw_p=_clone_optional(self.rw_p),
            rho_pp=_clone_optional(self.rho_pp),
            rtheta_pp=_clone_optional(self.rtheta_pp),
            ruAvg=_clone_optional(self.ruAvg),
            wwAvg=_clone_optional(self.wwAvg),
            coeffs_reconstruct=_clone_optional(self.coeffs_reconstruct),
            latCell=_clone_optional(self.latCell),
            lonCell=_clone_optional(self.lonCell),
        )

    def to(self, device: torch.device | str) -> "DycoreState":
        """
        Return a new DycoreState with tensor fields moved to ``device``.
        """
        target_device = torch.device(device)
        kwargs = {}
        for field in fields(self):
            value = getattr(self, field.name)
            kwargs[field.name] = (
                value.to(target_device) if isinstance(value, torch.Tensor) else value
            )
        return DycoreState(**kwargs)  # type: ignore[arg-type]

    def validate(self) -> bool:
        """
        Validate state array shapes and dtypes.

        Raises:
            ValueError: If any array has incorrect shape or dtype.

        Returns:
            True if all validations pass.
        """
        nC, nE, nV = self.nCells, self.nEdges, self.nVertLevels

        # Check required field shapes
        _check(
            self.u.shape == (nE + 1, nV), f"u shape {self.u.shape} != ({nE + 1}, {nV})"
        )
        _check(
            self.w.shape == (nC + 1, nV + 1),
            f"w shape {self.w.shape} != ({nC + 1}, {nV + 1})",
        )
        _check(
            self.rho_zz.shape == (nC + 1, nV),
            f"rho_zz shape {self.rho_zz.shape} != ({nC + 1}, {nV})",
        )
        _check(
            self.theta_m.shape == (nC + 1, nV),
            f"theta_m shape {self.theta_m.shape} != ({nC + 1}, {nV})",
        )
        _check(
            self.scalars.ndim == 3 and self.scalars.shape[:2] == (nC + 1, nV),
            f"scalars shape {self.scalars.shape} has wrong first two dims, expected ({nC + 1}, {nV}, ...)",
        )
        _check(
            self.scalars.shape[2] > 0,
            f"scalars has 0 tracers (shape={self.scalars.shape}), need at least 1 tracer",
        )

        # Check required field dtypes
        _check(self.u.dtype == DTYPE, f"u dtype {self.u.dtype} != {DTYPE}")
        _check(self.w.dtype == DTYPE, f"w dtype {self.w.dtype} != {DTYPE}")
        _check(
            self.rho_zz.dtype == DTYPE, f"rho_zz dtype {self.rho_zz.dtype} != {DTYPE}"
        )
        _check(
            self.theta_m.dtype == DTYPE,
            f"theta_m dtype {self.theta_m.dtype} != {DTYPE}",
        )
        _check(
            self.scalars.dtype == DTYPE,
            f"scalars dtype {self.scalars.dtype} != {DTYPE}",
        )

        # Check optional field shapes and dtypes if present
        if self.ru is not None:
            _check(
                self.ru.shape == (nE + 1, nV),
                f"ru shape {self.ru.shape} != ({nE + 1}, {nV})",
            )
            _check(self.ru.dtype == DTYPE, f"ru dtype {self.ru.dtype} != {DTYPE}")
        if self.rw is not None:
            _check(
                self.rw.shape == (nC + 1, nV + 1),
                f"rw shape {self.rw.shape} != ({nC + 1}, {nV + 1})",
            )
            _check(self.rw.dtype == DTYPE, f"rw dtype {self.rw.dtype} != {DTYPE}")
        if self.exner is not None:
            _check(
                self.exner.shape == (nC + 1, nV),
                f"exner shape {self.exner.shape} != ({nC + 1}, {nV})",
            )
            _check(
                self.exner.dtype == DTYPE, f"exner dtype {self.exner.dtype} != {DTYPE}"
            )
        if self.pressure_p is not None:
            _check(
                self.pressure_p.shape == (nC + 1, nV),
                f"pressure_p shape {self.pressure_p.shape} != ({nC + 1}, {nV})",
            )
            _check(
                self.pressure_p.dtype == DTYPE,
                f"pressure_p dtype {self.pressure_p.dtype} != {DTYPE}",
            )
        if self.rho_p is not None:
            _check(
                self.rho_p.shape == (nC + 1, nV),
                f"rho_p shape {self.rho_p.shape} != ({nC + 1}, {nV})",
            )
            _check(
                self.rho_p.dtype == DTYPE, f"rho_p dtype {self.rho_p.dtype} != {DTYPE}"
            )
        if self.rtheta_p is not None:
            _check(
                self.rtheta_p.shape == (nC + 1, nV),
                f"rtheta_p shape {self.rtheta_p.shape} != ({nC + 1}, {nV})",
            )
            _check(
                self.rtheta_p.dtype == DTYPE,
                f"rtheta_p dtype {self.rtheta_p.dtype} != {DTYPE}",
            )
        if self.ru_p is not None:
            _check(
                self.ru_p.shape == (nE + 1, nV),
                f"ru_p shape {self.ru_p.shape} != ({nE + 1}, {nV})",
            )
            _check(self.ru_p.dtype == DTYPE, f"ru_p dtype {self.ru_p.dtype} != {DTYPE}")
        if self.rw_p is not None:
            _check(
                self.rw_p.shape == (nC + 1, nV + 1),
                f"rw_p shape {self.rw_p.shape} != ({nC + 1}, {nV + 1})",
            )
            _check(self.rw_p.dtype == DTYPE, f"rw_p dtype {self.rw_p.dtype} != {DTYPE}")
        if self.rho_pp is not None:
            _check(
                self.rho_pp.shape == (nC + 1, nV),
                f"rho_pp shape {self.rho_pp.shape} != ({nC + 1}, {nV})",
            )
            _check(
                self.rho_pp.dtype == DTYPE,
                f"rho_pp dtype {self.rho_pp.dtype} != {DTYPE}",
            )
        if self.rtheta_pp is not None:
            _check(
                self.rtheta_pp.shape == (nC + 1, nV),
                f"rtheta_pp shape {self.rtheta_pp.shape} != ({nC + 1}, {nV})",
            )
            _check(
                self.rtheta_pp.dtype == DTYPE,
                f"rtheta_pp dtype {self.rtheta_pp.dtype} != {DTYPE}",
            )
        if self.ruAvg is not None:
            _check(
                self.ruAvg.shape == (nE + 1, nV),
                f"ruAvg shape {self.ruAvg.shape} != ({nE + 1}, {nV})",
            )
            _check(
                self.ruAvg.dtype == DTYPE, f"ruAvg dtype {self.ruAvg.dtype} != {DTYPE}"
            )
        if self.wwAvg is not None:
            _check(
                self.wwAvg.shape == (nC + 1, nV + 1),
                f"wwAvg shape {self.wwAvg.shape} != ({nC + 1}, {nV + 1})",
            )
            _check(
                self.wwAvg.dtype == DTYPE, f"wwAvg dtype {self.wwAvg.dtype} != {DTYPE}"
            )

        if self.latCell is not None:
            _check(
                self.latCell.ndim == 1,
                f"latCell must be rank-1, got {self.latCell.shape}",
            )
            _check(
                self.latCell.shape[0] in (nC, nC + 1),
                f"latCell first dim {self.latCell.shape[0]} must be nCells or nCells+1",
            )
            _check(
                self.latCell.dtype == DTYPE,
                f"latCell dtype {self.latCell.dtype} != {DTYPE}",
            )

        if self.lonCell is not None:
            _check(
                self.lonCell.ndim == 1,
                f"lonCell must be rank-1, got {self.lonCell.shape}",
            )
            _check(
                self.lonCell.shape[0] in (nC, nC + 1),
                f"lonCell first dim {self.lonCell.shape[0]} must be nCells or nCells+1",
            )
            _check(
                self.lonCell.dtype == DTYPE,
                f"lonCell dtype {self.lonCell.dtype} != {DTYPE}",
            )

        if self.coeffs_reconstruct is not None:
            _check(
                self.coeffs_reconstruct.ndim == 3,
                f"coeffs_reconstruct must be rank-3, got {self.coeffs_reconstruct.shape}",
            )
            cshape = self.coeffs_reconstruct.shape
            has_comp3 = 3 in cshape
            has_cells = (nC in cshape) or (nC + 1 in cshape)
            _check(
                has_comp3 and has_cells,
                f"coeffs_reconstruct shape {cshape} must include component size 3 and cell dimension nCells/nCells+1",
            )
            _check(
                self.coeffs_reconstruct.dtype == DTYPE,
                f"coeffs_reconstruct dtype {self.coeffs_reconstruct.dtype} != {DTYPE}",
            )

        return True


def _load_optional(
    data: dict,
    name: str,
    default_shape: tuple,
    dtype=None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Load optional field from checkpoint data, returning zeros if not present.

    Args:
        data: Dictionary of checkpoint data (already loaded tensors)
        name: Name of the variable to load
        default_shape: Shape to use if variable not found
        dtype: Data type (defaults to DTYPE = torch.float64)
        device: Device to place tensor on (default 'cpu')

    Returns:
        Tensor from checkpoint if present, otherwise zeros with default_shape
    """
    target_device = torch.device(device)
    if dtype is None:
        dtype = DTYPE
    if name in data:
        arr = data[name]
        if arr.shape == default_shape:
            return cast(torch.Tensor, arr.to(device=target_device, dtype=dtype))
    return torch.zeros(default_shape, dtype=dtype, device=target_device)


def _load_with_save_fallback(
    data: dict,
    primary_name: str,
    save_name: str,
    expected_shape: tuple,
    strict: bool,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Load field with explicit preference for primary key, fallback to _save variant.

    Uses explicit key presence check (NOT sum()==0 which masks legitimate zeros).
    Order of preference:
    1. primary_name if present with correct shape
    2. save_name if present with correct shape
    3. zeros

    Args:
        data: Dictionary of checkpoint data
        primary_name: Primary field name (e.g., 'ru')
        save_name: Save variant name (e.g., 'ru_save')
        expected_shape: Expected tensor shape
        strict: If True, warn when using _save variant
        device: Device to place tensor on

    Returns:
        Tensor from checkpoint or zeros
    """
    target_device = torch.device(device)
    # Check primary key first
    if primary_name in data:
        arr = data[primary_name]
        if arr.shape == expected_shape:
            return cast(torch.Tensor, arr.to(device=target_device, dtype=DTYPE))

    # Fallback to _save variant
    if save_name in data:
        arr = data[save_name]
        if arr.shape == expected_shape:
            if strict:
                warnings.warn(
                    f"Using '{save_name}' instead of '{primary_name}' from checkpoint. "
                    f"This indicates the checkpoint may be from a different timestep stage.",
                    RuntimeWarning,
                )
            return cast(torch.Tensor, arr.to(device=target_device, dtype=DTYPE))

    # Neither found - return zeros
    return torch.zeros(expected_shape, dtype=DTYPE, device=target_device)


def load_state_from_checkpoint(
    filepath: str,
    strict: bool = True,
    config: dict | None = None,
    required_fields: list[str] | None = None,
    device: torch.device | str = "cpu",
) -> "DycoreState":
    """
    Load dycore state from a checkpoint NetCDF file.

    Loads prognostic variables (u, w, rho_zz, theta_m, scalars) and optional
    diagnostic and perturbation variables from an MPAS NetCDF file.

    The checkpoint file must contain at minimum:
        - nCells, nEdges, nVertLevels (scalar integers)

    Prognostic variables are loaded if present, otherwise initialized to zeros.
    Arrays are padded to add the required leading dimension for 1-based indexing.

    Args:
        filepath: Path to NetCDF checkpoint file
        strict: If True (default), raise ValueError for missing required prognostic
                variables (u, w, rho_zz, theta_m, scalars). If False, use zeros with warning.
                Use strict=False ONLY for testing/debugging.
        config: Optional configuration dictionary. If provided and config['config_apply_lbcs']
                is True, regional boundary fields are also required.
        required_fields: Optional list of additional field names that must be present.
                Combined with CORE_REQUIRED_FIELDS and REGIONAL_REQUIRED_FIELDS (if applicable).
        device: Device for loaded tensors (default CPU).

    Returns:
        DycoreState populated from checkpoint data

    Raises:
        FileNotFoundError: If checkpoint file does not exist
        ValueError: If required dimension variables are missing, or if strict=True
                    and a required prognostic field is missing
        KeyError: If required dimension variables are missing (backward compatibility)
    """
    from mpas_pytorch.utils import load_checkpoint

    target_device = torch.device(device)
    data = load_checkpoint(filepath, device=target_device)

    # Determine required fields based on config
    must_have = set(CORE_REQUIRED_FIELDS)
    if config and config.get("config_apply_lbcs", False):
        must_have.update(REGIONAL_REQUIRED_FIELDS)
    if required_fields:
        must_have.update(required_fields)

    # Get list of available fields in checkpoint
    available_fields = set(data.keys()) if data else set()

    # Validate required fields BEFORE zero-filling
    missing_required = [f for f in must_have if f not in available_fields]
    if missing_required and strict:
        raise MpasConfigError(
            f"Missing required fields for exact-repro run: {missing_required}. "
            f"These cannot be zero-filled. Either provide checkpoint with these fields "
            f"or set strict=False to allow zero-fill (not recommended for production)."
        )
    elif missing_required:
        warnings.warn(
            f"Missing required fields {missing_required} will be zero-filled. "
            "Results may not satisfy the full-precision input contract.",
            RuntimeWarning,
        )

    # Extract dimensions (required)
    nCells = int(data["nCells"].item())
    nEdges = int(data["nEdges"].item())
    nVertLevels = int(data["nVertLevels"].item())

    # Checkpoint arrays already include 1-based indexing padding, so shapes are:
    # - Edge arrays: (nEdges+1, nVertLevels)
    # - Cell arrays: (nCells+1, nVertLevels)
    # - Cell w-level arrays: (nCells+1, nVertLevels+1)
    # - Scalars: (nCells+1, nVertLevels, num_scalars)

    def get_edge_array(
        names: list, fallback: bool = True, is_required: bool = False
    ) -> torch.Tensor | None:
        """Get edge array (nEdges+1, nVertLevels), fallback to zeros if not found."""
        expected_shape = (nEdges + 1, nVertLevels)
        for name in names:
            if name in data:
                arr = data[name]
                if arr.shape == expected_shape:
                    return cast(torch.Tensor, arr.to(device=target_device, dtype=DTYPE))
        if fallback:
            # Check if this is a required field and strict mode is enabled
            if strict and is_required:
                raise KeyError(
                    f"Required prognostic field {names} not found or wrong shape in checkpoint. "
                    f"Expected shape {expected_shape}. Use strict=False for testing only."
                )
            warnings.warn(
                f"Variables {names} not found or wrong shape in checkpoint, "
                f"using zeros for shape {expected_shape}",
                RuntimeWarning,
            )
            return torch.zeros(expected_shape, dtype=DTYPE, device=target_device)
        return None

    def get_cell_array(
        names: list, fallback: bool = True, is_required: bool = False
    ) -> torch.Tensor | None:
        """Get cell array (nCells+1, nVertLevels), fallback to zeros if not found."""
        expected_shape = (nCells + 1, nVertLevels)
        for name in names:
            if name in data:
                arr = data[name]
                if arr.shape == expected_shape:
                    return cast(torch.Tensor, arr.to(device=target_device, dtype=DTYPE))
        if fallback:
            # Check if this is a required field and strict mode is enabled
            if strict and is_required:
                raise KeyError(
                    f"Required prognostic field {names} not found or wrong shape in checkpoint. "
                    f"Expected shape {expected_shape}. Use strict=False for testing only."
                )
            warnings.warn(
                f"Variables {names} not found or wrong shape in checkpoint, "
                f"using zeros for shape {expected_shape}",
                RuntimeWarning,
            )
            return torch.zeros(expected_shape, dtype=DTYPE, device=target_device)
        return None

    def get_cell_w_array(
        names: list, fallback: bool = True, is_required: bool = False
    ) -> torch.Tensor | None:
        """Get cell w-level array (nCells+1, nVertLevels+1), fallback to zeros if not found."""
        expected_shape = (nCells + 1, nVertLevels + 1)
        for name in names:
            if name in data:
                arr = data[name]
                if arr.shape == expected_shape:
                    return cast(torch.Tensor, arr.to(device=target_device, dtype=DTYPE))
        if fallback:
            # Check if this is a required field and strict mode is enabled
            if strict and is_required:
                raise KeyError(
                    f"Required prognostic field {names} not found or wrong shape in checkpoint. "
                    f"Expected shape {expected_shape}. Use strict=False for testing only."
                )
            warnings.warn(
                f"Variables {names} not found or wrong shape in checkpoint, "
                f"using zeros for shape {expected_shape}",
                RuntimeWarning,
            )
            return torch.zeros(expected_shape, dtype=DTYPE, device=target_device)
        return None

    # Extract prognostic variables with fallbacks to zeros
    # Required prognostic fields will raise KeyError if missing and strict=True
    u = get_edge_array(["u", "u_new"], is_required=True)
    w = get_cell_w_array(["w", "w_new"], is_required=True)
    rho_zz = get_cell_array(["rho_zz", "rho_zz_new"], is_required=True)
    theta_m = get_cell_array(["theta_m", "theta_m_new"], is_required=True)

    # Scalars: checkpoint shape is (nCells+1, nVertLevels, num_scalars)
    # Handle num_scalars with strict mode validation
    if "num_scalars" in data:
        num_scalars = int(data["num_scalars"].item())
    else:
        # num_scalars is missing - behavior depends on strict mode
        if strict:
            raise MpasConfigError(
                "Required field 'num_scalars' not found in checkpoint. "
                "This can cause silent data truncation if tracers are present. "
                "Either add num_scalars to checkpoint or use strict=False to infer from scalars array shape."
            )
        # In non-strict mode, try to infer from scalars array shape
        if (
            "scalars" in data
            and hasattr(data["scalars"], "shape")
            and data["scalars"].ndim == 3
        ):
            inferred_num_scalars = data["scalars"].shape[2]
            warnings.warn(
                f"num_scalars not found in checkpoint, inferring from scalars array shape: {inferred_num_scalars}. "
                f"This may indicate a malformed checkpoint. Consider adding num_scalars explicitly.",
                RuntimeWarning,
            )
            num_scalars = inferred_num_scalars
        else:
            # Cannot infer, fall back to 1
            warnings.warn(
                "num_scalars not found in checkpoint and cannot infer from scalars array. "
                "Defaulting to 1 tracer. This may cause data loss if checkpoint has more tracers.",
                RuntimeWarning,
            )
            num_scalars = 1
    scalars_shape = (nCells + 1, nVertLevels, num_scalars)
    if "scalars" in data:
        scalars_raw = data["scalars"]
        # Checkpoint stores as (nCells+1, nVertLevels, num_scalars)
        # Validate ALL dimensions, not just the first
        if scalars_raw.ndim != 3:
            # Wrong number of dimensions
            if strict:
                raise KeyError(
                    f"Required prognostic field 'scalars' has wrong ndim={scalars_raw.ndim} "
                    f"in checkpoint. Expected 3D array (nCells+1, nVertLevels, num_scalars). "
                    f"Use strict=False for testing only."
                )
            warnings.warn(
                f"Variable 'scalars' has wrong ndim={scalars_raw.ndim} in checkpoint, "
                f"using zeros for shape {scalars_shape}",
                RuntimeWarning,
            )
            scalars = torch.zeros(scalars_shape, dtype=DTYPE, device=target_device)
        elif scalars_raw.shape[0] != nCells + 1:
            # Wrong nCells dimension
            if strict:
                raise KeyError(
                    f"Required prognostic field 'scalars' has wrong shape {scalars_raw.shape} "
                    f"in checkpoint. Expected shape[0]={nCells + 1} (nCells+1), got {scalars_raw.shape[0]}. "
                    f"Use strict=False for testing only."
                )
            warnings.warn(
                f"Variable 'scalars' has wrong shape {scalars_raw.shape} in checkpoint, "
                f"using zeros for shape {scalars_shape}",
                RuntimeWarning,
            )
            scalars = torch.zeros(scalars_shape, dtype=DTYPE, device=target_device)
        elif scalars_raw.shape[1] != nVertLevels:
            # Wrong nVertLevels dimension - this is a critical error
            if strict:
                raise KeyError(
                    f"Required prognostic field 'scalars' has wrong shape {scalars_raw.shape} "
                    f"in checkpoint. Expected shape[1]={nVertLevels} (nVertLevels), got {scalars_raw.shape[1]}. "
                    f"Use strict=False for testing only."
                )
            warnings.warn(
                f"Variable 'scalars' has wrong nVertLevels {scalars_raw.shape[1]} in checkpoint, "
                f"expected {nVertLevels}. Using zeros for shape {scalars_shape}",
                RuntimeWarning,
            )
            scalars = torch.zeros(scalars_shape, dtype=DTYPE, device=target_device)
        else:
            # Shape[0] and shape[1] are correct
            # Validate num_scalars matches actual scalars array shape
            if scalars_raw.shape[2] != num_scalars:
                if strict:
                    raise MpasConfigError(
                        f"Mismatch between num_scalars ({num_scalars}) and scalars array shape[2] "
                        f"({scalars_raw.shape[2]}). This indicates a malformed checkpoint that could "
                        f"cause silent data truncation. Fix the checkpoint or use strict=False."
                    )
                warnings.warn(
                    f"Scalars has {scalars_raw.shape[2]} tracers, expected {num_scalars}. "
                    f"Using actual count from checkpoint. This may indicate data corruption.",
                    RuntimeWarning,
                )
            scalars = scalars_raw.to(device=target_device, dtype=DTYPE)
    else:
        if strict:
            raise KeyError(
                "Required prognostic field 'scalars' not found in checkpoint. "
                "Use strict=False for testing only."
            )
        warnings.warn(
            f"Variable 'scalars' not found in checkpoint, "
            f"using zeros for shape {scalars_shape}",
            RuntimeWarning,
        )
        scalars = torch.zeros(scalars_shape, dtype=DTYPE, device=target_device)

    # Optional derived/perturbation fields from checkpoint
    ru = get_edge_array(["ru", "ru_save"], fallback=False)
    rw = get_cell_w_array(["rw", "rw_save"], fallback=False)

    exner = get_cell_array(["exner"], fallback=False)
    pressure_p = get_cell_array(["pressure_p"], fallback=False)
    rho_p = get_cell_array(["rho_p", "rho_p_save"], fallback=False)
    rtheta_p = get_cell_array(["rtheta_p", "rtheta_p_save"], fallback=False)

    ru_p = get_edge_array(["ru_p"], fallback=False)
    rw_p = get_cell_w_array(["rw_p"], fallback=False)
    rho_pp = get_cell_array(["rho_pp"], fallback=False)
    rtheta_pp = get_cell_array(["rtheta_pp"], fallback=False)

    ruAvg = get_edge_array(["ruAvg"], fallback=False)
    wwAvg = get_cell_w_array(["wwAvg"], fallback=False)

    coeffs_reconstruct = None
    if "coeffs_reconstruct" in data:
        coeffs_raw = data["coeffs_reconstruct"]
        if coeffs_raw.ndim == 3:
            coeffs_reconstruct = coeffs_raw.to(device=target_device, dtype=DTYPE)

    latCell = None
    if "latCell" in data and data["latCell"].ndim == 1:
        latCell = data["latCell"].to(device=target_device, dtype=DTYPE)

    lonCell = None
    if "lonCell" in data and data["lonCell"].ndim == 1:
        lonCell = data["lonCell"].to(device=target_device, dtype=DTYPE)

    assert u is not None and w is not None
    assert rho_zz is not None and theta_m is not None

    return DycoreState(
        nCells=nCells,
        nEdges=nEdges,
        nVertLevels=nVertLevels,
        u=u,
        w=w,
        rho_zz=rho_zz,
        theta_m=theta_m,
        scalars=scalars,
        ru=ru,
        rw=rw,
        exner=exner,
        pressure_p=pressure_p,
        rho_p=rho_p,
        rtheta_p=rtheta_p,
        ru_p=ru_p,
        rw_p=rw_p,
        rho_pp=rho_pp,
        rtheta_pp=rtheta_pp,
        ruAvg=ruAvg,
        wwAvg=wwAvg,
        coeffs_reconstruct=coeffs_reconstruct,
        latCell=latCell,
        lonCell=lonCell,
    )


def load_full_checkpoint(
    filepath: str, strict: bool = True, device: str = "cpu"
) -> dict:
    """
    Load all fields from a checkpoint NetCDF file for full RK3 integration.

    This comprehensive loader returns a dictionary with all arrays needed by
    the RK3 time integration, including:
    - Dimensions (nCells, nEdges, nVertLevels, maxEdges, maxEdges2, etc.)
    - Prognostic variables (u, w, rho_zz, theta_m, scalars)
    - Derived/diagnostic variables (ru, rw, exner, pressure_p, etc.)
    - Acoustic perturbation variables (ru_p, rw_p, rho_pp, rtheta_pp)
    - Perturbation coefficients for small-step-perturbation (zb, zb3, zb_cell, zb3_cell)
    - Tendency arrays (tend_ru, tend_rho, tend_rt, tend_rw)
    - Base state arrays (rho_base, rtheta_base, etc.)
    - Mesh connectivity arrays (edgesOnCell, cellsOnCell, etc.)

    Optional fields are initialized to zeros if not present in checkpoint.

    Args:
        filepath: Path to NetCDF checkpoint file
        strict: If True (default), raise KeyError for missing required prognostic
                variables. If False, use zeros with warning (for testing only).
        device: Device to place tensors on (default 'cpu')

    Returns:
        Dictionary containing all checkpoint data as torch tensors.
        Dimensions are returned as Python ints.

    Raises:
        FileNotFoundError: If checkpoint file does not exist
        KeyError: If required dimension variables are missing
    """
    from mpas_pytorch.utils import load_checkpoint

    data = load_checkpoint(filepath, device=device)

    # Extract dimensions - use explicit values if present, otherwise infer from arrays
    def _get_dim(
        name: str,
        infer_arrays: list[Any] | None = None,
        dim_idx: int = 0,
        subtract_one: bool = True,
    ) -> int | None:
        """Get dimension from checkpoint, inferring from arrays if not explicit."""
        if name in data:
            return int(data[name].item())
        if infer_arrays:
            for arr_name in infer_arrays:
                if arr_name in data and hasattr(data[arr_name], "shape"):
                    val = data[arr_name].shape[dim_idx]
                    return int(val - 1 if subtract_one else val)
        return None

    # Required dimensions - infer from arrays if not explicit
    nCells = _get_dim(
        "nCells", ["rho_zz", "theta_m", "w", "exner"], dim_idx=0, subtract_one=True
    )
    if nCells is None:
        raise KeyError(
            "Cannot determine nCells from checkpoint - no nCells dimension or cell arrays found"
        )

    nVertLevels = _get_dim(
        "nVertLevels",
        ["rho_zz", "theta_m", "u", "exner"],
        dim_idx=1,
        subtract_one=False,
    )
    if nVertLevels is None:
        raise KeyError(
            "Cannot determine nVertLevels from checkpoint - no nVertLevels dimension or level arrays found"
        )

    # nEdges may not be present in cell-only checkpoints
    nEdges = _get_dim(
        "nEdges", ["u", "ru", "ru_p", "cellsOnEdge"], dim_idx=0, subtract_one=True
    )

    # Check if any edge arrays are present in the checkpoint
    edge_array_names = [
        "u",
        "ru",
        "ru_p",
        "ru_save",
        "ruAvg",
        "cellsOnEdge",
        "edgesOnCell",
        "fEdge",
        "specZoneMaskEdge",
        "bdyMaskEdge",
        "dvEdge",
        "dcEdge",
    ]
    has_edge_arrays = any(name in data for name in edge_array_names)

    if nEdges is None:
        if has_edge_arrays and strict:
            # Edge arrays present but nEdges cannot be determined - this is an error
            present_edge_arrays = [name for name in edge_array_names if name in data]
            raise MpasConfigError(
                f"Cannot determine nEdges from checkpoint but edge arrays are present: {present_edge_arrays}. "
                f"Either add 'nEdges' dimension to checkpoint or use strict=False. "
                f"The 3*nCells heuristic is incorrect for real meshes and will cause shape mismatches."
            )
        elif has_edge_arrays:
            # Non-strict mode: use heuristic with strong warning
            nEdges = 3 * nCells
            warnings.warn(
                f"CAUTION: nEdges not found in checkpoint but edge arrays are present. "
                f"Using approximate heuristic nEdges = 3*nCells = {nEdges}. "
                f"This is generally incorrect because an unstructured mesh need not have an exact nEdges/nCells ratio of 3. "
                f"Results may have incorrect shapes. Consider adding 'nEdges' to checkpoint.",
                RuntimeWarning,
            )
        else:
            # No edge arrays present - set nEdges = 0 (cell-only checkpoint)
            nEdges = 0
            # Note: This is fine - downstream code will create empty edge arrays

    # Optional dimensions with defaults
    maxEdges = _get_dim("maxEdges", ["edgesOnCell"], dim_idx=1, subtract_one=False)
    if maxEdges is None:
        maxEdges = 10
    maxEdges2 = _get_dim(
        "maxEdges2", ["edgesOnEdge", "weightsOnEdge"], dim_idx=1, subtract_one=False
    )
    if maxEdges2 is None:
        maxEdges2 = 2 * maxEdges
    nVertices = _get_dim("nVertices", ["cellsOnVertex"], dim_idx=0, subtract_one=True)
    if nVertices is None:
        nVertices = nCells
    vertexDegree = _get_dim(
        "vertexDegree", ["cellsOnVertex"], dim_idx=1, subtract_one=False
    )
    if vertexDegree is None:
        vertexDegree = 3
    # Handle num_scalars with strict mode validation
    num_scalars_explicit = "num_scalars" in data
    if num_scalars_explicit:
        num_scalars = int(data["num_scalars"].item())
    else:
        # num_scalars is missing - behavior depends on strict mode
        if strict:
            raise MpasConfigError(
                "Required field 'num_scalars' not found in checkpoint. "
                "This can cause silent data truncation if tracers are present. "
                "Either add num_scalars to checkpoint or use strict=False to infer from scalars array shape."
            )
        # In non-strict mode, try to infer from scalars array shape
        if (
            "scalars" in data
            and hasattr(data["scalars"], "shape")
            and data["scalars"].ndim == 3
        ):
            num_scalars = data["scalars"].shape[2]
            warnings.warn(
                f"num_scalars not found in checkpoint, inferring from scalars array shape: {num_scalars}. "
                f"This may indicate a malformed checkpoint. Consider adding num_scalars explicitly.",
                RuntimeWarning,
            )
        else:
            # Cannot infer, fall back to 1
            warnings.warn(
                "num_scalars not found in checkpoint and cannot infer from scalars array. "
                "Defaulting to 1 tracer. This may cause data loss if checkpoint has more tracers.",
                RuntimeWarning,
            )
            num_scalars = 1

    result: dict[str, Any] = {
        # Dimensions
        "nCells": nCells,
        "nEdges": nEdges,
        "nVertLevels": nVertLevels,
        "maxEdges": maxEdges,
        "maxEdges2": maxEdges2,
        "nVertices": nVertices,
        "vertexDegree": vertexDegree,
        "num_scalars": num_scalars,
    }

    # Define shape templates for different array types
    edge_shape = (nEdges + 1, nVertLevels)
    cell_shape = (nCells + 1, nVertLevels)
    cell_w_shape = (nCells + 1, nVertLevels + 1)
    scalars_shape = (nCells + 1, nVertLevels, num_scalars)

    # Perturbation coefficient shapes for small-step-perturbation/large-step-recovery
    # zb/zb3: (nEdges+1, 2, nVertLevels+1) - edge-based
    # zb_cell/zb3_cell: (nCells+1, maxEdges, nVertLevels+1) - cell-based
    zb_edge_shape = (nEdges + 1, 2, nVertLevels + 1)
    zb_cell_shape = (nCells + 1, maxEdges, nVertLevels + 1)

    # ==========================================================================
    # Required prognostic variables
    # ==========================================================================

    def _load_required(
        names: list, expected_shape: tuple, field_name: str
    ) -> torch.Tensor:
        """Load required field, raising error if missing and strict=True."""
        for name in names:
            if name in data and data[name].shape == expected_shape:
                return cast(torch.Tensor, data[name].to(DTYPE).to(device))
        if strict:
            raise KeyError(
                f"Required prognostic field {field_name} (tried {names}) not found or wrong shape "
                f"in checkpoint. Expected shape {expected_shape}. Use strict=False for testing only."
            )
        warnings.warn(
            f"Variables {names} not found or wrong shape in checkpoint, "
            f"using zeros for shape {expected_shape}",
            RuntimeWarning,
        )
        return torch.zeros(expected_shape, dtype=DTYPE, device=device)

    result["u"] = _load_required(["u", "u_new"], edge_shape, "u")
    result["w"] = _load_required(["w", "w_new"], cell_w_shape, "w")
    result["rho_zz"] = _load_required(["rho_zz", "rho_zz_new"], cell_shape, "rho_zz")
    result["theta_m"] = _load_required(
        ["theta_m", "theta_m_new"], cell_shape, "theta_m"
    )

    # Scalars: special handling for variable tracer count
    if "scalars" in data:
        scalars_raw = data["scalars"]
        if (
            scalars_raw.ndim == 3
            and scalars_raw.shape[0] == nCells + 1
            and scalars_raw.shape[1] == nVertLevels
        ):
            # Validate num_scalars matches actual scalars array shape (only if num_scalars was explicit)
            if num_scalars_explicit and scalars_raw.shape[2] != num_scalars:
                if strict:
                    raise MpasConfigError(
                        f"Mismatch between num_scalars ({num_scalars}) and scalars array shape[2] "
                        f"({scalars_raw.shape[2]}). This indicates a malformed checkpoint that could "
                        f"cause silent data truncation. Fix the checkpoint or use strict=False."
                    )
                warnings.warn(
                    f"Scalars has {scalars_raw.shape[2]} tracers, expected {num_scalars}. "
                    f"Using actual count from checkpoint. This may indicate data corruption.",
                    RuntimeWarning,
                )
            result["scalars"] = scalars_raw.to(DTYPE).to(device)
        else:
            if strict:
                raise KeyError(
                    f"Required prognostic field 'scalars' has wrong shape {scalars_raw.shape} "
                    f"in checkpoint. Expected (nCells+1, nVertLevels, num_scalars)."
                )
            warnings.warn(
                f"Scalars has wrong shape {scalars_raw.shape}, using zeros",
                RuntimeWarning,
            )
            result["scalars"] = torch.zeros(scalars_shape, dtype=DTYPE, device=device)
    else:
        if strict:
            raise KeyError(
                "Required prognostic field 'scalars' not found in checkpoint."
            )
        warnings.warn("Scalars not found in checkpoint, using zeros", RuntimeWarning)
        result["scalars"] = torch.zeros(scalars_shape, dtype=DTYPE, device=device)

    # ==========================================================================
    # Optional derived/diagnostic variables (default to zeros if missing)
    # ==========================================================================

    # Momentum flux - use explicit key preference (NOT sum()==0 which masks legitimate zeros)
    result["ru"] = _load_with_save_fallback(
        data, "ru", "ru_save", edge_shape, strict, device
    )
    result["rw"] = _load_with_save_fallback(
        data, "rw", "rw_save", cell_w_shape, strict, device
    )

    # Thermodynamic diagnostics
    result["exner"] = _load_optional(data, "exner", cell_shape, device=device)
    result["pressure_p"] = _load_optional(data, "pressure_p", cell_shape, device=device)
    result["rho_p"] = _load_with_save_fallback(
        data, "rho_p", "rho_p_save", cell_shape, strict, device
    )
    result["rtheta_p"] = _load_with_save_fallback(
        data, "rtheta_p", "rtheta_p_save", cell_shape, strict, device
    )

    # ==========================================================================
    # Acoustic perturbation variables (for acoustic-step acoustic step)
    # ==========================================================================

    result["ru_p"] = _load_optional(data, "ru_p", edge_shape, device=device)
    result["rw_p"] = _load_optional(data, "rw_p", cell_w_shape, device=device)
    result["rho_pp"] = _load_optional(data, "rho_pp", cell_shape, device=device)
    result["rtheta_pp"] = _load_optional(data, "rtheta_pp", cell_shape, device=device)

    # Time-averaged fluxes for scalar transport
    result["ruAvg"] = _load_optional(data, "ruAvg", edge_shape, device=device)
    result["wwAvg"] = _load_optional(data, "wwAvg", cell_w_shape, device=device)

    # ==========================================================================
    # Perturbation coefficients for small-step-perturbation (smlstep_pert) / large-step-recovery (recover_large_step)
    # ==========================================================================

    result["zb"] = _load_optional(data, "zb", zb_edge_shape, device=device)
    result["zb3"] = _load_optional(data, "zb3", zb_edge_shape, device=device)
    result["zb_cell"] = _load_optional(data, "zb_cell", zb_cell_shape, device=device)
    result["zb3_cell"] = _load_optional(data, "zb3_cell", zb_cell_shape, device=device)

    # ==========================================================================
    # Tendency arrays (from dynamics-tendency dyn_tend)
    # ==========================================================================

    result["tend_ru"] = _load_optional(data, "tend_ru", edge_shape, device=device)
    result["tend_rho"] = _load_optional(data, "tend_rho", cell_shape, device=device)
    result["tend_rt"] = _load_optional(data, "tend_rt", cell_shape, device=device)
    result["tend_rw"] = _load_optional(data, "tend_rw", cell_w_shape, device=device)

    # ==========================================================================
    # Base state arrays (for thermodynamic computations)
    # ==========================================================================

    result["rho_base"] = _load_optional(data, "rho_base", cell_shape, device=device)
    result["rtheta_base"] = _load_optional(
        data, "rtheta_base", cell_shape, device=device
    )
    result["theta_base"] = _load_optional(data, "theta_base", cell_shape, device=device)
    result["pressure_base"] = _load_optional(
        data, "pressure_base", cell_shape, device=device
    )
    result["exner_base"] = _load_optional(data, "exner_base", cell_shape, device=device)

    # Saved state arrays (for multi-stage integration)
    result["rho_p_save"] = _load_optional(data, "rho_p_save", cell_shape, device=device)
    result["rtheta_p_save"] = _load_optional(
        data, "rtheta_p_save", cell_shape, device=device
    )

    # ==========================================================================
    # Additional arrays for acoustic step / diagnostics
    # ==========================================================================

    # Vertical coordinate
    result["zz"] = _load_optional(data, "zz", cell_w_shape, device=device)

    # Coriolis
    result["fEdge"] = _load_optional(data, "fEdge", (nEdges + 1,), device=device)

    # Boundary masks (speczone)
    result["specZoneMaskEdge"] = _load_optional(
        data, "specZoneMaskEdge", (nEdges + 1,), device=device
    )
    result["specZoneMaskCell"] = _load_optional(
        data, "specZoneMaskCell", (nCells + 1,), device=device
    )
    result["bdyMaskEdge"] = _load_optional(
        data, "bdyMaskEdge", (nEdges + 1,), device=device
    )
    result["bdyMaskCell"] = _load_optional(
        data, "bdyMaskCell", (nCells + 1,), device=device
    )

    # Geometric coefficients
    result["areaCell"] = _load_optional(data, "areaCell", (nCells + 1,), device=device)
    result["dvEdge"] = _load_optional(data, "dvEdge", (nEdges + 1,), device=device)
    result["dcEdge"] = _load_optional(data, "dcEdge", (nEdges + 1,), device=device)

    # ==========================================================================
    # Pass through any additional arrays from checkpoint
    # ==========================================================================

    # Copy any remaining arrays not yet in result, coercing float tensors to
    # float64 to prevent silent precision loss from float32 leakage.
    for key, val in data.items():
        if key not in result and isinstance(val, torch.Tensor):
            # Convert scalar tensors (e.g., cf1/cf2/cf3) at the boundary so
            # downstream callers do not pay repeated .item() sync overhead.
            if val.ndim == 0:
                result[key] = _to_python_scalar_if_0d(val)
            elif val.dtype.is_floating_point:
                result[key] = val.to(DTYPE).to(device)
            else:
                result[key] = val.to(device)

    return result
