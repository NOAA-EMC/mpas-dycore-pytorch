"""Structured exceptions for MPAS-PyTorch public APIs.

These types give user-facing entrypoints stable, typed failures instead of bare
``ValueError``/``RuntimeError`` stack traces.

Compatibility note:
    ``MpasConfigError`` intentionally also subclasses ``ValueError``. That keeps
    existing callers that still catch ``ValueError`` working while new code can
    target the typed ``MpasError`` hierarchy. All configuration-adjacent public
    exceptions inherit from ``MpasConfigError`` for the same reason.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "MpasConfigError",
    "MpasDeviceError",
    "MpasError",
    "MpasMeshMismatchError",
    "MpasMissingFieldError",
    "MpasNumericalAgreementError",
    "MpasProvenanceError",
    "MpasStrictModeError",
    "MpasUnsupportedCapabilityError",
]


class MpasError(Exception):
    """Base class for user-facing MPAS-PyTorch failures.

    Catch this when callers want one stable umbrella type for public API errors.
    """


class MpasConfigError(MpasError, ValueError):
    """Raised for invalid runtime configuration or namelist values.

    This also subclasses :class:`ValueError` for backward compatibility with
    pre-hierarchy callers.

    Example: parsing ``output_interval="bogus"`` from ``streams.atmosphere``.
    """


class MpasProvenanceError(MpasConfigError):
    """Raised when a forbidden derived field is injected from external data.

    Example: loading ``theta_m`` from an input file before the init chain.
    """


class MpasMissingFieldError(MpasConfigError):
    """Raised when required IC or mesh fields are absent at load time.

    Example: ``cellsOnEdge`` missing from ``grid.nc`` during ``load_initial_state``.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        fields: Iterable[str] | None = None,
        source: str = "unknown",
    ) -> None:
        self.fields: tuple[str, ...] = tuple(fields or ())
        self.source: str = str(source)
        if message is None and self.fields:
            message = f"Missing required IC fields from {self.source}: {self.fields}"
        super().__init__(message if message is not None else "")


class MpasMeshMismatchError(MpasConfigError):
    """Raised when init.nc and grid/static mesh dimensions disagree.

    Example: ``nCells`` in ``init.nc`` is 10242 but ``grid.nc`` implies 40962.
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        dim_name: str | None = None,
        init_value: int | None = None,
        grid_value: int | None = None,
    ) -> None:
        self.dim_name = dim_name
        self.init_value = init_value
        self.grid_value = grid_value
        if message is None and dim_name is not None:
            message = (
                f"Mesh dimension mismatch for {dim_name}: "
                f"init.nc={init_value}, grid.nc={grid_value}"
            )
        super().__init__(message if message is not None else "")


class MpasDeviceError(MpasConfigError):
    """Raised when the requested execution device is unavailable.

    Example: requesting ``device='cuda'`` on a CPU-only host.
    """

    def __init__(
        self, message: str | None = None, *, requested: str | None = None
    ) -> None:
        self.requested = "" if requested is None else str(requested)
        if message is None and self.requested:
            message = (
                f"{self.requested!r} requested but the device runtime is unavailable. "
                f"{self.hint}"
            )
        super().__init__(message if message is not None else "")

    @property
    def hint(self) -> str:
        if self.requested.startswith("cuda"):
            return (
                "Install PyTorch with CUDA support, ensure a CUDA device is visible, "
                "or pass device='cpu'."
            )
        if self.requested.startswith("mps"):
            return "Enable Apple Metal support or pass device='cpu'."
        return "Use an available device such as device='cpu'."


class MpasStrictModeError(MpasConfigError):
    """Raised when strict validation rejects unsupported runtime configuration.

    Example: ``--strict-namelist`` encountering an unknown key in a user namelist.
    """


class MpasUnsupportedCapabilityError(MpasConfigError):
    """Raised when a request is outside the frozen release capability contract."""

    def __init__(
        self,
        message: str | None = None,
        *,
        capability: str,
        requested: object,
    ) -> None:
        self.capability = capability
        self.requested = requested
        super().__init__(message or f"Unsupported {capability}: {requested!r}")


class MpasNumericalAgreementError(MpasError):
    """Raised when a runtime numerical-agreement contract fails.

    Example: a history comparison exceeding its declared error contract.
    """
