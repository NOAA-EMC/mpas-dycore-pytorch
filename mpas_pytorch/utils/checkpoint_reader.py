"""
Checkpoint reader for MPAS NetCDF files.

Comprehensive utility for loading, inspecting, and validating checkpoint data
from MPAS NetCDF files. Supports arbitrary checkpoint files with
flexible data access patterns and quality validation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import netCDF4 as nc
import numpy as np
import torch

from mpas_pytorch.constants import DTYPE
from mpas_pytorch.exceptions import MpasConfigError
from mpas_pytorch.mesh import MESHState

CONNECTIVITY_INDEX_VARS = frozenset(
    {
        "cellsOnEdge",
        "edgesOnCell",
        "verticesOnCell",
        "cellsOnCell",
        "verticesOnEdge",
        "edgesOnVertex",
        "cellsOnVertex",
        "edgesOnEdge",
        "advCellsForEdge",
        "kiteForCell",
        "nEdgesOnCell",
        "nEdgesOnEdge",
        "nAdvCellsForEdge",
    }
)


@dataclass
class VariableInfo:
    """Metadata about a NetCDF variable."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    dimensions: tuple[str, ...]
    is_scalar: bool
    value_range: tuple[float, float] | None = None
    has_nan: bool = False
    has_inf: bool = False


class CheckpointReader:
    """
    Context manager for reading MPAS checkpoint NetCDF files.

    Handles Fortran column-major to PyTorch row-major conversion,
    provides data quality checks, and supports flexible data access.

    Usage:
        with CheckpointReader('checkpoint.nc') as reader:
            data = reader.load_all()
            reader.describe()
    """

    def __init__(
        self,
        filepath: str,
        transpose: bool = False,
        device: torch.device | str | None = None,
    ):
        """
        Initialize checkpoint reader.

        Args:
            filepath: Path to NetCDF checkpoint file
            transpose: If True, transpose multi-dim arrays from column-major
                      to row-major. If False, preserve Fortran memory layout.
                      Default False to maintain exact ordering.
            device: Target device for loaded tensors. Defaults to CPU.

        Raises:
            FileNotFoundError: If checkpoint file does not exist
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.filepath}")

        self.transpose = transpose
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self._dataset: nc.Dataset | None = None

    def __enter__(self) -> Self:
        """Open NetCDF dataset."""
        self._dataset = nc.Dataset(self.filepath, "r")
        return self

    def __exit__(self, *args):
        """Close NetCDF dataset."""
        if self._dataset:
            self._dataset.close()
            self._dataset = None

    def _ensure_open(self) -> nc.Dataset:
        """Ensure dataset is open (for non-context usage)."""
        if self._dataset is None:
            raise RuntimeError(
                "Dataset not open. Use CheckpointReader as context manager:\n"
                "  with CheckpointReader(filepath) as reader:\n"
                "      data = reader.load_all()"
            )
        return self._dataset

    def _numpy_to_torch(self, data: np.ndarray, name: str) -> torch.Tensor:
        """
        Convert NumPy array to PyTorch tensor with appropriate dtype.

        Args:
            data: NumPy array from NetCDF
            name: Variable name (for error reporting)

        Returns:
            PyTorch tensor with correct dtype
        """
        device = getattr(self, "device", torch.device("cpu"))

        # Determine target dtype
        if np.issubdtype(data.dtype, np.floating):
            # All floating point -> torch.float64 (DTYPE constant)
            tensor = torch.tensor(data, dtype=DTYPE, device=device)
        elif np.issubdtype(data.dtype, np.integer):
            # Integer types
            if name in CONNECTIVITY_INDEX_VARS:
                tensor = torch.tensor(data, dtype=torch.int64, device=device)
            elif data.dtype == np.int32 or data.dtype.itemsize <= 4:
                tensor = torch.tensor(data, dtype=torch.int32, device=device)
            else:
                tensor = torch.tensor(data, dtype=torch.int64, device=device)
        else:
            # Fallback: preserve as-is and let PyTorch decide
            tensor = torch.from_numpy(data).to(device)

        # Optionally transpose multi-dimensional arrays
        # Fortran: column-major (dims stored as [nCells, nVertLevels])
        # PyTorch: row-major preference
        if self.transpose and tensor.ndim > 1:
            # Transpose ALL dimensions (reverse order)
            # e.g., (nCells, nVertLevels) -> (nVertLevels, nCells)
            perm = list(range(tensor.ndim - 1, -1, -1))
            tensor = tensor.permute(*perm).contiguous()

        return tensor

    @staticmethod
    def _is_string_dtype(dtype: np.dtype) -> bool:
        """Return True for NetCDF/string-like dtypes not representable as tensors."""
        kind = np.dtype(dtype).kind
        return kind in {"S", "U", "O", "V"}

    def load_all(
        self,
        dtype: torch.dtype | None = None,
        max_total_bytes: int | None = None,
        validated_fields: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        """
        Load all variables from checkpoint as PyTorch tensors.

        Args:
            dtype: Override dtype for floating-point variables (default: torch.float64)
            max_total_bytes: Maximum total bytes to load. If checkpoint exceeds this,
                             raises MemoryError. Default: None (no limit).

        Returns:
            Dictionary mapping variable names to tensors

        Raises:
            RuntimeError: If dataset is not open
            MemoryError: If max_total_bytes is set and checkpoint exceeds it
        """
        dataset = self._ensure_open()

        # Calculate total size before loading if memory limit is set
        if max_total_bytes is not None:
            total_bytes = sum(
                int(np.dtype(dataset.variables[name].dtype).itemsize)
                * int(dataset.variables[name].size)
                for name in dataset.variables
            )
            if total_bytes > max_total_bytes:
                raise MemoryError(
                    f"Checkpoint total size ({total_bytes / 1e9:.2f} GB) exceeds "
                    f"max_total_bytes ({max_total_bytes / 1e9:.2f} GB). "
                    f"Use load_variables() to load specific variables, or increase limit."
                )

        data = {}
        for name in dataset.variables:
            var = dataset.variables[name]
            arr = var[:]
            if name in validated_fields:
                self._validate_raw_array(arr, name)

            # Skip string metadata variables (e.g., netCDF char arrays).
            if self._is_string_dtype(arr.dtype):
                continue

            # Convert to tensor
            tensor = self._numpy_to_torch(arr, name)

            # Optional dtype override for float variables
            if dtype is not None and tensor.dtype.is_floating_point:
                tensor = tensor.to(dtype)

            data[name] = tensor

        return data

    def _validate_raw_array(self, data: np.ndarray, name: str) -> None:
        """Reject missing scientific inputs before tensor conversion erases masks."""
        if np.ma.getmaskarray(data).any() or (
            np.issubdtype(data.dtype, np.number)
            and not np.isfinite(np.asarray(data)).all()
        ):
            raise MpasConfigError(
                f"{self.filepath.name}: {name} contains missing or nonfinite values"
            )

    def load_variables(
        self, var_names: list[str], dtype: torch.dtype | None = None
    ) -> dict[str, Any]:
        """
        Load specific variables by name.

        Args:
            var_names: List of variable names to load
            dtype: Override dtype for floating-point variables

        Returns:
            Dictionary of requested variables as tensors

        Raises:
            KeyError: If any variable name is not found
            RuntimeError: If dataset is not open
        """
        dataset = self._ensure_open()

        data = {}
        for name in var_names:
            if name not in dataset.variables:
                available = list(dataset.variables.keys())
                raise KeyError(
                    f"Variable '{name}' not found in checkpoint.\n"
                    f"Available variables: {available}"
                )

            var = dataset.variables[name]
            arr = var[:]
            if self._is_string_dtype(arr.dtype):
                raise TypeError(
                    f"Variable '{name}' has non-numeric dtype {arr.dtype} and "
                    "cannot be converted to torch.Tensor"
                )
            tensor = self._numpy_to_torch(arr, name)

            if dtype is not None and tensor.dtype.is_floating_point:
                tensor = tensor.to(dtype)

            data[name] = tensor

        return data

    def get_variable_info(self, name: str) -> VariableInfo:
        """
        Get metadata about a specific variable.

        Args:
            name: Variable name

        Returns:
            VariableInfo object with metadata

        Raises:
            KeyError: If variable not found
        """
        dataset = self._ensure_open()

        if name not in dataset.variables:
            raise KeyError(f"Variable '{name}' not found")

        var = dataset.variables[name]
        arr = var[:]

        is_scalar = len(var.shape) == 0

        # Compute value range and quality flags for numeric data
        value_range = None
        has_nan = False
        has_inf = False

        if (
            np.issubdtype(arr.dtype, np.floating)
            or np.issubdtype(arr.dtype, np.integer)
        ) and not is_scalar:
            value_range = (float(np.min(arr)), float(np.max(arr)))
            if np.issubdtype(arr.dtype, np.floating):
                has_nan = bool(np.any(np.isnan(arr)))
                has_inf = bool(np.any(np.isinf(arr)))

        return VariableInfo(
            name=name,
            shape=var.shape,
            dtype=arr.dtype,
            dimensions=var.dimensions,
            is_scalar=is_scalar,
            value_range=value_range,
            has_nan=has_nan,
            has_inf=has_inf,
        )

    def list_variables(self) -> list[str]:
        """
        List all variable names in checkpoint.

        Returns:
            List of variable names
        """
        dataset = self._ensure_open()
        return list(dataset.variables.keys())

    def describe(self, verbose: bool = True) -> dict[str, VariableInfo]:
        """
        Print detailed information about all variables in checkpoint.

        Args:
            verbose: If True, print to stdout; if False, only return metadata

        Returns:
            Dictionary mapping variable names to VariableInfo objects
        """
        dataset = self._ensure_open()

        var_info = {}
        for name in dataset.variables:
            var_info[name] = self.get_variable_info(name)

        if verbose:
            print("=" * 80)
            print(f"Checkpoint: {self.filepath.name}")
            print(f"Full path: {self.filepath}")
            print("=" * 80)
            print(f"\nTotal variables: {len(var_info)}")
            print("\nVariable inventory:")
            print("-" * 80)
            print(f"{'Name':<30} {'Dtype':<12} {'Shape':<20} {'Value Range'}")
            print("-" * 80)

            for name, info in var_info.items():
                dtype_str = str(info.dtype)

                if info.is_scalar:
                    shape_str = "scalar"
                    # Get scalar value
                    val = dataset.variables[name][...]
                    range_str = f"value = {val}"
                else:
                    shape_str = str(info.shape)
                    if info.value_range:
                        vmin, vmax = info.value_range
                        range_str = f"[{vmin:.6e}, {vmax:.6e}]"
                        if info.has_nan or info.has_inf:
                            flags = []
                            if info.has_nan:
                                flags.append("NaN")
                            if info.has_inf:
                                flags.append("Inf")
                            range_str += f" WARNING: {', '.join(flags)}"
                    else:
                        range_str = "N/A"

                print(f"{name:<30} {dtype_str:<12} {shape_str:<20} {range_str}")

            print("=" * 80)

        return var_info

    def validate(self, verbose: bool = True) -> bool:
        """
        Validate checkpoint data quality.

        Checks for:
        - NaN values in floating-point arrays
        - Inf values in floating-point arrays
        - Unreasonable value ranges (optional, basic checks)

        Args:
            verbose: If True, print detailed validation report

        Returns:
            True if all checks pass, False otherwise
        """
        dataset = self._ensure_open()

        issues = []

        for name in dataset.variables:
            info = self.get_variable_info(name)

            # Check for NaN/Inf in floating-point data
            if info.has_nan:
                issues.append(f"Variable '{name}' contains NaN values")

            if info.has_inf:
                issues.append(f"Variable '{name}' contains Inf values")

            # Basic sanity checks on value ranges
            if info.value_range:
                vmin, vmax = info.value_range

                # Check for suspicious patterns
                if vmin == vmax and not info.is_scalar and info.shape[0] > 1:
                    # Constant array (might be intentional, but worth noting)
                    issues.append(
                        f"Variable '{name}' is constant (all values = {vmin:.6e})"
                    )

        if verbose:
            print("=" * 80)
            print("Checkpoint Validation Report")
            print("=" * 80)
            print(f"File: {self.filepath.name}\n")

            if issues:
                print(f"FAILED: {len(issues)} issue(s) found:\n")
                for i, issue in enumerate(issues, 1):
                    print(f"  {i}. {issue}")
                print("\n" + "=" * 80)
            else:
                print("PASSED: No data quality issues detected")
                print("=" * 80)

        return len(issues) == 0


# Standalone functions for common operations


def load_checkpoint(
    filepath: str,
    transpose: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    max_total_bytes: int | None = None,
    validated_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """
    Load all variables from a checkpoint NetCDF file as torch tensors.

    Convenience function for one-shot loading without context manager.

    Args:
        filepath: Path to NetCDF checkpoint file
        transpose: If True, transpose multi-dim arrays from column-major to row-major
        device: Target device for loaded tensors. Defaults to CPU.
        dtype: Override dtype for floating-point variables (default: torch.float64)
        max_total_bytes: Maximum total bytes to load. If checkpoint exceeds this,
                         raises MemoryError. Default: None (no limit).

    Returns:
        Dictionary mapping variable names to PyTorch tensors

    Raises:
        MemoryError: If max_total_bytes is set and checkpoint exceeds it

    Example:
        data = load_checkpoint('checkpoint.nc')
        nCells = data['nCells'].item()
        zz = data['zz']  # Shape: (nCells, nVertLevels) or transposed
    """
    with CheckpointReader(filepath, transpose=transpose, device=device) as reader:
        return cast(
            dict[str, Any],
            reader.load_all(
                dtype=dtype,
                max_total_bytes=max_total_bytes,
                validated_fields=validated_fields,
            ),
        )


def describe_checkpoint(filepath: str) -> None:
    """
    Print variable inventory for a checkpoint file.

    Args:
        filepath: Path to NetCDF checkpoint file

    Example:
        describe_checkpoint('model_input.nc')
    """
    with CheckpointReader(filepath) as reader:
        reader.describe(verbose=True)


def validate_checkpoint(filepath: str, verbose: bool = True) -> bool:
    """
    Validate data quality in a checkpoint file.

    Args:
        filepath: Path to NetCDF checkpoint file
        verbose: If True, print detailed validation report

    Returns:
        True if validation passes, False otherwise

    Example:
        if not validate_checkpoint('checkpoint.nc'):
            raise RuntimeError("Checkpoint validation failed")
    """
    with CheckpointReader(filepath) as reader:
        return bool(reader.validate(verbose=verbose))


# Mesh loading functions


def _convert_fortran_index_array(
    arr: torch.Tensor,
    n_slice: int,
    max_valid_1based: int,
    name: str,
    validate: bool = True,
) -> torch.Tensor:
    """
    Convert Fortran 1-based connectivity array to 0-based indexing.

    Fortran checkpoint arrays have:
    - Shape (N+1, ...) where index 0 through N-1 is valid data, index N is padding
    - Values are 1-indexed (1 to max_valid_1based)
    - Sentinel values (0 or max_valid_1based+1) indicate "no neighbor"

    PyTorch arrays should have:
    - Shape (N, ...) with valid data only
    - Values are 0-indexed (0 to max_valid_1based-1)
    - -1 indicates "no neighbor"

    Args:
        arr: Fortran array with shape (N+1, ...)
        n_slice: Number of entries to keep (N), removes padding row
        max_valid_1based: Maximum valid 1-based index in the data
            (e.g., nCells for cell indices, nEdges for edge indices)
        name: Variable name for error messages
        validate: If True (default), validate that converted indices are within
            valid bounds [0, max_valid_1based-1]. Set to False to skip validation
            (not recommended for production use).

    Returns:
        PyTorch array with shape (N, ...) and 0-based indexing

    Raises:
        ValueError: If validate=True and indices are out of valid bounds.
    """
    # Slice to remove padding row at the end
    sliced = arr[:n_slice]

    # Identify sentinel values:
    # - 0 means "no neighbor" (becomes -1)
    # - max_valid_1based + 1 means "no neighbor" (also becomes -1)
    sentinel_value = max_valid_1based + 1

    # Create mask for sentinel values
    is_sentinel = (sliced == 0) | (sliced == sentinel_value)

    # Convert 1-based to 0-based indexing
    # Valid values: 1 -> 0, 2 -> 1, ..., max_valid_1based -> max_valid_1based-1
    converted = sliced - 1

    # Set sentinel positions to -1
    converted = torch.where(
        is_sentinel,
        torch.tensor(-1, dtype=converted.dtype, device=converted.device),
        converted,
    )

    # Validate index bounds (after conversion)
    if validate:
        non_sentinel = converted[converted >= 0]
        if len(non_sentinel) > 0:
            min_idx = non_sentinel.min().item()
            max_idx = non_sentinel.max().item()
            if min_idx < 0:
                raise ValueError(
                    f"{name}: invalid negative index {min_idx} found (non-sentinel)"
                )
            if max_idx >= max_valid_1based:
                raise ValueError(
                    f"{name}: index {max_idx} out of bounds [0, {max_valid_1based - 1}]. "
                    f"This may indicate corrupt checkpoint data or wrong n_slice value."
                )

    return converted


def _slice_padded_array(arr: torch.Tensor, n_valid: int, name: str) -> torch.Tensor:
    """
    Slice Fortran array to remove padding row.

    Fortran checkpoint arrays have shape (N+1, ...) where the last row is padding.
    This function slices to (N, ...).

    Args:
        arr: Fortran array with shape (N+1, ...)
        n_valid: Number of valid entries (N)
        name: Variable name for error messages

    Returns:
        Array with shape (N, ...) containing valid data only
    """
    return arr[:n_valid]


def load_mesh_from_checkpoints(
    dim_path: str,
    conn_path: str,
    geom_path: str,
) -> MESHState:
    """
    Load a complete mesh from MPAS NetCDF component files.

    This function loads mesh dimensions, connectivity arrays, and geometry/metric
    arrays from three separate MPAS NetCDF files.

    IMPORTANT - RETURNS 0-BASED INDEXED ARRAYS:
        This function returns a MESHState with 0-based indexed connectivity arrays
        suitable for Pythonic array access. However, dynamics kernels (solve_diagnostics,
        acoustic_step, recover_large_step, etc.) expect 1-based Fortran-style arrays
        with +1 padding.

        For dynamics kernels, use load_checkpoint() directly which preserves the
        original Fortran array format, or apply manual conversion.

        See MESHState docstring for detailed indexing convention documentation.

    Index Conversion:
        Fortran uses 1-based indexing with 0 for "no neighbor".
        PyTorch uses 0-based indexing with -1 for "no neighbor".
        All connectivity arrays are converted appropriately.

    Array Slicing:
        Fortran checkpoint arrays have shape (N+1, ...) where the last entry
        is zero-padding. Arrays are sliced to (N, ...) containing valid data only.

    Args:
        dim_path: Path to mesh_dimensions.nc (contains nCells, nEdges, etc.)
        conn_path: Path to mesh_connectivity.nc (contains cellsOnEdge, etc.)
        geom_path: Path to mesh_geometry.nc (contains areaCell, dvEdge, etc.)

    Returns:
        MESHState with all connectivity and geometry arrays populated.
        Connectivity arrays use 0-based indexing with -1 for missing neighbors.
        Arrays have shape (N, ...) without padding rows.
        All floating-point arrays use torch.float64.

    Raises:
        FileNotFoundError: If any checkpoint file does not exist
        KeyError: If required variables are missing from checkpoint files

    Example:
        mesh = load_mesh_from_checkpoints(
            dim_path='mesh_dimensions.nc',
            conn_path='mesh_connectivity.nc',
            geom_path='mesh_geometry.nc',
        )
        mesh.validate()  # Verify shapes and index ranges
    """
    # Load dimensions
    dims = load_checkpoint(dim_path)
    nCells = int(dims["nCells"].item())
    nEdges = int(dims["nEdges"].item())
    nVertices = int(dims["nVertices"].item())
    maxEdges = int(dims["maxEdges"].item())
    vertexDegree = int(dims["vertexDegree"].item())
    nVertLevels = int(dims["nVertLevels"].item())
    maxEdges2 = int(dims["maxEdges2"].item())

    # Load connectivity arrays (need index conversion)
    conn = load_checkpoint(conn_path)

    # Convert connectivity arrays from Fortran 1-based to PyTorch 0-based
    # Arrays have shape (N+1, ...) in checkpoint, need slicing and index conversion
    # Arguments: (array, n_slice, max_valid_1based, name)
    # - n_slice: number of rows to keep (removes padding row)
    # - max_valid_1based: max valid index value (for detecting sentinels)

    # cellsOnEdge: indexed by edges, contains cell indices
    cellsOnEdge = _convert_fortran_index_array(
        conn["cellsOnEdge"], nEdges, nCells, "cellsOnEdge"
    )
    # edgesOnCell: indexed by cells, contains edge indices
    edgesOnCell = _convert_fortran_index_array(
        conn["edgesOnCell"], nCells, nEdges, "edgesOnCell"
    )
    # cellsOnCell: indexed by cells, contains cell indices
    cellsOnCell = _convert_fortran_index_array(
        conn["cellsOnCell"], nCells, nCells, "cellsOnCell"
    )
    # verticesOnCell: indexed by cells, contains vertex indices
    verticesOnCell = _convert_fortran_index_array(
        conn["verticesOnCell"], nCells, nVertices, "verticesOnCell"
    )
    # verticesOnEdge: indexed by edges, contains vertex indices
    verticesOnEdge = _convert_fortran_index_array(
        conn["verticesOnEdge"], nEdges, nVertices, "verticesOnEdge"
    )
    # edgesOnVertex: indexed by vertices, contains edge indices
    edgesOnVertex = _convert_fortran_index_array(
        conn["edgesOnVertex"], nVertices, nEdges, "edgesOnVertex"
    )
    # cellsOnVertex: indexed by vertices, contains cell indices
    cellsOnVertex = _convert_fortran_index_array(
        conn["cellsOnVertex"], nVertices, nCells, "cellsOnVertex"
    )

    # nEdgesOnCell doesn't need index conversion (it's a count, not an index)
    # But still needs slicing to remove padding
    nEdgesOnCell = _slice_padded_array(conn["nEdgesOnCell"], nCells, "nEdgesOnCell")

    # Load geometry/metric arrays (need slicing but no index conversion)
    geom = load_checkpoint(geom_path)

    dvEdge = _slice_padded_array(geom["dvEdge"], nEdges, "dvEdge")
    dcEdge = _slice_padded_array(geom["dcEdge"], nEdges, "dcEdge")
    areaCell = _slice_padded_array(geom["areaCell"], nCells, "areaCell")
    areaTriangle = _slice_padded_array(geom["areaTriangle"], nVertices, "areaTriangle")
    fEdge = _slice_padded_array(geom["fEdge"], nEdges, "fEdge")
    fVertex = _slice_padded_array(geom["fVertex"], nVertices, "fVertex")
    invAreaCell = _slice_padded_array(geom["invAreaCell"], nCells, "invAreaCell")
    weightsOnEdge = _slice_padded_array(geom["weightsOnEdge"], nEdges, "weightsOnEdge")
    kiteAreasOnVertex = _slice_padded_array(
        geom["kiteAreasOnVertex"], nVertices, "kiteAreasOnVertex"
    )

    # Construct and return MESHState
    return MESHState(
        # Dimensions
        nCells=nCells,
        nVertices=nVertices,
        nEdges=nEdges,
        maxEdges=maxEdges,
        vertexDegree=vertexDegree,
        nVertLevels=nVertLevels,
        maxEdges2=maxEdges2,
        # Connectivity (0-indexed, -1 for missing)
        cellsOnCell=cellsOnCell,
        edgesOnCell=edgesOnCell,
        verticesOnCell=verticesOnCell,
        cellsOnEdge=cellsOnEdge,
        verticesOnEdge=verticesOnEdge,
        edgesOnVertex=edgesOnVertex,
        cellsOnVertex=cellsOnVertex,
        nEdgesOnCell=nEdgesOnCell,
        # Metric arrays
        dvEdge=dvEdge,
        dcEdge=dcEdge,
        areaCell=areaCell,
        areaTriangle=areaTriangle,
        # Physics arrays
        fEdge=fEdge,
        fVertex=fVertex,
        # Additional metric arrays
        weightsOnEdge=weightsOnEdge,
        kiteAreasOnVertex=kiteAreasOnVertex,
        invAreaCell=invAreaCell,
    )


def load_mesh_for_kernels(
    dim_path: str,
    conn_path: str,
    geom_path: str,
) -> dict[str, Any]:
    """
    Load mesh checkpoint data in Fortran 1-based+padded format for kernels.

    This is a lightweight helper that merges the raw checkpoint arrays from:
    - mesh_dimensions.nc
    - mesh_connectivity.nc
    - mesh_geometry.nc

    Unlike load_mesh_from_checkpoints(), this function does NOT convert indices
    to 0-based. Use it when passing mesh data directly into dynamics kernels
    or time integration routines that expect 1-based+padded arrays.

    Args:
        dim_path: Path to mesh_dimensions.nc
        conn_path: Path to mesh_connectivity.nc
        geom_path: Path to mesh_geometry.nc

    Returns:
        Dictionary of raw checkpoint tensors (1-based+padded indexing).
    """
    dims = load_checkpoint(dim_path)
    conn = load_checkpoint(conn_path)
    geom = load_checkpoint(geom_path)

    mesh = {}
    mesh.update(dims)
    mesh.update(conn)
    mesh.update(geom)
    return mesh


def load_mesh(filepath: str) -> MESHState:
    """
    Load MPAS mesh from NetCDF file.

    NOTE: This function is not implemented. Use load_mesh_from_checkpoints() instead.

    Raises:
        NotImplementedError: Always - use load_mesh_from_checkpoints() instead.
    """
    raise NotImplementedError(
        "load_mesh() is not implemented.\n"
        "Use load_mesh_from_checkpoints() for MPAS NetCDF component files:\n\n"
        "    from mpas_pytorch.utils import load_mesh_from_checkpoints\n"
        "    mesh = load_mesh_from_checkpoints(\n"
        "        dim_path='mesh_dimensions.nc',\n"
        "        conn_path='mesh_connectivity.nc',\n"
        "        geom_path='mesh_geometry.nc',\n"
        "    )"
    )


def load_state(filepath: str, mesh: MESHState) -> dict[str, Any]:
    """
    Load state variables from NetCDF checkpoint.

    NOTE: This function is not implemented.
    Use load_checkpoint() + DycoreState for state loading.

    Raises:
        NotImplementedError: Always - use alternative approach.
    """
    raise NotImplementedError(
        "load_state() is not yet implemented.\n"
        "Use load_checkpoint() for general checkpoint access:\n\n"
        "    from mpas_pytorch.utils import load_checkpoint\n"
        "    data = load_checkpoint('checkpoint.nc')\n\n"
        "Or use load_state_from_checkpoint() for DycoreState:\n\n"
        "    from mpas_pytorch.state import load_state_from_checkpoint\n"
        "    state = load_state_from_checkpoint('checkpoint.nc')"
    )


def save_checkpoint(
    filepath: str,
    state: dict[str, Any],
    mesh: dict[str, Any] | None = None,
) -> None:
    """
    Save state tensors to NetCDF checkpoint file.

    Converts PyTorch tensors to numpy arrays and writes to NetCDF format
    using xarray for compatibility with MPAS NetCDF files.

    Args:
        filepath: Path to output NetCDF file
        state: Dictionary of tensors to save. Keys become variable names.
        mesh: Optional dictionary of mesh tensors to include.

    Dimension Naming Convention:
        Dimensions are automatically named based on array shapes:
        - Scalars: saved as 0-dimensional variables
        - 1D arrays: dim_0, dim_1, ... (based on size)
        - 2D+ arrays: dim_0_size, dim_1_size, ... (dimensions named by position and size)

        Common MPAS dimensions are detected by shape:
        - Size matching nCells+1: may be 'nCells_p1' dimension
        - Size matching nVertLevels: may be 'nVertLevels' dimension
        - Size matching nVertLevels+1: may be 'nVertLevels_p1' dimension

    Example:
        state = {
            'u': torch.randn(100, 55, dtype=torch.float64),
            'nCells': torch.tensor(99, dtype=torch.int32),
        }
        save_checkpoint('output.nc', state)

    Raises:
        ImportError: If xarray is not available
        IOError: If file cannot be written
    """
    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError(
            "xarray is required for save_checkpoint. Install with: pip install xarray"
        ) from e

    # Collect all tensors to save
    all_data = {}
    all_data.update(state)
    if mesh is not None:
        all_data.update(mesh)

    # Build dimension registry to ensure consistent dimension names across variables
    # Map shape -> dimension_name for each unique dimension size
    dim_registry: dict[int, str] = {}

    # First pass: identify all unique dimension sizes
    for name, tensor in all_data.items():
        if isinstance(tensor, torch.Tensor):
            for i, size in enumerate(tensor.shape):
                if size not in dim_registry:
                    dim_registry[size] = f"dim_{size}"

    # Build xarray data variables
    data_vars = {}
    for name, tensor in all_data.items():
        if isinstance(tensor, torch.Tensor):
            # Convert tensor to numpy
            arr = tensor.detach().cpu().numpy()

            if arr.ndim == 0:
                # Scalar: no dimensions
                data_vars[name] = xr.Variable((), arr)
            else:
                # Generate dimension names based on shape
                dims = []
                for i, size in enumerate(arr.shape):
                    # Use indexed dimension name to allow same-size dims in different positions
                    dim_name = f"{name}_dim{i}"
                    dims.append(dim_name)
                data_vars[name] = xr.Variable(tuple(dims), arr)
        else:
            # Handle non-tensor types (scalars)
            try:
                arr = np.array(tensor)
                if arr.ndim == 0:
                    data_vars[name] = xr.Variable((), arr)
                else:
                    dims = [f"{name}_dim{i}" for i in range(arr.ndim)]
                    data_vars[name] = xr.Variable(tuple(dims), arr)
            except (TypeError, ValueError):
                # Skip non-convertible types
                pass

    # Create dataset and write to NetCDF
    ds = xr.Dataset(data_vars)
    ds.to_netcdf(filepath)
