"""
Mesh data structures for MPAS-Atmosphere.

This module defines the mesh representation including cells, vertices, edges,
and all connectivity arrays required for the dynamics core.

The mesh is loaded from NetCDF checkpoint files or grid files.
All connectivity and geometric arrays are stored as torch tensors in double precision.
"""

from dataclasses import dataclass

import torch


def _check(condition: bool, message: str) -> None:
    """
    Validate a condition and raise ValueError if false.

    Unlike assert, this check is NOT disabled with python -O.
    """
    if not condition:
        raise ValueError(message)


@dataclass
class MESHState:
    """
    MPAS mesh state containing all connectivity and geometric information.

    INDEXING CONVENTION:
        All connectivity arrays in MESHState use 0-based indexing:
        - Valid indices range from 0 to N-1 (where N is nCells, nEdges, etc.)
        - Missing neighbors are indicated by -1 (upstream MPAS uses 0)
        - Arrays have shape (N,) or (N, M) without padding rows

        This convention is set by load_mesh_from_checkpoints() which converts
        from Fortran 1-based indexing during loading.

    WARNING - KERNEL COMPATIBILITY:
        Dynamics kernels (solve_diagnostics, acoustic_step, recover_large_step, etc.)
        expect 1-based MPAS arrays with a padding row:
        - Arrays have shape (N+1, ...) where index 0 is padding
        - Valid indices range from 1 to N (code subtracts 1 to convert)
        - Sentinel value 0 indicates "no neighbor"

        If MESHState arrays are passed directly to kernel functions without
        conversion, indices will be off by 1, causing silent physics corruption.

        For kernel use, either:
        1. Load state via load_state_from_checkpoint(), which preserves MPAS layout
        2. Use kernel-specific checkpoint loading that maintains 1-based indexing
        3. Apply manual conversion before passing to kernels

    Attributes:
        Dimensions:
            nCells: Number of cells
            nVertices: Number of vertices
            nEdges: Number of edges
            maxEdges: Maximum number of edges per cell
            vertexDegree: Maximum number of cells per vertex (typically 3)
            nVertLevels: Number of vertical levels (optional)
            maxEdges2: 2 * maxEdges (optional, used for weightsOnEdge)

        Connectivity arrays (0-indexed, -1 indicates missing neighbor):
            cellsOnCell: Neighbor cells for each cell, shape (nCells, maxEdges)
            edgesOnCell: Edges bounding each cell, shape (nCells, maxEdges)
            verticesOnCell: Vertices of each cell, shape (nCells, maxEdges)
            cellsOnEdge: Cells adjacent to each edge, shape (nEdges, 2)
            verticesOnEdge: Vertices of each edge, shape (nEdges, 2)
            edgesOnVertex: Edges meeting at each vertex, shape (nVertices, vertexDegree)
            cellsOnVertex: Cells meeting at each vertex, shape (nVertices, vertexDegree)
            nEdgesOnCell: Number of edges on each cell, shape (nCells,)

        Metric/geometric arrays (always present):
            dvEdge: Distance between cell centers across edge, shape (nEdges,)
            dcEdge: Distance between vertices of edge, shape (nEdges,)
            areaCell: Cell area, shape (nCells,)
            areaTriangle: Dual cell (triangle) area, shape (nVertices,)

        Physics arrays (always present):
            fEdge: Coriolis parameter at edge midpoints, shape (nEdges,)
            fVertex: Coriolis parameter at vertices, shape (nVertices,)

        Additional metric arrays:
            weightsOnEdge: Reconstruction weights, shape (nEdges, maxEdges2)
            kiteAreasOnVertex: Kite areas at each vertex, shape (nVertices, vertexDegree)
            invAreaCell: Inverse of cell area, shape (nCells,)

        Coordinate arrays (optional, may not be in all checkpoints):
            latCell, lonCell: Cell center coordinates (radians)
            latEdge, lonEdge: Edge midpoint coordinates (radians)
            latVertex, lonVertex: Vertex coordinates (radians)
            xCell, yCell, zCell: Cell center Cartesian coordinates (m)
            xEdge, yEdge, zEdge: Edge midpoint Cartesian coordinates (m)
            xVertex, yVertex, zVertex: Vertex Cartesian coordinates (m)
            angleEdge: Angle of edge relative to east direction
    """

    # Dimensions (required)
    nCells: int
    nVertices: int
    nEdges: int
    maxEdges: int
    vertexDegree: int

    # Connectivity arrays (required, 0-indexed with -1 for missing)
    cellsOnCell: torch.Tensor
    edgesOnCell: torch.Tensor
    verticesOnCell: torch.Tensor
    cellsOnEdge: torch.Tensor
    verticesOnEdge: torch.Tensor
    edgesOnVertex: torch.Tensor
    cellsOnVertex: torch.Tensor
    nEdgesOnCell: torch.Tensor

    # Metric arrays (required)
    dvEdge: torch.Tensor
    dcEdge: torch.Tensor
    areaCell: torch.Tensor
    areaTriangle: torch.Tensor

    # Physics arrays (required)
    fEdge: torch.Tensor
    fVertex: torch.Tensor

    # Additional metric arrays (required)
    weightsOnEdge: torch.Tensor
    kiteAreasOnVertex: torch.Tensor
    invAreaCell: torch.Tensor

    # Optional dimensions
    nVertLevels: int | None = None
    maxEdges2: int | None = None

    # Coordinate arrays - optional (not in all checkpoints)
    latCell: torch.Tensor | None = None
    lonCell: torch.Tensor | None = None
    latEdge: torch.Tensor | None = None
    lonEdge: torch.Tensor | None = None
    latVertex: torch.Tensor | None = None
    lonVertex: torch.Tensor | None = None
    xCell: torch.Tensor | None = None
    yCell: torch.Tensor | None = None
    zCell: torch.Tensor | None = None
    xEdge: torch.Tensor | None = None
    yEdge: torch.Tensor | None = None
    zEdge: torch.Tensor | None = None
    xVertex: torch.Tensor | None = None
    yVertex: torch.Tensor | None = None
    zVertex: torch.Tensor | None = None
    angleEdge: torch.Tensor | None = None

    def _move_optional(
        self, tensor: torch.Tensor | None, device: str
    ) -> torch.Tensor | None:
        """Move optional tensor to device if it exists."""
        return tensor.to(device) if tensor is not None else None

    def to(self, device: str) -> "MESHState":
        """Move all tensors to specified device."""
        return MESHState(
            # Dimensions
            nCells=self.nCells,
            nVertices=self.nVertices,
            nEdges=self.nEdges,
            maxEdges=self.maxEdges,
            vertexDegree=self.vertexDegree,
            nVertLevels=self.nVertLevels,
            maxEdges2=self.maxEdges2,
            # Connectivity
            cellsOnCell=self.cellsOnCell.to(device),
            edgesOnCell=self.edgesOnCell.to(device),
            verticesOnCell=self.verticesOnCell.to(device),
            cellsOnEdge=self.cellsOnEdge.to(device),
            verticesOnEdge=self.verticesOnEdge.to(device),
            edgesOnVertex=self.edgesOnVertex.to(device),
            cellsOnVertex=self.cellsOnVertex.to(device),
            nEdgesOnCell=self.nEdgesOnCell.to(device),
            # Metric arrays
            dvEdge=self.dvEdge.to(device),
            dcEdge=self.dcEdge.to(device),
            areaCell=self.areaCell.to(device),
            areaTriangle=self.areaTriangle.to(device),
            # Physics
            fEdge=self.fEdge.to(device),
            fVertex=self.fVertex.to(device),
            # Additional metrics
            weightsOnEdge=self.weightsOnEdge.to(device),
            kiteAreasOnVertex=self.kiteAreasOnVertex.to(device),
            invAreaCell=self.invAreaCell.to(device),
            # Optional coordinates
            latCell=self._move_optional(self.latCell, device),
            lonCell=self._move_optional(self.lonCell, device),
            latEdge=self._move_optional(self.latEdge, device),
            lonEdge=self._move_optional(self.lonEdge, device),
            latVertex=self._move_optional(self.latVertex, device),
            lonVertex=self._move_optional(self.lonVertex, device),
            xCell=self._move_optional(self.xCell, device),
            yCell=self._move_optional(self.yCell, device),
            zCell=self._move_optional(self.zCell, device),
            xEdge=self._move_optional(self.xEdge, device),
            yEdge=self._move_optional(self.yEdge, device),
            zEdge=self._move_optional(self.zEdge, device),
            xVertex=self._move_optional(self.xVertex, device),
            yVertex=self._move_optional(self.yVertex, device),
            zVertex=self._move_optional(self.zVertex, device),
            angleEdge=self._move_optional(self.angleEdge, device),
        )

    def validate(self) -> bool:
        """
        Validate mesh connectivity and dimensions.

        Returns:
            True if mesh is valid, raises ValueError otherwise.
        """

        def _check_integer_tensor(tensor: torch.Tensor, name: str) -> None:
            if (
                tensor.dtype == torch.bool
                or tensor.is_floating_point()
                or tensor.is_complex()
            ):
                raise ValueError(
                    f"{name} must be an integer tensor, got {tensor.dtype}"
                )

        # Connectivity array shapes
        _check(
            self.cellsOnCell.shape == (self.nCells, self.maxEdges),
            f"cellsOnCell shape {self.cellsOnCell.shape} != ({self.nCells}, {self.maxEdges})",
        )
        _check(
            self.edgesOnCell.shape == (self.nCells, self.maxEdges),
            f"edgesOnCell shape {self.edgesOnCell.shape} != ({self.nCells}, {self.maxEdges})",
        )
        _check(
            self.verticesOnCell.shape == (self.nCells, self.maxEdges),
            f"verticesOnCell shape {self.verticesOnCell.shape} != ({self.nCells}, {self.maxEdges})",
        )
        _check(
            self.cellsOnEdge.shape == (self.nEdges, 2),
            f"cellsOnEdge shape {self.cellsOnEdge.shape} != ({self.nEdges}, 2)",
        )
        _check(
            self.verticesOnEdge.shape == (self.nEdges, 2),
            f"verticesOnEdge shape {self.verticesOnEdge.shape} != ({self.nEdges}, 2)",
        )
        _check(
            self.edgesOnVertex.shape == (self.nVertices, self.vertexDegree),
            f"edgesOnVertex shape {self.edgesOnVertex.shape} != ({self.nVertices}, {self.vertexDegree})",
        )
        _check(
            self.cellsOnVertex.shape == (self.nVertices, self.vertexDegree),
            f"cellsOnVertex shape {self.cellsOnVertex.shape} != ({self.nVertices}, {self.vertexDegree})",
        )
        _check(
            self.nEdgesOnCell.shape == (self.nCells,),
            f"nEdgesOnCell shape {self.nEdgesOnCell.shape} != ({self.nCells},)",
        )

        # Connectivity array dtypes (must be integer)
        _check_integer_tensor(self.cellsOnCell, "cellsOnCell")
        _check_integer_tensor(self.edgesOnCell, "edgesOnCell")
        _check_integer_tensor(self.verticesOnCell, "verticesOnCell")
        _check_integer_tensor(self.cellsOnEdge, "cellsOnEdge")
        _check_integer_tensor(self.verticesOnEdge, "verticesOnEdge")
        _check_integer_tensor(self.edgesOnVertex, "edgesOnVertex")
        _check_integer_tensor(self.cellsOnVertex, "cellsOnVertex")
        _check_integer_tensor(self.nEdgesOnCell, "nEdgesOnCell")

        # Metric array shapes
        _check(
            self.dvEdge.shape == (self.nEdges,),
            f"dvEdge shape {self.dvEdge.shape} != ({self.nEdges},)",
        )
        _check(
            self.dcEdge.shape == (self.nEdges,),
            f"dcEdge shape {self.dcEdge.shape} != ({self.nEdges},)",
        )
        _check(
            self.areaCell.shape == (self.nCells,),
            f"areaCell shape {self.areaCell.shape} != ({self.nCells},)",
        )
        _check(
            self.areaTriangle.shape == (self.nVertices,),
            f"areaTriangle shape {self.areaTriangle.shape} != ({self.nVertices},)",
        )

        # Physics array shapes
        _check(
            self.fEdge.shape == (self.nEdges,),
            f"fEdge shape {self.fEdge.shape} != ({self.nEdges},)",
        )
        _check(
            self.fVertex.shape == (self.nVertices,),
            f"fVertex shape {self.fVertex.shape} != ({self.nVertices},)",
        )

        # Additional metric array shapes
        expected_maxEdges2 = (
            self.maxEdges2 if self.maxEdges2 is not None else 2 * self.maxEdges
        )
        _check(
            self.weightsOnEdge.shape == (self.nEdges, expected_maxEdges2),
            f"weightsOnEdge shape {self.weightsOnEdge.shape} != ({self.nEdges}, {expected_maxEdges2})",
        )
        _check(
            self.kiteAreasOnVertex.shape == (self.nVertices, self.vertexDegree),
            f"kiteAreasOnVertex shape {self.kiteAreasOnVertex.shape} != ({self.nVertices}, {self.vertexDegree})",
        )
        _check(
            self.invAreaCell.shape == (self.nCells,),
            f"invAreaCell shape {self.invAreaCell.shape} != ({self.nCells},)",
        )

        # Validate optional coordinate arrays if present
        if self.latCell is not None:
            _check(
                self.latCell.shape == (self.nCells,),
                f"latCell shape {self.latCell.shape} != ({self.nCells},)",
            )
        if self.lonCell is not None:
            _check(
                self.lonCell.shape == (self.nCells,),
                f"lonCell shape {self.lonCell.shape} != ({self.nCells},)",
            )
        if self.latEdge is not None:
            _check(
                self.latEdge.shape == (self.nEdges,),
                f"latEdge shape {self.latEdge.shape} != ({self.nEdges},)",
            )
        if self.lonEdge is not None:
            _check(
                self.lonEdge.shape == (self.nEdges,),
                f"lonEdge shape {self.lonEdge.shape} != ({self.nEdges},)",
            )
        if self.latVertex is not None:
            _check(
                self.latVertex.shape == (self.nVertices,),
                f"latVertex shape {self.latVertex.shape} != ({self.nVertices},)",
            )
        if self.lonVertex is not None:
            _check(
                self.lonVertex.shape == (self.nVertices,),
                f"lonVertex shape {self.lonVertex.shape} != ({self.nVertices},)",
            )
        if self.xCell is not None:
            _check(
                self.xCell.shape == (self.nCells,),
                f"xCell shape {self.xCell.shape} != ({self.nCells},)",
            )
        if self.yCell is not None:
            _check(
                self.yCell.shape == (self.nCells,),
                f"yCell shape {self.yCell.shape} != ({self.nCells},)",
            )
        if self.zCell is not None:
            _check(
                self.zCell.shape == (self.nCells,),
                f"zCell shape {self.zCell.shape} != ({self.nCells},)",
            )
        if self.xEdge is not None:
            _check(
                self.xEdge.shape == (self.nEdges,),
                f"xEdge shape {self.xEdge.shape} != ({self.nEdges},)",
            )
        if self.yEdge is not None:
            _check(
                self.yEdge.shape == (self.nEdges,),
                f"yEdge shape {self.yEdge.shape} != ({self.nEdges},)",
            )
        if self.zEdge is not None:
            _check(
                self.zEdge.shape == (self.nEdges,),
                f"zEdge shape {self.zEdge.shape} != ({self.nEdges},)",
            )
        if self.xVertex is not None:
            _check(
                self.xVertex.shape == (self.nVertices,),
                f"xVertex shape {self.xVertex.shape} != ({self.nVertices},)",
            )
        if self.yVertex is not None:
            _check(
                self.yVertex.shape == (self.nVertices,),
                f"yVertex shape {self.yVertex.shape} != ({self.nVertices},)",
            )
        if self.zVertex is not None:
            _check(
                self.zVertex.shape == (self.nVertices,),
                f"zVertex shape {self.zVertex.shape} != ({self.nVertices},)",
            )
        if self.angleEdge is not None:
            _check(
                self.angleEdge.shape == (self.nEdges,),
                f"angleEdge shape {self.angleEdge.shape} != ({self.nEdges},)",
            )

        # =========================================================================
        # Check dtype: Float arrays must be torch.float64 (machine precision)
        # =========================================================================
        REQUIRED_DTYPE = torch.float64

        # Required float arrays (must always be float64)
        required_float_arrays: list[tuple[str, torch.Tensor]] = [
            ("dvEdge", self.dvEdge),
            ("dcEdge", self.dcEdge),
            ("areaCell", self.areaCell),
            ("areaTriangle", self.areaTriangle),
            ("fEdge", self.fEdge),
            ("fVertex", self.fVertex),
            ("weightsOnEdge", self.weightsOnEdge),
            ("kiteAreasOnVertex", self.kiteAreasOnVertex),
            ("invAreaCell", self.invAreaCell),
        ]

        for name, tensor in required_float_arrays:
            _check(
                tensor.dtype == REQUIRED_DTYPE,
                f"{name} must be {REQUIRED_DTYPE}, got {tensor.dtype} "
                "(machine precision requires float64)",
            )

        # Optional float arrays (check only if not None)
        optional_float_arrays: list[tuple[str, torch.Tensor | None]] = [
            ("latCell", self.latCell),
            ("lonCell", self.lonCell),
            ("latEdge", self.latEdge),
            ("lonEdge", self.lonEdge),
            ("latVertex", self.latVertex),
            ("lonVertex", self.lonVertex),
            ("xCell", self.xCell),
            ("yCell", self.yCell),
            ("zCell", self.zCell),
            ("xEdge", self.xEdge),
            ("yEdge", self.yEdge),
            ("zEdge", self.zEdge),
            ("xVertex", self.xVertex),
            ("yVertex", self.yVertex),
            ("zVertex", self.zVertex),
            ("angleEdge", self.angleEdge),
        ]

        for name, optional_tensor in optional_float_arrays:
            if optional_tensor is not None:
                _check(
                    optional_tensor.dtype == REQUIRED_DTYPE,
                    f"{name} must be {REQUIRED_DTYPE}, got {optional_tensor.dtype} "
                    "(machine precision requires float64)",
                )

        # Validate index ranges for connectivity arrays
        # All should be -1 (missing) or in valid range [0, N-1]
        self._validate_connectivity_indices(
            self.cellsOnCell, self.nCells, "cellsOnCell"
        )
        self._validate_connectivity_indices(
            self.edgesOnCell, self.nEdges, "edgesOnCell"
        )
        self._validate_connectivity_indices(
            self.verticesOnCell, self.nVertices, "verticesOnCell"
        )
        self._validate_connectivity_indices(
            self.cellsOnEdge, self.nCells, "cellsOnEdge"
        )
        self._validate_connectivity_indices(
            self.verticesOnEdge, self.nVertices, "verticesOnEdge"
        )
        self._validate_connectivity_indices(
            self.edgesOnVertex, self.nEdges, "edgesOnVertex"
        )
        self._validate_connectivity_indices(
            self.cellsOnVertex, self.nCells, "cellsOnVertex"
        )

        # Validate per-cell sentinel consistency
        # Ensures valid slots have valid indices and padding slots have sentinels
        self._validate_per_cell_sentinels()

        return True

    def _validate_connectivity_indices(
        self, arr: torch.Tensor, max_valid: int, name: str
    ) -> None:
        """Validate that connectivity array has indices in valid range."""
        min_val = arr.min().item()
        max_val = arr.max().item()
        _check(min_val >= -1, f"{name} has invalid index {min_val} < -1")
        _check(
            max_val < max_valid, f"{name} has invalid index {max_val} >= {max_valid}"
        )

    def _validate_per_cell_sentinels(self) -> None:
        """
        Validate per-cell sentinel consistency for variable-sized connectivity arrays.

        For arrays like edgesOnCell, cellsOnCell, verticesOnCell:
        - Slots [0:nEdgesOnCell[iCell]] must contain valid indices (>= 0)
        - Slots [nEdgesOnCell[iCell]:maxEdges] must contain sentinels (-1)

        This prevents silent indexing errors where a sentinel (-1) in a valid
        slot would cause array[-1] to silently access the last element.

        Raises:
            ValueError: If any cell has inconsistent sentinel placement.
        """
        maxEdges = self.maxEdges

        # Create a mask for valid slots: mask[iCell, j] = True if j < nEdgesOnCell[iCell]
        # Shape: (nCells, maxEdges)
        col_indices = torch.arange(maxEdges, device=self.nEdgesOnCell.device)
        valid_mask = col_indices.unsqueeze(0) < self.nEdgesOnCell.unsqueeze(1)

        # Sentinel mask is the complement
        sentinel_mask = ~valid_mask

        # For each connectivity array, check:
        # 1. Valid slots must have indices >= 0 (not sentinel)
        # 2. Sentinel slots must have indices == -1

        arrays_to_check = [
            (self.edgesOnCell, "edgesOnCell"),
            (self.cellsOnCell, "cellsOnCell"),
            (self.verticesOnCell, "verticesOnCell"),
        ]

        for arr, name in arrays_to_check:
            # Check 1: Valid slots should not contain sentinel (-1)
            valid_values = arr[valid_mask]
            sentinels_in_valid = (valid_values == -1).any()
            if sentinels_in_valid:
                # Find first offending cell for error message
                bad_cells = ((arr == -1) & valid_mask).any(dim=1)
                first_bad = int(bad_cells.nonzero(as_tuple=True)[0][0].item())
                n_edges = int(self.nEdgesOnCell[first_bad].item())
                _check(
                    False,
                    f"{name} has sentinel (-1) in valid slot for cell {first_bad} "
                    f"(nEdgesOnCell={n_edges}, but found -1 in first {n_edges} slots)",
                )

            # Check 2: Sentinel slots should contain only -1
            sentinel_values = arr[sentinel_mask]
            non_sentinels_in_padding = (sentinel_values != -1).any()
            if non_sentinels_in_padding:
                # Find first offending cell for error message
                bad_cells = ((arr != -1) & sentinel_mask).any(dim=1)
                first_bad = int(bad_cells.nonzero(as_tuple=True)[0][0].item())
                n_edges = int(self.nEdgesOnCell[first_bad].item())
                bad_val = arr[first_bad, n_edges:][arr[first_bad, n_edges:] != -1][
                    0
                ].item()
                _check(
                    False,
                    f"{name} has non-sentinel value {bad_val} in padding slot for cell {first_bad} "
                    f"(nEdgesOnCell={n_edges}, but found {bad_val} in slots >= {n_edges})",
                )
