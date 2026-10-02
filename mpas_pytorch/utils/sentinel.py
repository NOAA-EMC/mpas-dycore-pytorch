"""
Sentinel value handling for MPAS connectivity arrays.

MPAS-Atmosphere uses two sentinel values in connectivity arrays to mark
invalid or boundary entries:
- 0: Indicates no valid neighbor (boundary edge with no adjacent cell)
- max_valid + 1: Padding entry beyond the valid range (Fortran 1-based indexing)

For example, with nCells=2562:
- Valid cell indices (1-based): 1 to 2562
- Sentinel for boundary: 0
- Sentinel for padding: 2563 (nCells + 1)

These functions provide consistent sentinel checking across all kernel code.
"""

import torch


def is_sentinel(value: int | torch.Tensor, max_valid: int) -> bool:
    """
    Check if a 1-based index is a sentinel value.

    A value is a sentinel if it is either:
    - 0 (boundary marker, no valid neighbor)
    - max_valid + 1 (padding entry beyond valid range)

    Args:
        value: The 1-based index to check. Can be int or 0-d torch.Tensor.
        max_valid: Maximum valid index (e.g., nCells for cell connectivity,
                   nEdges for edge connectivity, nVertices for vertex connectivity).

    Returns:
        True if value is a sentinel (0 or max_valid+1), False if valid.

    Examples:
        >>> is_sentinel(0, nCells=2562)  # Boundary marker
        True
        >>> is_sentinel(2563, nCells=2562)  # Padding entry
        True
        >>> is_sentinel(1, nCells=2562)  # Valid cell index
        False
        >>> is_sentinel(2562, nCells=2562)  # Last valid cell
        False
        >>> is_sentinel(torch.tensor(0), nCells=2562)  # Tensor input
        True
    """
    if isinstance(value, torch.Tensor):
        # Extract scalar value from 0-d tensor
        v = int(value.item())
    else:
        v = int(value)

    # Sentinel values are 0 (boundary) or max_valid+1 (padding)
    return v == 0 or v == max_valid + 1


def is_sentinel_batch(tensor: torch.Tensor, max_valid: int) -> torch.Tensor:
    """Vectorized sentinel check for 0-based index tensors.

    Returns a boolean mask on the same device as ``tensor`` where ``True``
    marks sentinel/out-of-range entries. This is intended for batched kernels
    that already converted MPAS 1-based connectivity indices to 0-based.

    A value is treated as sentinel if it is outside [0, max_valid].
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be torch.Tensor, got {type(tensor)!r}")
    return (tensor < 0) | (tensor > max_valid)


def is_valid_index(value: int | torch.Tensor, max_valid: int) -> bool:
    """
    Check if a 1-based index is a valid (non-sentinel) value.

    A value is valid if it is in the range [1, max_valid] (inclusive).
    This is the inverse of is_sentinel().

    Args:
        value: The 1-based index to check. Can be int or 0-d torch.Tensor.
        max_valid: Maximum valid index (e.g., nCells for cell connectivity,
                   nEdges for edge connectivity, nVertices for vertex connectivity).

    Returns:
        True if value is valid (in range [1, max_valid]), False if sentinel.

    Examples:
        >>> is_valid_index(1, nCells=2562)  # First valid cell
        True
        >>> is_valid_index(2562, nCells=2562)  # Last valid cell
        True
        >>> is_valid_index(0, nCells=2562)  # Boundary marker (sentinel)
        False
        >>> is_valid_index(2563, nCells=2562)  # Padding entry (sentinel)
        False
        >>> is_valid_index(torch.tensor(100), nCells=2562)  # Tensor input
        True
    """
    return not is_sentinel(value, max_valid)
