"""
Utility functions for MPAS PyTorch port.

This package contains checkpoint I/O, validation utilities, and other
helper functions for development and testing.

Modules:
    checkpoint_reader: Read mesh and state from NetCDF checkpoint files
    namelist_parser: Parse MPAS namelist files for configuration
"""

from .checkpoint_reader import (
    CheckpointReader,
    VariableInfo,
    describe_checkpoint,
    load_checkpoint,
    load_mesh,
    load_mesh_for_kernels,
    load_mesh_from_checkpoints,
    load_state,
    save_checkpoint,
    validate_checkpoint,
)
from .namelist_parser import (
    compare_configs,
    get_namelist_keys,
    parse_namelist,
    parse_namelist_string,
)
from .sentinel import is_sentinel, is_valid_index
from .streams_parser import (
    INITIAL_ONLY,
    NEVER,
    StreamsConfig,
    StreamSpec,
    parse_interval,
    parse_streams,
)

__all__ = [
    "INITIAL_ONLY",
    "NEVER",
    "CheckpointReader",
    "StreamSpec",
    "StreamsConfig",
    "VariableInfo",
    "compare_configs",
    "describe_checkpoint",
    "get_namelist_keys",
    "is_sentinel",
    "is_valid_index",
    "load_checkpoint",
    "load_mesh",
    "load_mesh_for_kernels",
    "load_mesh_from_checkpoints",
    "load_state",
    "parse_interval",
    "parse_namelist",
    "parse_namelist_string",
    "parse_streams",
    "save_checkpoint",
    "validate_checkpoint",
]
