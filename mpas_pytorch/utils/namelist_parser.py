"""
Fortran namelist parser and writer for MPAS configuration files.

The public flat parser API is retained for existing runtime callers.
Grouped parsing/writing is added for namelist round-tripping and
MPAS configuration workflows.
"""

from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

NamelistGroups = dict[str, dict[str, Any]]


def _split_unquoted(line: str, target: str) -> list[str]:
    """Split a string on unquoted target characters."""
    parts: list[str] = []
    start = 0
    in_string = False
    quote_char = None
    i = 0
    while i < len(line):
        ch = line[i]
        if ch in ("'", '"'):
            if not in_string:
                in_string = True
                quote_char = ch
            elif ch == quote_char:
                if quote_char == "'" and i + 1 < len(line) and line[i + 1] == "'":
                    i += 1
                else:
                    in_string = False
                    quote_char = None
        elif ch == target and not in_string:
            parts.append(line[start:i].strip())
            start = i + 1
        i += 1
    parts.append(line[start:].strip())
    return parts


def _expand_repeat_count(token: str) -> list[str]:
    """Expand namelist repeat-count syntax such as ``3*0.5``."""
    if "*" not in token:
        return [token]
    count_str, value_str = token.split("*", 1)
    count_str = count_str.strip()
    value_str = value_str.strip()
    if not count_str.isdigit() or not value_str:
        return [token]
    return [value_str] * int(count_str)


def _strip_inline_comment(line: str) -> str:
    """Remove an inline namelist ``!`` comment unless it appears inside quotes."""
    in_string = False
    quote_char = None
    for i, ch in enumerate(line):
        if ch in ("'", '"') and not in_string:
            in_string = True
            quote_char = ch
        elif ch == quote_char and in_string:
            in_string = False
            quote_char = None
        elif ch == "!" and not in_string:
            return line[:i].strip()
    return line.strip()


def _find_unquoted_char(line: str, target: str) -> int:
    """Return the first unquoted target character index, or -1 if absent."""
    in_string = False
    quote_char = None
    for i, ch in enumerate(line):
        if ch in ("'", '"') and not in_string:
            in_string = True
            quote_char = ch
        elif ch == quote_char and in_string:
            in_string = False
            quote_char = None
        elif ch == target and not in_string:
            return i
    return -1


def _parse_value(value_str: str) -> Any:
    """
    Parse a Fortran namelist value string into a Python type.

    Args:
        value_str: Raw value string from namelist (whitespace-trimmed).

    Returns:
        Parsed value as bool, int, float, or str.
    """
    s = value_str.strip()
    if not s:
        return ""

    if _find_unquoted_char(s, ",") >= 0:
        values: list[Any] = []
        for token in _split_unquoted(s, ","):
            if not token:
                continue
            for expanded in _expand_repeat_count(token):
                values.append(_parse_value(expanded))
        return values

    expanded_values = _expand_repeat_count(s)
    if len(expanded_values) > 1:
        return [_parse_value(item) for item in expanded_values]

    s_lower = s.lower()
    if s_lower in ("true", ".true."):
        return True
    if s_lower in ("false", ".false."):
        return False

    if (s.startswith("'") and s.endswith("'")) or (
        s.startswith('"') and s.endswith('"')
    ):
        return s[1:-1].replace("''", "'")

    try:
        return int(s)
    except ValueError:
        pass

    try:
        return float(s.lower().replace("d", "e"))
    except ValueError:
        pass

    return s


def _format_value(value: Any) -> str:
    """Format a Python scalar as deterministic Fortran namelist syntax."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
        if not values:
            return ""
        non_bool_types = {
            bool if isinstance(item, bool) else type(item)
            for item in values
            if item is not None
        }
        if len(non_bool_types) > 1:
            raise ValueError(
                f"Mixed-type namelist arrays are not supported: {values!r}"
            )
        return ", ".join(_format_value(item) for item in values)
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if value is None:
        return "''"
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def parse_namelist_groups_string(content: str) -> NamelistGroups:
    """Parse namelist content while preserving section structure and key order."""
    groups: NamelistGroups = {}
    current_section: str | None = None

    for line_num, raw_line in enumerate(content.splitlines(), start=1):
        line = _strip_inline_comment(raw_line.strip())
        if not line:
            continue

        if line.startswith("&"):
            current_section = line[1:].strip()
            if not current_section:
                raise ValueError(f"Invalid namelist section header on line {line_num}")
            groups.setdefault(current_section, {})
            continue

        if line == "/":
            current_section = None
            continue

        if current_section is None:
            raise ValueError(
                f"Key/value pair outside any namelist section on line {line_num}"
            )

        eq_pos = _find_unquoted_char(line, "=")
        if eq_pos <= 0:
            raise ValueError(
                f"Invalid namelist assignment on line {line_num}: {raw_line.rstrip()}"
            )

        key = line[:eq_pos].strip()
        value_str = line[eq_pos + 1 :].strip()
        if value_str.endswith(","):
            value_str = value_str[:-1].strip()
        groups[current_section][key] = _parse_value(value_str)

    return groups


def parse_namelist_groups(filepath: str | Path) -> NamelistGroups:
    """Parse a namelist file while preserving section structure."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Namelist file not found: {filepath}")
    return parse_namelist_groups_string(filepath.read_text(encoding="utf-8"))


def parse_namelist(filepath: str | Path) -> dict[str, Any]:
    """
    Parse a Fortran namelist file into a flat dictionary.

    Section names are intentionally discarded for backward compatibility.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Namelist file not found: {filepath}")
    return parse_namelist_string(filepath.read_text(encoding="utf-8"))


def parse_namelist_string(content: str) -> dict[str, Any]:
    """Parse namelist content from a string into a flat dictionary."""
    result: dict[str, Any] = {}
    for group_values in parse_namelist_groups_string(content).values():
        result.update(group_values)
    return result


def write_namelist(path: str | Path, groups: NamelistGroups) -> None:
    """Write grouped namelist data using deterministic Fortran syntax."""
    lines: list[str] = []
    for group_name, values in groups.items():
        lines.append(f"&{group_name}")
        for key, value in values.items():
            lines.append(f"    {key} = {_format_value(value)}")
        lines.append("/")
        lines.append("")
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def get_namelist_keys(filepath: str | Path) -> list[str]:
    """Get all keys from a namelist file without preserving sections."""
    parsed = parse_namelist(filepath)
    return sorted(parsed.keys())


def compare_configs(
    pytorch_config: dict[str, Any],
    namelist_config: dict[str, Any],
    keys_to_compare: Collection[str] | None = None,
) -> list[tuple[str, Any, Any]]:
    """Compare PyTorch config against parsed namelist values."""
    mismatches = []
    if keys_to_compare is None:
        keys_to_compare = set(pytorch_config.keys()) & set(namelist_config.keys())

    for key in keys_to_compare:
        if key not in pytorch_config or key not in namelist_config:
            continue
        pt_val = pytorch_config[key]
        nl_val = namelist_config[key]
        if not _values_equal(pt_val, nl_val):
            mismatches.append((key, pt_val, nl_val))
    return mismatches


def _values_equal(val1: Any, val2: Any) -> bool:
    """Compare two values with type-aware tolerance for floats."""
    if isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            return False
        return all(_values_equal(left, right) for left, right in zip(val1, val2))

    if isinstance(val1, float) and isinstance(val2, float):
        if val1 == 0.0 and val2 == 0.0:
            return True
        if val1 == 0.0 or val2 == 0.0:
            return abs(val1 - val2) < 1e-15
        return abs(val1 - val2) / max(abs(val1), abs(val2)) < 1e-12

    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        return float(val1) == float(val2)

    if isinstance(val1, str) and isinstance(val2, str):
        return val1.lower() == val2.lower()

    return bool(val1 == val2)
