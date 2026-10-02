"""Parse the supported MPAS streams XML with independent input/output alarms."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from mpas_pytorch.exceptions import MpasConfigError


class _IntervalSentinel:
    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


INITIAL_ONLY = _IntervalSentinel("INITIAL_ONLY")
NEVER = _IntervalSentinel("NEVER")
_TOKEN_MAP = {
    "$Y": slice(0, 4),
    "$M": slice(5, 7),
    "$D": slice(8, 10),
    "$h": slice(11, 13),
    "$m": slice(14, 16),
    "$s": slice(17, 19),
}
_SUPPORTED_STREAM_ATTRS = {
    "name",
    "type",
    "filename_template",
    "output_interval",
    "input_interval",
    "clobber_mode",
    "precision",
    "packages",
    "filename_interval",
}
_VALID_CLOBBER_MODES = {
    "replace_files",
    "overwrite",
    "truncate",
    "append",
    "never_modify",
}
_VALID_PRECISIONS = {"single", "double", "native"}
_VALID_KINDS = {"input", "output", "input;output", "none"}
StreamKind = Literal["input", "output", "input;output", "none"]
_INTERVAL_RE = re.compile(
    r"^(?:(?P<days>\d+)_)?(?P<hours>\d+):(?P<minutes>\d{2}):(?P<seconds>\d{2})$"
)


@dataclass(frozen=True)
class StreamSpec:
    name: str
    kind: StreamKind
    immutable: bool
    filename_template: str
    interval_seconds: int | object
    clobber_mode: str
    precision: str
    fields: tuple[str, ...]
    packages: str | None = None
    input_interval_seconds: int | object = NEVER
    output_interval_seconds: int | object = NEVER
    filename_interval: str = "output_interval"

    def render(self, xtime: str) -> str:
        if (
            len(xtime) != 19
            or xtime[4] != "-"
            or xtime[7] != "-"
            or xtime[10] != "_"
            or xtime[13] not in {":", "."}
            or xtime[16] not in {":", "."}
        ):
            raise MpasConfigError(f"Invalid MPAS timestamp: {xtime!r}")
        result = self.filename_template
        for token, where in _TOKEN_MAP.items():
            result = result.replace(token, xtime[where])
        if "$" in result:
            raise MpasConfigError(f"Unsupported filename token in stream {self.name!r}")
        return result


@dataclass(frozen=True)
class StreamsConfig:
    input_streams: tuple[StreamSpec, ...]
    output_streams: tuple[StreamSpec, ...]
    restart_stream: StreamSpec | None
    disabled_streams: tuple[StreamSpec, ...] = ()


def _parse_interval(raw: str) -> int | _IntervalSentinel:
    norm = raw.strip().lower()
    if norm == "initial_only":
        return INITIAL_ONLY
    if norm in {"none", "never"}:
        return NEVER
    match = _INTERVAL_RE.fullmatch(norm)
    if match is None:
        raise MpasConfigError(f"Invalid MPAS stream interval: {raw!r}")
    days = int(match.group("days") or 0)
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    total = (((days * 24) + hours) * 60 + minutes) * 60 + seconds
    if minutes >= 60 or seconds >= 60 or total <= 0:
        raise MpasConfigError(
            f"Invalid MPAS stream interval: {raw!r}; require positive time and minutes/seconds below 60"
        )
    return total


def _read_field_list(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MpasConfigError(f"Cannot read stream-list file {path}: {exc}") from exc
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _reject_unknown_attributes(element: ET.Element, allowed: set[str]) -> None:
    unknown = set(element.attrib) - allowed
    if unknown:
        raise MpasConfigError(
            f"Unsupported streams.atmosphere attributes: {sorted(unknown)}"
        )


def _build_stream_spec(
    element: ET.Element, root_dir: Path, *, immutable: bool
) -> StreamSpec:
    _reject_unknown_attributes(element, _SUPPORTED_STREAM_ATTRS)
    name = element.get("name", "").strip()
    kind = element.get("type", "").strip()
    template = element.get("filename_template", "").strip()
    if not name or kind not in _VALID_KINDS or not template:
        raise MpasConfigError(
            "Every stream requires a name, supported type and filename_template"
        )
    input_alarm = _parse_interval(element.get("input_interval", "none"))
    output_alarm = _parse_interval(element.get("output_interval", "none"))
    clobber = element.get("clobber_mode", "never_modify").strip()
    precision = element.get("precision", "native").strip()
    if clobber not in _VALID_CLOBBER_MODES:
        raise MpasConfigError(f"Unsupported clobber_mode for {name!r}: {clobber!r}")
    if precision not in _VALID_PRECISIONS:
        raise MpasConfigError(f"Unsupported precision for {name!r}: {precision!r}")
    fields = []
    for child in element:
        if child.tag not in {"file", "var", "var_array"}:
            raise MpasConfigError(
                f"Unsupported streams.atmosphere child element: {child.tag}"
            )
        _reject_unknown_attributes(child, {"name"})
        field = child.get("name", "").strip()
        if not field:
            raise MpasConfigError(f"Stream {name!r} has <{child.tag}> without a name")
        if child.tag == "file":
            relative = Path(field)
            if relative.is_absolute() or ".." in relative.parts:
                raise MpasConfigError(
                    "Stream-list paths must remain inside the streams directory"
                )
            fields.extend(_read_field_list(root_dir / relative))
        else:
            if child.tag == "var_array" and field != "scalars":
                raise MpasConfigError(f"Unsupported variable array: {field!r}")
            fields.append(field)
    # Retain the old convenience attribute for callers; scheduling uses the two alarms.
    legacy = output_alarm if "output_interval" in element.attrib else input_alarm
    return StreamSpec(
        name=name,
        kind=cast(StreamKind, kind),
        immutable=immutable,
        filename_template=template,
        interval_seconds=legacy,
        clobber_mode=clobber,
        precision=precision,
        fields=tuple(dict.fromkeys(fields)),
        packages=element.get("packages"),
        input_interval_seconds=input_alarm,
        output_interval_seconds=output_alarm,
        filename_interval=element.get("filename_interval", "output_interval"),
    )


def parse_streams(path: str | Path) -> StreamsConfig:
    source = Path(path)
    try:
        root = ET.parse(source).getroot()
    except (OSError, ET.ParseError) as exc:
        raise MpasConfigError(f"Cannot read streams file {source}: {exc}") from exc
    if root.tag != "streams":
        raise MpasConfigError(f"Expected <streams> root, found <{root.tag}>")
    _reject_unknown_attributes(root, set())
    inputs = []
    outputs = []
    disabled = []
    restart = None
    names = set()
    for child in root:
        if child.tag not in {"stream", "immutable_stream"}:
            raise MpasConfigError(
                f"Unsupported streams.atmosphere element: {child.tag}"
            )
        spec = _build_stream_spec(
            child, source.parent, immutable=child.tag == "immutable_stream"
        )
        if spec.name in names:
            raise MpasConfigError(f"Duplicate stream name: {spec.name!r}")
        names.add(spec.name)
        if spec.kind in {"input", "input;output"}:
            inputs.append(spec)
        if spec.kind in {"output", "input;output"}:
            outputs.append(spec)
        if spec.kind == "none":
            disabled.append(spec)
        if spec.name == "restart":
            restart = spec
    return StreamsConfig(tuple(inputs), tuple(outputs), restart, tuple(disabled))


def parse_interval(raw: str) -> int | _IntervalSentinel:
    """Parse positive MPAS duration/cadence, initial_only, none or never."""
    return _parse_interval(raw)


__all__ = [
    "INITIAL_ONLY",
    "NEVER",
    "StreamSpec",
    "StreamsConfig",
    "parse_interval",
    "parse_streams",
]
