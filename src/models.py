from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Region:
    chrom: str
    start: int  # zero-based, inclusive
    end: int  # zero-based, exclusive

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def label(self) -> str:
        return f"{self.chrom}:{self.start + 1:,}-{self.end:,}"


@dataclass(frozen=True)
class TrackSpec:
    name: str
    alignment_path: Path
    variant_path: Path | None = None

    @property
    def is_annotation(self) -> bool:
        return self.alignment_path.name.lower().endswith((".gtf", ".gtf.gz"))


_COLON_REGION_RE = re.compile(r"^\s*([^:\s]+)\s*:\s*([\d,]+)\s*-\s*([\d,]+)\s*$")
_COLUMNS_REGION_RE = re.compile(r"^\s*([^:\s]+)[ \t]+([\d,]+)[ \t]+([\d,]+)\s*$")


def parse_region(text: str, max_length: int = 200_000) -> Region:
    match = _COLON_REGION_RE.match(text) or _COLUMNS_REGION_RE.match(text)
    if not match:
        raise ValueError(
            "Use a region such as chr1:1,000-2,000 or chr1 1,000 2,000"
        )
    chrom, raw_start, raw_end = match.groups()
    start = int(raw_start.replace(",", ""))
    end = int(raw_end.replace(",", ""))
    if start < 1:
        raise ValueError("Region start must be at least 1")
    if end < start:
        raise ValueError("Region end must not be before its start")
    if end - start + 1 > max_length:
        raise ValueError(f"Region is larger than the {max_length:,} bp limit")
    return Region(chrom, start - 1, end)


def parse_track_lines(text: str, base_dir: Path | None = None) -> list[TrackSpec]:
    root = (base_dir or Path.cwd()).resolve()
    tracks: list[TrackSpec] = []
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in next(csv.reader([raw_line], skipinitialspace=True))]
        if len(fields) not in (1, 2, 3) or not fields[0]:
            raise ValueError(
                f"Track line {line_no}: expected BAM, optional VCF, optional name; or GTF/GTF.gz, optional name"
            )
        alignment = Path(fields[0]).expanduser()
        if not alignment.is_absolute():
            alignment = root / alignment
        if alignment.name.lower().endswith((".gtf", ".gtf.gz")):
            if len(fields) > 2:
                raise ValueError(f"Track line {line_no}: use GTF/GTF.gz, optional name")
            name = fields[1] if len(fields) == 2 and fields[1] else alignment.name
            tracks.append(TrackSpec(name, alignment.resolve()))
            continue
        variant: Path | None = None
        if len(fields) >= 2 and fields[1]:
            variant = Path(fields[1]).expanduser()
            if not variant.is_absolute():
                variant = root / variant
        name = fields[2] if len(fields) == 3 and fields[2] else alignment.name
        tracks.append(TrackSpec(name, alignment.resolve(), variant.resolve() if variant else None))
    names = [track.name for track in tracks]
    if len(names) != len(set(names)):
        raise ValueError("Track names must be unique; supply optional names to distinguish tracks")
    return tracks
