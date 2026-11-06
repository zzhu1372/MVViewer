from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
import tempfile
from typing import BinaryIO, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pysam

from .models import Region


UCSC_SEQUENCE_API = "https://api.genome.ucsc.edu/getData/sequence"
SUPPORTED_ASSEMBLIES = ("hg19", "hg38")
DEFAULT_GENOMES_DIR = Path(__file__).resolve().parent.parent / "genomes"
UCSC_FASTA_URLS = {
    "hg19": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz",
    "hg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
}


@dataclass(frozen=True)
class PreparedReference:
    label: str
    path: Path


@dataclass(frozen=True)
class CpGReference:
    path: Path
    index_path: Path
    contigs: tuple[str, ...]

    def fetch_positions(self, chrom: str, start: int, end: int) -> np.ndarray:
        if chrom not in self.contigs:
            return np.empty(0, dtype=np.int64)
        with pysam.TabixFile(str(self.path), index=str(self.index_path)) as reference:
            rows = reference.fetch(chrom, max(0, start), end + 1)
            return np.fromiter(
                (position for row in rows
                 if start <= (position := int(row.split("\t", 2)[1])) <= end),
                dtype=np.int64,
            )


def open_cpg_reference(path: str | Path) -> CpGReference:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CpG reference does not exist: {path}")
    for suffix in (".csi", ".tbi"):
        index = Path(str(path) + suffix)
        if index.is_file() and index.stat().st_mtime_ns >= path.stat().st_mtime_ns:
            try:
                with pysam.TabixFile(str(path), index=str(index)) as reader:
                    return CpGReference(path, index, tuple(reader.contigs))
            except (OSError, ValueError):
                continue
    raise ValueError(
        f'CpG reference needs a current Tabix index. Run outside the web app: python init.py --index "{path}"'
    )


def _publish_indexed_cpg(temporary: Path, target: Path) -> None:
    pysam.tabix_index(str(temporary), preset="bed", force=True, csi=True)
    index = Path(str(temporary) + ".csi")
    with pysam.TabixFile(str(temporary), index=str(index)):
        pass
    os.replace(temporary, target)
    os.replace(index, Path(str(target) + ".csi"))
    Path(str(target) + ".tbi").unlink(missing_ok=True)


def index_cpg_reference(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"CpG reference does not exist: {source}")
    try:
        return open_cpg_reference(source).path
    except ValueError:
        pass
    target = source if source.suffix.lower() == ".gz" else Path(str(source) + ".gz")
    if target != source and target.exists():
        raise FileExistsError(f"Target already exists; index it directly instead: {target}")
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".bed.gz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with source.open("rb") as handle:
            compressed = handle.read(2) == b"\x1f\x8b"
        opener = gzip.open if compressed else open
        with opener(source, "rb") as input_file, pysam.BGZFile(str(temporary), "w") as output:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
        _publish_indexed_cpg(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
        Path(str(temporary) + ".csi").unlink(missing_ok=True)
    return target


def reference_cache_dir() -> Path:
    configured = os.environ.get("MVVIEWER_CACHE_DIR")
    root = Path(configured).expanduser() if configured else DEFAULT_GENOMES_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def prepared_reference_dir(cache_dir: Path | None = None) -> Path:
    if cache_dir is not None or os.environ.get("MVVIEWER_CACHE_DIR"):
        return (cache_dir or reference_cache_dir()).expanduser().resolve()
    return DEFAULT_GENOMES_DIR.resolve()


def _prepared_target(reference_id: str, cache_dir: Path | None = None) -> Path:
    return prepared_reference_dir(cache_dir) / f"{reference_id}.CpG.bed.gz"


def _metadata_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".json")


def _write_metadata(target: Path, label: str, source: str) -> None:
    metadata = {"label": label, "source": source}
    _metadata_path(target).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def list_prepared_references(cache_dir: Path | None = None) -> list[PreparedReference]:
    roots = [prepared_reference_dir(cache_dir)]
    paths = {}
    for root in roots:
        for path in sorted(root.glob(f"*.CpG.bed.gz")):
            paths.setdefault(path.name, path)
    references: list[PreparedReference] = []
    for path in sorted(paths.values(), key=lambda path: path.name):
        label = path.name.removesuffix(".CpG.bed.gz")
        metadata_path = _metadata_path(path)
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                label = str(metadata.get("label") or label)
            except (OSError, json.JSONDecodeError):
                pass
        references.append(PreparedReference(label, path.resolve()))
    return references


def fasta_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def prepared_ucsc_reference(assembly: str, cache_dir: Path | None = None) -> Path | None:
    target = _prepared_target(assembly, cache_dir)
    return target if target.is_file() else None


def prepared_fasta_reference(data: bytes, cache_dir: Path | None = None) -> Path | None:
    target = _prepared_target(f"custom_{fasta_digest(data)}", cache_dir)
    return target if target.is_file() else None


def _write_cpg_pair(output: io.TextIOBase, chrom: str, start: int) -> None:
    output.write(f"{chrom}\t{start}\t{start + 1}\t.\t0\t+\n")
    output.write(f"{chrom}\t{start + 1}\t{start + 2}\t.\t0\t-\n")


def _prepare_from_fasta_stream(
    fasta: io.TextIOBase,
    target: Path,
    progress_callback=None,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    cpg_count = 0
    chrom: str | None = None
    chrom_position = 0
    previous_base = ""
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".bed.gz", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        with io.TextIOWrapper(pysam.BGZFile(str(temporary_path), "w"), encoding="utf-8", newline="") as output:
            for raw_line in fasta:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    chrom = line[1:].split()[0]
                    chrom_position = 0
                    previous_base = ""
                    if progress_callback is not None:
                        progress_callback(chrom)
                    continue
                if chrom is None:
                    raise ValueError("Invalid FASTA: sequence appears before the first header")
                sequence = line.upper()
                if previous_base == "C" and sequence.startswith("G"):
                    _write_cpg_pair(output, chrom, chrom_position - 1)
                    cpg_count += 1
                position = sequence.find("CG")
                while position >= 0:
                    _write_cpg_pair(output, chrom, chrom_position + position)
                    cpg_count += 1
                    position = sequence.find("CG", position + 1)
                chrom_position += len(sequence)
                previous_base = sequence[-1:] or previous_base
        if cpg_count == 0:
            raise ValueError("No CpG sites were found in the FASTA")
        _publish_indexed_cpg(temporary_path, target)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
        if temporary_path:
            Path(str(temporary_path) + ".csi").unlink(missing_ok=True)
    return target


def prepare_fasta_reference(
    data: bytes,
    cache_dir: Path | None = None,
    progress_callback=None,
) -> Path:
    if not data:
        raise ValueError("Upload a non-empty FASTA file")
    target = _prepared_target(f"custom_{fasta_digest(data)}", cache_dir)
    if target.is_file():
        return index_cpg_reference(target)
    with _fasta_text_stream(data) as fasta:
        result = _prepare_from_fasta_stream(fasta, target, progress_callback)
    _write_metadata(result, f"custom:{fasta_digest(data)}", "uploaded FASTA")
    return result


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


@contextmanager
def _fasta_path_stream(path: Path) -> Iterator[io.TextIOWrapper]:
    raw = path.open("rb")
    if path.suffix.lower() == ".gz":
        binary: BinaryIO = gzip.GzipFile(fileobj=raw)
    else:
        magic = raw.read(2)
        raw.seek(0)
        binary = gzip.GzipFile(fileobj=raw) if magic == b"\x1f\x8b" else raw
    text = io.TextIOWrapper(binary, encoding="utf-8")
    try:
        yield text
    finally:
        text.close()


def prepare_fasta_path(
    fasta_path: str | Path,
    cache_dir: Path | None = None,
    progress_callback=None,
) -> Path:
    path = Path(fasta_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FASTA does not exist: {path}")
    digest = _file_digest(path)
    stem = path.name[:-3] if path.name.lower().endswith(".gz") else path.name
    for suffix in (".fasta", ".fna", ".fa"):
        if stem.lower().endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    safe_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem) or "reference"
    target = _prepared_target(f"custom_{safe_stem}_{digest}", cache_dir)
    if target.is_file():
        return index_cpg_reference(target)
    with _fasta_path_stream(path) as fasta:
        result = _prepare_from_fasta_stream(fasta, target, progress_callback)
    _write_metadata(result, safe_stem, str(path))
    return result


def prepare_ucsc_reference(
    assembly: str,
    cache_dir: Path | None = None,
    progress_callback=None,
) -> Path:
    if assembly not in UCSC_FASTA_URLS:
        raise ValueError(f"Unsupported reference assembly: {assembly}")
    target = _prepared_target(assembly, cache_dir)
    if target.is_file():
        return index_cpg_reference(target)
    request = Request(UCSC_FASTA_URLS[assembly], headers={"User-Agent": "MVViewer/1.0"})
    try:
        with urlopen(request, timeout=120) as response:
            compressed = gzip.GzipFile(fileobj=response)
            fasta = io.TextIOWrapper(compressed, encoding="utf-8")
            try:
                result = _prepare_from_fasta_stream(fasta, target, progress_callback)
            finally:
                fasta.close()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Could not initialize {assembly} from UCSC. Check network access: {exc}"
        ) from exc
    _write_metadata(result, assembly, UCSC_FASTA_URLS[assembly])
    return result


def _write_cpg_bed(chrom: str, sequence: str, sequence_start: int, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".bed.gz", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        with gzip.open(temporary_path, "wt", encoding="utf-8", newline="") as output:
            upper = sequence.upper()
            position = upper.find("CG")
            while position >= 0:
                start = sequence_start + position
                output.write(f"{chrom}\t{start}\t{start + 1}\t.\t0\t+\n")
                output.write(f"{chrom}\t{start + 1}\t{start + 2}\t.\t0\t-\n")
                count += 1
                position = upper.find("CG", position + 1)
        os.replace(temporary_path, target)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
    return count


def _window(region: Region, flank: int = 1) -> tuple[int, int]:
    return max(0, region.start - flank), region.end + flank


def alignment_span(region: Region, alignment_paths: list[Path]) -> Region:
    import pysam

    start, end = region.start, region.end
    for path in alignment_paths:
        mode = "rc" if path.suffix.lower() == ".cram" else "rb"
        with pysam.AlignmentFile(str(path), mode) as alignment:
            for read in alignment.fetch(region.chrom, region.start, region.end):
                if read.is_unmapped or read.reference_end is None:
                    continue
                start = min(start, read.reference_start)
                end = max(end, read.reference_end)
    return Region(region.chrom, max(0, start), end)


def cpg_bed_for_ucsc(assembly: str, region: Region, cache_dir: Path | None = None) -> Path:
    if assembly not in SUPPORTED_ASSEMBLIES:
        raise ValueError(f"Unsupported reference assembly: {assembly}")
    start, end = _window(region)
    cache = cache_dir or reference_cache_dir()
    safe_chrom = "".join(c if c.isalnum() or c in "._-" else "_" for c in region.chrom)
    target = cache / f"{assembly}_{safe_chrom}_{start}_{end}.CpG.bed.gz"
    if target.is_file():
        return target

    url = (
        f"{UCSC_SEQUENCE_API}?genome={assembly};chrom={region.chrom};"
        f"start={start};end={end}"
    )
    request = Request(url, headers={"User-Agent": "MVViewer/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not retrieve {assembly} sequence from UCSC. Check network access and contig naming: {exc}"
        ) from exc
    sequence = payload.get("dna")
    if not isinstance(sequence, str):
        message = payload.get("error") or payload.get("warning") or "response contained no DNA"
        raise RuntimeError(f"UCSC could not return {assembly} {region.chrom}: {message}")
    _write_cpg_bed(region.chrom, sequence, start, target)
    return target


@contextmanager
def _fasta_text_stream(data: bytes) -> Iterator[io.TextIOWrapper]:
    raw = io.BytesIO(data)
    if data.startswith(b"\x1f\x8b"):
        binary: BinaryIO = gzip.GzipFile(fileobj=raw)
    else:
        binary = raw
    text = io.TextIOWrapper(binary, encoding="utf-8")
    try:
        yield text
    finally:
        text.close()


def _fasta_window(data: bytes, chrom: str, start: int, end: int) -> str:
    pieces: list[str] = []
    current_name: str | None = None
    current_position = 0
    found = False
    with _fasta_text_stream(data) as fasta:
        for raw_line in fasta:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if found:
                    break
                current_name = line[1:].split()[0]
                current_position = 0
                found = current_name == chrom
                continue
            if not found:
                continue
            line_end = current_position + len(line)
            overlap_start = max(start, current_position)
            overlap_end = min(end, line_end)
            if overlap_start < overlap_end:
                pieces.append(line[overlap_start - current_position:overlap_end - current_position])
            current_position = line_end
            if current_position >= end:
                break
    if not found:
        raise ValueError(f"Contig {chrom!r} was not found in the uploaded FASTA")
    sequence = "".join(pieces)
    expected = end - start
    if len(sequence) != expected:
        raise ValueError(
            f"FASTA contig {chrom!r} ends before coordinate {end:,} "
            f"(read {len(sequence):,} of {expected:,} requested bases)"
        )
    return sequence


def cpg_bed_for_fasta(data: bytes, region: Region, cache_dir: Path | None = None) -> Path:
    if not data:
        raise ValueError("Upload a non-empty FASTA file")
    start, end = _window(region)
    digest = hashlib.sha256(data).hexdigest()[:16]
    cache = cache_dir or reference_cache_dir()
    safe_chrom = "".join(c if c.isalnum() or c in "._-" else "_" for c in region.chrom)
    target = cache / f"custom_{digest}_{safe_chrom}_{start}_{end}.CpG.bed.gz"
    if target.is_file():
        return target
    sequence = _fasta_window(data, region.chrom, start, end)
    _write_cpg_bed(region.chrom, sequence, start, target)
    return target
