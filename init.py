#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os

from src.references import (
    SUPPORTED_ASSEMBLIES,
    prepare_fasta_path,
    prepare_ucsc_reference,
    index_cpg_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize a genome-wide CpG reference for MVViewer."
    )
    parser.add_argument("reference", help="hg19, hg38, a FASTA path, or a CpG BED path with --index")
    parser.add_argument("--index", action="store_true", help="Index an existing sorted CpG BED/BED.gz without reading FASTA")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    last_contig = None

    def progress(contig: str) -> None:
        nonlocal last_contig
        if contig != last_contig:
            print(f"Preparing {contig}...", file=sys.stderr, flush=True)
            last_contig = contig

    try:
        cache_dir = os.environ.get("MVVIEWER_CACHE_DIR")
        if not cache_dir:
            cache_dir = Path(__file__).resolve().parent / "genomes"
            print('MVVIEWER_CACHE_DIR:',cache_dir)
        else:
            cache_dir = Path(cache_dir).resolve()
            print('MVVIEWER_CACHE_DIR set to',cache_dir)
        if args.index:
            result = index_cpg_reference(args.reference)
        elif args.reference in SUPPORTED_ASSEMBLIES:
            result = prepare_ucsc_reference(
                args.reference,
                cache_dir=cache_dir,
                progress_callback=progress,
            )
        else:
            result = prepare_fasta_path(
                args.reference,
                cache_dir=cache_dir,
                progress_callback=progress,
            )
    except Exception as exc:
        print(f"Reference initialization failed: {exc}", file=sys.stderr)
        return 1
    print(f"Reference ready: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
