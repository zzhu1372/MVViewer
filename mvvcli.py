#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import data_processing, plotting
from src.models import parse_region, parse_track_lines
from src.plotly_utils import as_plotly_payload, plotly_pdf_bytes
from src.references import list_prepared_references, open_cpg_reference


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MVViewer command-line interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tracks", required=True, type=Path,
                        help="Track TXT file; relative track paths are resolved beside this file")
    parser.add_argument("--region", required=True,
                        help='1-based region, e.g. "chr19:56837539-56840146"')
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="Output .pdf, .png, .svg, or .html (HTML requires Plotly)")
    parser.add_argument("--reference", help="Prepared reference name (e.g. hg38) or indexed CpG BED.gz path; required for BAM tracks", required=True)
    parser.add_argument("--renderer", choices=["plotly", "seaborn"],
                        default=argparse.SUPPRESS,
                        help="Renderer override; by default use Plotly for up to 10 tracks, "
                             "Seaborn for more than 10 (faster). "
                             "Use --renderer plotly for HTML output.")
    parser.add_argument("--modification-cutoff", type=float, default=0.8)
    parser.add_argument("--max-indel-length", type=int, default=50)
    parser.add_argument("--sort-by", choices=["methylation", "position", "hp-strand"], default="methylation")
    parser.add_argument("--sort-position", type=int, help="1-based position; required with --sort-by position")
    parser.add_argument("--pvalue-cutoff", type=float, default=0.05)
    parser.add_argument("--original-genomic-coordinates", action="store_true")
    parser.add_argument("--webgl-pixel-base", type=float, default=plotting.WEBGL_PIXEL_BASE)
    parser.add_argument("--max-webgl-figure-height", type=int, default=plotting.MAX_WEBGL_FIGURE_HEIGHT)
    parser.add_argument("--source-data", type=Path, help="Also save plotted calls and GTF annotations as TSV")
    args = parser.parse_args(argv)
    args.renderer = getattr(args, "renderer", None)
    for name in ("modification_cutoff", "pvalue_cutoff"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.max_indel_length < 1 or args.max_webgl_figure_height < 1 or not 0 < args.webgl_pixel_base < float("inf"):
        parser.error("Indel length, WebGL height, and pixel base must be positive")
    if args.sort_by == "position" and args.sort_position is None:
        parser.error("--sort-by position requires --sort-position")
    if args.sort_by != "position" and args.sort_position is not None:
        parser.error("--sort-position requires --sort-by position")
    allowed = {".pdf", ".png", ".svg", ".html"} if args.renderer != "seaborn" else {".pdf", ".png", ".svg"}
    if args.output.suffix.lower() not in allowed:
        parser.error(f"Unsupported output extension for {args.renderer}: choose {', '.join(sorted(allowed))}")
    args.output = args.output.expanduser().resolve()
    if args.source_data:
        args.source_data = args.source_data.expanduser().resolve()
        if args.source_data == args.output:
            parser.error("Figure and source-data output paths must be different")
    return args


def resolve_reference(value):
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    matches = [ref.path for ref in list_prepared_references()
               if value in {ref.label, ref.path.name, ref.path.name.removesuffix(".CpG.bed.gz")}]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Reference not found or ambiguous: {value}. Initialize it with init.py or supply its CpG BED path.")


def render(args):
    def progress(message):
        print(message, file=sys.stderr, flush=True)

    region = parse_region(args.region)
    if args.sort_position is not None and not region.start + 1 <= args.sort_position <= region.end:
        raise ValueError("Sort position must be within the selected region")
    tracks_path = args.tracks.expanduser().resolve()
    tracks = parse_track_lines(tracks_path.read_text(encoding="utf-8-sig"), tracks_path.parent)
    if not tracks:
        raise ValueError("The track file contains no tracks")
    renderer = args.renderer or ("seaborn" if len(tracks) > 10 else "plotly")
    selection = "explicit choice" if args.renderer else "automatic selection"
    progress(f"Renderer: {renderer} ({selection}; {len(tracks)} tracks)")
    if renderer == "seaborn" and args.output.suffix.lower() == ".html":
        raise ValueError("HTML output requires Plotly; use --renderer plotly or choose PDF, PNG, or SVG")
    inputs = {tracks_path}
    for track in tracks:
        for path in (track.alignment_path, track.variant_path):
            if path is not None:
                if not path.is_file():
                    raise FileNotFoundError(f"Track file does not exist: {path}")
                inputs.add(path)
    alignment_tracks = [track for track in tracks if not track.is_annotation]
    if alignment_tracks and not args.reference:
        raise ValueError("BAM tracks require --reference (run init.py first)")
    reference = resolve_reference(args.reference) if args.reference else None
    if reference:
        inputs.add(reference)
    for output in (args.output, args.source_data):
        if output is not None:
            if output in inputs:
                raise ValueError(f"Output would overwrite an input file: {output}")
            if not output.parent.is_dir():
                raise ValueError(f"Output directory does not exist: {output.parent}")
    cpg_table = pd.DataFrame(columns=[0, 1])
    if alignment_tracks:
        progress("Opening reference index")
        cpg_table = open_cpg_reference(reference)
    loaded = data_processing.load_region_data(
        region.chrom, region.start, region.end,
        [str(track.alignment_path) for track in alignment_tracks],
        [str(track.variant_path) if track.variant_path else None for track in alignment_tracks],
        [track.name for track in alignment_tracks], cpg_table, progress_callback=progress,
    )
    loaded["annotations"] = {}
    for track in tracks:
        if track.is_annotation:
            progress("Loading " + track.name)
            loaded["annotations"][track.name] = data_processing.load_gtf_track(
                track.alignment_path, region.chrom, loaded["long_start"], loaded["long_end"],
            )
    loaded["track_order"] = [track.name for track in tracks]
    data = data_processing.process_region_data(
        loaded, max_indel_length=args.max_indel_length,
        modification_cutoff=args.modification_cutoff, sort_position=args.sort_position,
        sort_methylation_average=args.sort_by == "methylation",
        coordinate_mode="genomic" if args.original_genomic_coordinates else "compact",
        progress_callback=progress,
    )
    if renderer == "plotly":
        figure = plotting.plot_plotly(
            data, progress_callback=progress, pvalue_cutoff=args.pvalue_cutoff,
            webgl_pixel_base=args.webgl_pixel_base,
            max_webgl_figure_height=args.max_webgl_figure_height,
        )
        if args.output.suffix.lower() == ".html":
            ratio = figure.layout.meta["webgl_pixel_ratio"]
            figure.write_html(args.output, include_plotlyjs=True, auto_open=False,
                              config={"scrollZoom": False, "displaylogo": False,
                                      "plotGlPixelRatio": ratio,
                                      "modeBarButtonsToAdd": ["drawline", "eraseshape"]})
        elif args.output.suffix.lower() == ".pdf":
            args.output.write_bytes(plotly_pdf_bytes(as_plotly_payload(figure)))
        else:
            payload = as_plotly_payload(figure)
            if args.output.suffix.lower() == ".svg":
                for trace in payload["data"]:
                    if trace.get("type") == "scattergl":
                        trace["type"] = "scatter"
            import plotly.io as pio
            pio.write_image(payload, args.output, format=args.output.suffix[1:].lower())
    else:
        figure = plotting.plot_seaborn(data, progress_callback=progress, pvalue_cutoff=args.pvalue_cutoff)
        try:
            figure.savefig(args.output, format=args.output.suffix[1:].lower(), dpi=100, bbox_inches="tight")
        finally:
            plt.close(figure)
    if args.source_data:
        data.to_source_dataframe().to_csv(args.source_data, sep="\t", index=False)
        progress(f"Source data saved: {args.source_data}")
    progress(f"Figure saved: {args.output}")


def main(argv=None):
    args = parse_args(argv)
    try:
        render(args)
    except Exception as exc:
        print(f"MVViewer could not render this region: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
