from __future__ import annotations

import time
import importlib
import io
from pathlib import Path
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from src.models import parse_region, parse_track_lines
from src import data_processing, plotting
from src.plotly_utils import as_plotly_payload, plotly_pdf_bytes
from src.references import list_prepared_references, open_cpg_reference


PLOT_LOCK = threading.Lock()


def input_file_version(path: Path | None):
    if path is None:
        return None
    candidates = [path, *[Path(str(path) + suffix)
                          for suffix in (".bai", ".csi", ".crai", ".tbi")],
                  path.with_suffix(".bai"), path.with_suffix(".crai")]
    versions = []
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            info = candidate.stat()
            versions.append((str(candidate.resolve()), info.st_size, info.st_mtime_ns))
    return tuple(versions)


def source_data_bytes(result: dict) -> bytes:
    with result["source_data_lock"]:
        if result.get("plot_data_tsv") is None:
            data = result.get("plot_data")
            frame = data.to_source_dataframe() if data is not None else result["plot_data_calls"]
            result["plot_data_tsv"] = frame.to_csv(index=False, sep="\t").encode("utf-8")
        return result["plot_data_tsv"]


def show_plot_download(slot, result: dict) -> None:
    result.setdefault("source_data_lock", threading.Lock())
    slot.download_button(
        "Download source data",
        data=lambda: source_data_bytes(result),
        file_name=f'{result["file_stub"]}_plotdata.tsv',
        mime="text/tab-separated-values",
        on_click="ignore",
        width="stretch",
    )


def show_plot_result(result: dict) -> None:
    if result["renderer"] == "Plotly (interactive)":
        webgl_pixel_ratio = result["figure_payload"]["layout"]["meta"]["webgl_pixel_ratio"]
        st.plotly_chart(
            result["figure_payload"],
            width="stretch",
            config={
                "scrollZoom": False,
                "plotGlPixelRatio": webgl_pixel_ratio,
                "displaylogo": False,
                "modeBarButtonsToAdd": ["drawline", "eraseshape"],
            },
        )
        if result.get("plotly_vector_pdf_60pct_width") is None:
            if st.button("Generate figure PDF"):
                try:
                    with st.spinner("Preparing PDF…"):
                        result["plotly_vector_pdf_60pct_width"] = plotly_pdf_bytes(result["figure_payload"])
                except Exception as exc:
                    st.error(
                        "PDF export failed. Use the documented Conda environment, which "
                        "includes Kaleido with bundled Chromium. On Linux, missing browser "
                        "runtime libraries may still need to be installed. "
                        "The interactive plot is unchanged."
                    )
                    with st.expander("PDF export error details"):
                        st.code(f"{type(exc).__name__}: {exc}")
        if result.get("plotly_vector_pdf_60pct_width") is not None:
            st.download_button(
                "Download figure PDF",
                data=result["plotly_vector_pdf_60pct_width"],
                file_name=f'{result["file_stub"]}.pdf',
                mime="application/pdf",
                on_click="ignore",
            )
    else:
        output_format = result["seaborn_output_format"]
        st.download_button(
            f"Download Seaborn plot ({output_format.upper()})",
            data=result["seaborn_download"],
            file_name=f'{result["file_stub"]}.{output_format}',
            mime="application/pdf" if output_format == "pdf" else "image/png",
            type="primary",
            on_click="ignore",
        )


st.set_page_config(page_title="MVViewer", page_icon="🧬", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] {
        padding-left: 1rem;
        padding-top: 1rem;
    }
    .mvviewer-legend {
        display: flex;
        flex-direction: column;
        row-gap: 0.4rem;
        margin: -0.35rem 0 0.8rem 0;
        color: #555;
        font-size: 0.78rem;
        line-height: 1.35;
    }
    .mvviewer-legend-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        column-gap: 1.5rem;
        row-gap: 0.35rem;
    }
    .mvviewer-legend-group,
    .mvviewer-legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.28rem;
        white-space: nowrap;
    }
    .mvviewer-legend-group {
        gap: 0.85rem;
    }
    .mvviewer-legend-symbol {
        font-size: 0.88rem;
        line-height: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("MVViewer")
st.caption("Interactive visualization of CpG methylation and small variant with statistics for long-read sequencing data")
st.markdown(
    """
    <div class="mvviewer-legend" aria-label="Plot legend">
      <div class="mvviewer-legend-row">
        <span class="mvviewer-legend-group"><strong>Region</strong>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:#d9d9d9;opacity:0.3">■</span>selected</span>
        </span>
        <span class="mvviewer-legend-group"><strong>Haplotype</strong>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:pink">■</span>HP1</span>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:lightblue">■</span>HP2</span>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:grey">■</span>untagged</span>
        </span>
        <span class="mvviewer-legend-group"><strong>Strand</strong>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:darkgrey">■</span>forward</span>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:lightgrey">■</span>reverse</span>
        </span>
      </div>
      <div class="mvviewer-legend-row">
        <span class="mvviewer-legend-group"><strong>Modification</strong>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:#B71D48;opacity:0.3">◆</span>5mC</span>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:#E69F00;opacity:0.3">◆</span>5hmC</span>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:#A5B4D7;opacity:0.3">◆</span>unmodified</span>
        </span>
      </div>
      <div class="mvviewer-legend-row">
        <span class="mvviewer-legend-group"><strong>Variant</strong>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:grey">●</span>reference</span>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:#009600">●</span>A</span>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:#0000FF">●</span>C</span>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:#D17105">●</span>G</span>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:#FF0000">●</span>T</span>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:#E7298A">▲</span>insertion</span>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:#00BFC4">▼</span>deletion</span>
          <span class="mvviewer-legend-item"><span class="mvviewer-legend-symbol" style="color:#6A3D9A">★</span>methylation-associated</span>
        </span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Reference genome")
    prepared_references = list_prepared_references()
    if prepared_references:
        selected_reference = st.selectbox(
            "Prepared reference",
            prepared_references,
            format_func=lambda reference: reference.label,
            help="References should be initialized with init.py.",
        )
        prepared_reference = selected_reference.path
    else:
        prepared_reference = None
        st.warning("No prepared references found.")
        st.code("python init.py hg38", language="bash")

    prepared_reference_key = str(prepared_reference.resolve()) if prepared_reference else None
    prepared_reference_version = input_file_version(prepared_reference)
    st.session_state.pop("loaded_cpg_table", None)
    reference_is_loaded = (
        prepared_reference_key is not None
        and st.session_state.get("loaded_reference_path") == prepared_reference_key
        and "loaded_cpg_reference" in st.session_state
        and st.session_state.get("loaded_reference_version") == prepared_reference_version
    )
    if st.button("Load", disabled=prepared_reference is None or reference_is_loaded, width="stretch"):
        try:
            time_startloading = time.time()
            with st.spinner("Opening reference index…"):
                loaded_cpg_reference = open_cpg_reference(prepared_reference)
            st.session_state["loaded_cpg_reference"] = loaded_cpg_reference
            st.session_state["loaded_reference_path"] = prepared_reference_key
            st.session_state["loaded_reference_version"] = prepared_reference_version
            reference_is_loaded = True
            print('Loading time (seconds):',time.time()-time_startloading)
        except Exception as exc:
            st.error(f"Could not load reference genome: {exc}")
    if reference_is_loaded:
        st.caption("Reference loaded")
    elif prepared_reference is not None:
        st.caption("Click **Load** before plotting.")

    region_text = st.text_input(
        "Region",
        "chr19:56837539-56840146",
        help="Accepted formats: chr19:56837539-56840146, chr19 56837539 56840146, or tab-separated columns.",
    )

    st.header("Tracks")
    track_text = st.text_area(
        "Track definitions",
        help="One comma-separated line: BAM, optional VCF, optional sample name; or GTF/GTF.gz, optional track name. Leave the name blank to use the filename.",
        height=210,
        placeholder="examples:\n/data/sample1.bam,/data/sample1.vcf.gz,sample-name-1\n/data/sample2.bam,/data/sample2.vcf.gz\n/data/sample3.bam,,sample-name-3\n/data/sample4.bam,\n/data/annotation.gtf,genes",
        key="track_definitions",
    )
    with st.expander("Settings"):
        modification_cutoff = st.number_input(
            "Base modification cutoff",
            min_value=0.00,
            max_value=1.00,
            value=0.80,
            step=0.05,
            format="%.2f",
            help="Modification calls with a smaller probability, including unmethylated calls, are omitted.",
        )
        max_indel_length = st.number_input(
            "Maximum indel length (bp)",
            min_value=1,
            max_value=10_000,
            value=50,
            step=10,
            help="Larger insertions and deletions are omitted.",
        )
        sort_by = st.radio(
            "Sort by",
            ["methylation", "position", "HP and strand"],
            help=(
                "Methylation sorts by average methylation level of the selected region. "
                "Position sorts by REF/A/T/C/G and modification status. "
                "HP and strand sorts by haplotype and strand. "
            ),
        )
        sort_position_text = ""
        if sort_by == "position":
            sort_position_text = st.text_input(
                "Position",
                placeholder="e.g. 2905234",
                help="Enter a genomic coordinate within the selected region.",
            )
        pvalue_cutoff = st.number_input(
            "P-value cutoff", min_value=0.0, max_value=1.0, value=0.05,
            step=0.01, format="%.4f",
            help="Show stars only for loci with Kruskal–Wallis P-value strictly below this cutoff. Downloads retain all results.",
        )
        use_genomic_coordinates = st.checkbox(
            "Original genomic coordinates",
            value=False,
            help="Checked: Use true base-pair distances on the X-axis. Unchecked: compact spacing between observed positions.",
        )
        webgl_pixel_base = st.number_input(
            "WebGL pixel base",
            min_value=0.1,
            value=plotting.WEBGL_PIXEL_BASE,
            step=0.1, format="%.1f",
            help="Higher value increases figure resolution but also makes figure larger.",
        )
        max_webgl_figure_height = st.number_input(
            "Maximum WebGL figure height",
            min_value=1,
            value=plotting.MAX_WEBGL_FIGURE_HEIGHT,
            step=1000,
            help="WebGL height budget in pixels.",
        )
        renderer = st.radio(
            "Mode",
            ["Plotly (interactive)", "Seaborn (download only)"],
            help="Seaborn is better in making PDF if you have more than ten samples.",
        )
        seaborn_download_format = "PNG"
        if renderer == "Seaborn (download only)":
            seaborn_download_format = st.segmented_control(
                "Seaborn download format", ["PNG", "PDF"], default="PNG",
            )
    view = st.button(
        "Plot",
        type="primary",
        disabled=not track_text.strip(),
        width="stretch",
    )
    plot_download_slot = st.empty()

coordinate_mode = "genomic" if use_genomic_coordinates else "compact"
settings_key = (float(modification_cutoff), int(max_indel_length), sort_by,
                sort_position_text.strip(), float(pvalue_cutoff), renderer,
                seaborn_download_format, coordinate_mode, webgl_pixel_base, int(max_webgl_figure_height))
input_key = (region_text.strip(), track_text.strip(), prepared_reference_key)
if not view:
    cached_result = st.session_state.get("last_plot_result")
    if cached_result is not None:
        show_plot_download(plot_download_slot, cached_result)
        show_plot_result(cached_result)
        st.stop()
    plot_download_slot.download_button(
        "Download source data",
        data=b"",
        disabled=True,
        width="stretch",
    )
    st.info("Select a region, add tracks, and click **Plot**. Load a reference genome for BAM tracks.")
    st.stop()

try:
    region = parse_region(region_text)
    sort_position = None
    sort_methylation_average = sort_by == "methylation"
    if sort_by == "position":
        try:
            sort_position = int(sort_position_text.strip().replace(",", ""))
        except ValueError as exc:
            raise ValueError("Enter a genomic coordinate for position sorting") from exc
        if not region.start + 1 <= sort_position <= region.end:
            raise ValueError("Sort by position must be within the selected region")
    tracks = parse_track_lines(track_text, Path.cwd())
    if not tracks:
        raise ValueError("Add at least one track")
except ValueError as exc:
    st.error(str(exc))
    st.stop()

for track in tracks:
    if not track.alignment_path.is_file():
        st.error(f"Track file does not exist: {track.alignment_path}")
        st.stop()
    if track.variant_path is not None and not track.variant_path.is_file():
        st.error(f"VCF does not exist: {track.variant_path}")
        st.stop()

try:
    plot_status = st.status("Initiating...", expanded=False)
    time_init = time.time()

    def show_current_sample(sample_name: str) -> None:
        plot_status.update(label=f"{sample_name}…", state="running")

    with PLOT_LOCK:
        importlib.invalidate_caches()
        alignment_tracks = [track for track in tracks if not track.is_annotation]
        annotation_tracks = [track for track in tracks if track.is_annotation]
        if alignment_tracks and not reference_is_loaded:
            raise ValueError("Load the selected reference before plotting")
        importlib.reload(data_processing)
        importlib.reload(plotting)
        raw_key = (
            "indexed-cpg-input-v1",
            region.chrom, region.start, region.end,
            input_file_version(prepared_reference),
            tuple((track.name, input_file_version(track.alignment_path),
                   input_file_version(track.variant_path)) for track in tracks),
        )
        raw_cache = st.session_state.get("raw_region_cache")
        if raw_cache is None or raw_cache["key"] != raw_key:
            previous_samples = st.session_state.get("sample_input_cache", {})
            sample_cache, samples, spans = {}, [], []
            reference_version = input_file_version(prepared_reference)
            sample_keys = []
            for track in alignment_tracks:
                sample_key = ("indexed-cpg-v1", region.chrom, region.start, region.end, reference_version,
                              track.name, input_file_version(track.alignment_path),
                              input_file_version(track.variant_path))
                sample_keys.append(sample_key)
            missing = [(key, track) for key, track in zip(sample_keys, alignment_tracks)
                       if key not in previous_samples]
            new_samples = {}
            if missing:
                new_loaded = data_processing.load_region_data(
                    region.chrom, region.start, region.end,
                    [str(track.alignment_path) for _, track in missing],
                    [str(track.variant_path) if track.variant_path else None for _, track in missing],
                    [track.name for _, track in missing], st.session_state["loaded_cpg_reference"],
                    progress_callback=show_current_sample,
                )
                new_samples = {key: {"samples": [sample], "long_start": sample["long_start"],
                                     "long_end": sample["long_end"]}
                               for (key, _), sample in zip(missing, new_loaded["samples"])}
            for sample_key in sample_keys:
                entry = previous_samples.get(sample_key)
                if entry is None:
                    entry = new_samples[sample_key]
                sample_cache[sample_key] = entry
                samples.extend(entry["samples"])
                if entry["samples"][0].get("has_reads", True):
                    spans.append((entry["long_start"], entry["long_end"]))
            loaded = {"chrom": region.chrom, "start": region.start, "end": region.end,
                      "samples": samples,
                      "long_start": min((span[0] for span in spans), default=region.start),
                      "long_end": max((span[1] for span in spans), default=region.end)}
            previous_annotations = st.session_state.get("annotation_input_cache", {})
            annotation_cache = {}
            loaded["annotations"] = {}
            for track in annotation_tracks:
                annotation_key = (input_file_version(track.alignment_path), region.chrom,
                                  loaded["long_start"], loaded["long_end"])
                features = previous_annotations.get(annotation_key)
                if features is None:
                    show_current_sample("Loading " + track.name)
                    features = data_processing.load_gtf_track(
                        track.alignment_path, region.chrom, loaded["long_start"], loaded["long_end"],
                    )
                annotation_cache[annotation_key] = features
                loaded["annotations"][track.name] = features
            loaded["track_order"] = [track.name for track in tracks]
            st.session_state["sample_input_cache"] = sample_cache
            st.session_state["annotation_input_cache"] = annotation_cache
            raw_cache = {"key": raw_key, "data": loaded}
            st.session_state["raw_region_cache"] = raw_cache
            st.session_state.pop("processed_region_cache", None)
        process_key = ("sample-call-tables-v1", raw_key, int(max_indel_length), float(modification_cutoff),
                       sort_position, sort_methylation_average, coordinate_mode)
        processed_cache = st.session_state.get("processed_region_cache")
        if processed_cache is None or processed_cache["key"] != process_key:
            processed_data = data_processing.process_region_data(
                raw_cache["data"], max_indel_length=int(max_indel_length),
                modification_cutoff=float(modification_cutoff), sort_position=sort_position,
                progress_callback=show_current_sample,
                sort_methylation_average=sort_methylation_average,
                coordinate_mode=coordinate_mode,
            )
            processed_cache = {"key": process_key, "data": processed_data}
            st.session_state["processed_region_cache"] = processed_cache
        processed_data = processed_cache["data"]
        if renderer == "Plotly (interactive)":
            figure = plotting.plot_plotly(
                processed_data, progress_callback=show_current_sample,
                pvalue_cutoff=float(pvalue_cutoff),
                webgl_pixel_base=webgl_pixel_base,
                max_webgl_figure_height=int(max_webgl_figure_height),
            )
            figure_payload = as_plotly_payload(figure)
        else:
            plt.close("all")
            figure = plotting.plot_seaborn(
                processed_data, progress_callback=show_current_sample,
                pvalue_cutoff=float(pvalue_cutoff),
            )
            seaborn_download = io.BytesIO()
            seaborn_output_format = seaborn_download_format.lower()
            save_options = {
                "format": seaborn_output_format,
                "bbox_inches": "tight",
            }
            if seaborn_output_format == "png":
                save_options["dpi"] = 100
            figure.savefig(seaborn_download, **save_options)
            seaborn_download.seek(0)
            plt.close(figure)
    plot_status.update(label="Almost there...", state="running")
    safe_chrom = "".join(character if character.isalnum() or character in "._-" else "_" for character in region.chrom)
    result = {
        "input_key": input_key,
        "settings_key": settings_key,
        "renderer": renderer,
        "sample_count": len(tracks),
        "webgl_pixel_base": webgl_pixel_base,
        "max_webgl_figure_height": int(max_webgl_figure_height),
        "figure_payload": (
            figure_payload if renderer == "Plotly (interactive)" else None
        ),
        "seaborn_download": (
            seaborn_download.getvalue()
            if renderer == "Seaborn (download only)"
            else None
        ),
        "seaborn_output_format": (
            seaborn_output_format
            if renderer == "Seaborn (download only)"
            else None
        ),
        "plot_data": processed_data,
        "file_stub": f"MVViewer_{safe_chrom}_{region.start + 1}-{region.end}",
    }
    st.session_state["last_plot_result"] = result
    show_plot_download(plot_download_slot, result)
    show_plot_result(result)
    plot_status.update(label="Done. Time elapsed (seconds): "+str(time.time()-time_init), state="complete")
except Exception as exc:
    if "plot_status" in locals():
        plot_status.update(label="Plotting failed", state="error")
    st.error(f"MVViewer could not plot this region: {exc}")
    st.caption(
        "Please check the BAM/GTF tracks."
    )
