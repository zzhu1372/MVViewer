import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from math import log10
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

from src.data_processing import PlotData


HAPLOTYPE_COLORS = {1: "pink", 2: "lightblue", 0: "grey"}
STRAND_COLORS = {True: "lightgrey", False: "darkgrey"}
IGV_NUCLEOTIDE_COLORS = {
    "A": "#009600",
    "C": "#0000FF",
    "G": "#D17105",
    "T": "#FF0000",
    "N": "#808080",
}
INDEL_COLORS = {"insertion": "#E7298A", "deletion": "#00BFC4"}
MODIFICATION_COLORS = {0: "#A5B4D7", 0.5: "#E69F00", 1: "#B71D48"}
WEBGL_PIXEL_BASE = 2.0
MAX_WEBGL_FIGURE_HEIGHT = 8000
P_VALUE_COLUMN = "methylation_kruskal_pvalue"
GTF_ROW_SPACING = 0.15


def webgl_pixel_ratio(figure_height, webgl_pixel_base=WEBGL_PIXEL_BASE, max_webgl_figure_height=MAX_WEBGL_FIGURE_HEIGHT):
    return min(webgl_pixel_base, max_webgl_figure_height / max(1, figure_height))


def _track_height(data, sample):
    annotations = getattr(data, "annotations", {})
    if sample in annotations:
        features = annotations[sample]
        lanes = int(features["lane"].max()) + 1 if not features.empty else 0
        return max(60, 24 * (max(0, lanes - 1) * GTF_ROW_SPACING + 1) + 20)
    return max(32, 4 * (len(data.reads.get(sample, [])) + 6))


def _pvalue_opacities(pvalues):
    return [max(0.1, min(1.0, -0.5-log10(max(float(p), 1e-10)) / 2))
            for p in pvalues]


def _variant_pvalues(data, sample, pvalue_cutoff=0.05):
    variants = _sample_calls(data, sample, "variant")
    if P_VALUE_COLUMN not in variants:
        return pd.DataFrame(columns=["pos", "pos2id", "refallele", P_VALUE_COLUMN,
                                     "least_methylated_allele", "most_methylated_allele"])
    values = variants.loc[variants[P_VALUE_COLUMN] < pvalue_cutoff].drop_duplicates(
        ["chr", "pos", "refallele"]
    ).copy()
    for column in ("least_methylated_allele", "most_methylated_allele"):
        if column not in values:
            values[column] = ""
    return values


def _igv_nucleotide_palette(alleles):
    return {
        allele: IGV_NUCLEOTIDE_COLORS.get(
            str(allele).upper(), IGV_NUCLEOTIDE_COLORS["N"]
        )
        for allele in alleles
    }


def _selected_span(data):
    if getattr(data, "region_xrange", None) is not None:
        left, right = data.region_xrange
        return left - 0.5, right + 0.5
    position_ids = (
        data.position_table
        .set_index("pos")["pos2id"]
    )
    closest_start = min(
        position for position in position_ids.index if position >= data.start
    )
    closest_end = max(
        position for position in position_ids.index if position <= data.end
    )
    return (
        position_ids[closest_start] - 0.5,
        position_ids[closest_end] + 0.5,
    )


def _sample_calls(data, sample, call_type):
    frame = data.calls.get(sample)
    if frame is None:
        return pd.DataFrame(columns=["pos", "pos2id", "refallele"])
    return frame.loc[frame["call_type"] == call_type]


def _position_table(data):
    return data.position_table


def _view_xrange(data):
    if getattr(data, "view_xrange", None) is not None:
        return data.view_xrange
    positions = _position_table(data)
    return positions["pos2id"].min(), positions["pos2id"].max()


def _annotation_models(data, sample):
    features = getattr(data, "annotations", {}).get(sample)
    if features is None or features.empty:
        return []
    positions = _position_table(data)

    def x(position):
        return float(np.interp(position, positions["pos"], positions["pos2id"]))

    models = []
    view_start, view_end = getattr(data, "annotation_region", None) or (data.start, data.end)
    for (_, transcript), group in features.groupby(["gene_id", "transcript_id"], sort=False):
        first = group.iloc[0]
        left = max(view_start + 1, int(group["pos"].min()))
        right = min(view_end, int(group["end"].max()))
        blocks = []
        for feature in group.itertuples():
            if feature.feature not in {"exon", "CDS", "UTR", "five_prime_utr", "three_prime_utr"}:
                continue
            start, end = max(left, feature.pos), min(right, feature.end)
            if end >= start:
                blocks.append((x(start), x(end), feature.feature, int(feature.pos), int(feature.end)))
        models.append({"x0": x(left), "x1": x(right), "lane": first["lane"],
                       "gene": first["gene_name"], "transcript": transcript,
                       "strand": first["strand"], "blocks": blocks,
                       "start": int(group["pos"].min()), "end": int(group["end"].max())})
    return models


def _plot_gtf_plotly(figure, data, sample, row):
    models = _annotation_models(data, sample)
    block_padding = 0.5 if getattr(data, "coordinate_mode", "compact") == "genomic" else 0.08
    for model in models:
        y, x0, x1 = model["lane"] * GTF_ROW_SPACING, model["x0"], model["x1"]
        info = (f"{model['gene']}<br>transcript: {model['transcript']}"
                f"<br>{data.chrom}:{model['start']}-{model['end']}<br>strand: {model['strand']}")
        figure.add_trace(go.Scatter(
            x=[x0, x1], y=[y, y], mode="lines", line={"color": "#4C78A8", "width": 1},
            hovertemplate=info + "<extra></extra>", showlegend=False,
        ), row=row, col=2)
        for left, right, feature, start, end in sorted(model["blocks"], key=lambda block: block[2] == "CDS"):
            half_height = 0.05 if feature == "CDS" else 0.03
            figure.add_trace(go.Scatter(
                x=[left - block_padding, right + block_padding, right + block_padding,
                   left - block_padding, left - block_padding],
                y=[y - half_height, y - half_height, y + half_height, y + half_height, y - half_height],
                mode="lines", line={"width": 0}, fill="toself",
                fillcolor="#2C4A6B" if feature == "CDS" else "#4C78A8",
                hoveron="fills", hovertemplate=info + f"<br>{feature}: {start}-{end}<extra></extra>",
                showlegend=False,
            ), row=row, col=2)
        if model["strand"] in {"+", "-"}:
            figure.add_trace(go.Scatter(
                x=[(x0 + x1) / 2], y=[y], mode="markers",
                marker={"symbol": "triangle-right" if model["strand"] == "+" else "triangle-left",
                        "size": 3.5, "color": "white", "line": {"color": "#4C78A8", "width": 0.5}},
                hovertemplate=info + "<extra></extra>", showlegend=False,
            ), row=row, col=2)
    figure.update_xaxes(visible=False, fixedrange=True, row=row, col=1)
    figure.update_yaxes(range=[-0.6, max((model["lane"] for model in models), default=0) * GTF_ROW_SPACING + 0.1], row=row)


def _plot_gtf_seaborn(panel, data, sample):
    models = _annotation_models(data, sample)
    block_padding = 0.5 if getattr(data, "coordinate_mode", "compact") == "genomic" else 0.08
    for model in models:
        y, x0, x1 = model["lane"] * GTF_ROW_SPACING, model["x0"], model["x1"]
        panel.plot([x0, x1], [y, y], color="#4C78A8", linewidth=0.7)
        for left, right, feature, _, _ in sorted(model["blocks"], key=lambda block: block[2] == "CDS"):
            height = 0.1 if feature == "CDS" else 0.06
            panel.add_patch(plt.Rectangle((left - block_padding, y - height / 2),
                            right - left + 2 * block_padding, height,
                            color="#2C4A6B" if feature == "CDS" else "#4C78A8", linewidth=0))
        if model["strand"] in {"+", "-"}:
            panel.scatter([(x0 + x1) / 2], [y], marker=">" if model["strand"] == "+" else "<",
                          s=4.5, facecolor="white", edgecolor="#4C78A8", linewidth=0.3, zorder=3)
    panel.set_ylim(-0.6, max((model["lane"] for model in models), default=0) * GTF_ROW_SPACING + 0.1)
    panel.set_title(sample, pad=4)


def _categorical_marker_colors(categories, palette, fallback="grey"):
    colors = list(dict.fromkeys([*palette.values(), fallback]))
    color_ids = {color: index for index, color in enumerate(colors)}
    category_ids = {category: color_ids[color] for category, color in palette.items()}
    codes = pd.Series(categories, copy=False).map(category_ids).fillna(
        color_ids[fallback]
    ).to_numpy(dtype=np.int32)
    if not len(codes):
        return {"color": fallback}
    if np.all(codes == codes[0]):
        return {"color": colors[int(codes[0])]}
    count = len(colors)
    colorscale = [stop for index, color in enumerate(colors)
                  for stop in [(index / count, color), ((index + 1) / count, color)]]
    return {"color": codes, "colorscale": colorscale,
            "cmin": -0.5, "cmax": count - 0.5, "cauto": False, "showscale": False}


def _add_categorical_markers(
    figure,
    row,
    x,
    y,
    categories,
    palette,
    size,
    symbol,
    hover_name,
    read_names,
    loci=None,
    column=1,
    reference_alleles=None,
):
    frame = pd.DataFrame(
        {
            "x": list(x),
            "y": list(y),
            "category": list(categories),
            "read_name": list(read_names),
        }
    )
    custom_columns = ["category", "read_name"]
    hovertemplate = (
        f"{hover_name}: %{{customdata[0]}}<br>"
        "read: %{customdata[1]}<extra></extra>"
    )
    if loci is not None:
        frame["locus"] = list(loci)
        custom_columns.append("locus")
        hovertemplate = (
            "position %{customdata[2]}<br>"
            f"{hover_name}: %{{customdata[0]}}<br>"
            "read: %{customdata[1]}<extra></extra>"
        )
    if reference_alleles is not None:
        frame["reference_allele"] = list(reference_alleles)
        reference_index = len(custom_columns)
        custom_columns.append("reference_allele")
        hovertemplate = hovertemplate.replace(
            "read:", f"reference allele: %{{customdata[{reference_index}]}}<br>read:"
        )
    figure.add_trace(
        go.Scattergl(
            x=frame["x"],
            y=frame["y"],
            mode="markers",
            marker={
                "size": size,
                "symbol": symbol,
                **_categorical_marker_colors(frame["category"], palette),
            },
            customdata=frame[custom_columns].to_numpy(),
            hovertemplate=hovertemplate,
            showlegend=False,
        ),
        row=row,
        col=column,
    )


def plot_plotly(data: PlotData, progress_callback=None, pvalue_cutoff=0.05,
                webgl_pixel_base=WEBGL_PIXEL_BASE, max_webgl_figure_height=MAX_WEBGL_FIGURE_HEIGHT):
    annotations = getattr(data, "annotations", {})
    read_counts = {sample: len(data.reads.get(sample, [])) for sample in data.names}
    last_read_rows = {
        sample: read_counts[sample] - 1 for sample in data.names
    }
    first_position, number_of_positions = _view_xrange(data)
    read_heights = [_track_height(data, sample) for sample in data.names]
    row_heights, specs, titles = [], [], []
    height_scale = min(1, max_webgl_figure_height / (2 * sum(read_heights)))
    panel_gap_pixels = (22 + len(data.names)) * height_scale
    for index, sample in enumerate(data.names):
        row_heights.append(read_heights[index]*height_scale)
        specs.append([{}, {}])
        titles.extend([None, sample])
        if index < len(data.names) - 1:
            row_heights.append(panel_gap_pixels)
            specs.append([None, None])
    plot_area_height = sum(row_heights)
    pvalues = {sample: _variant_pvalues(data, sample, pvalue_cutoff)
               for sample in data.names}

    figure = make_subplots(
        rows=len(row_heights),
        cols=2,
        shared_xaxes="columns",
        shared_yaxes="rows",
        row_heights=row_heights,
        column_widths=[0.025, 0.975],
        horizontal_spacing=0.002,
        vertical_spacing=0,
        specs=specs,
        subplot_titles=titles,
    )
    for title in figure.layout.annotations:
        title.update(yanchor="bottom", yshift=2*height_scale, font={"size": 11})

    selected_x0, selected_x1 = _selected_span(data)
    for index, sample in enumerate(data.names):
        if progress_callback is not None:
            progress_callback("Plotting " + sample)
        row = 2 * index + 1
        if sample in annotations:
            _plot_gtf_plotly(figure, data, sample, row)
            continue
        read_table = data.reads[sample]
        figure.add_vrect(
            x0=selected_x0,
            x1=selected_x1,
            fillcolor="#d9d9d9",
            opacity=0.3,
            layer="above",
            line_width=0,
            row=row,
            col=2,
            exclude_empty_subplots=False,
        )
        _add_categorical_markers(
            figure, row, [0] * len(read_table), read_table[3], read_table[0],
            HAPLOTYPE_COLORS, 5, "square", "haplotype", read_table.index,
        )
        _add_categorical_markers(
            figure, row, [1] * len(read_table), read_table[3], read_table[1],
            STRAND_COLORS, 5, "square", "reverse strand", read_table.index,
        )

        modifications = _sample_calls(
            data, sample, "modification"
        ).loc[lambda frame: frame["mod"].notna()]
        figure.add_trace(
            go.Scattergl(
                x=modifications["pos2id"],
                y=modifications["rname2id"],
                mode="markers",
                marker={
                    "size": 3.5,
                    "symbol": "diamond-tall",
                    **_categorical_marker_colors(modifications["mod"], MODIFICATION_COLORS, "#808080"),
                    "showscale": False,
                },
                opacity=0.3,
                customdata=modifications[["pos", "mod", "rname"]].to_numpy(),
                hovertemplate=(
                    "position %{customdata[0]}<br>"
                    "modification %{customdata[1]}<br>"
                    "read: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=2,
        )

        variants = _sample_calls(data, sample, "variant")
        reference_calls = variants.loc[
            variants["base"] == variants["refallele"]
        ]
        figure.add_trace(
            go.Scattergl(
                x=reference_calls["pos2id"],
                y=reference_calls["rname2id"],
                mode="markers",
                marker={"size": 4.5, "symbol": "circle", "color": "grey"},
                customdata=reference_calls[
                    ["pos", "base", "refallele", "rname"]
                ].to_numpy(),
                hovertemplate=(
                    "position %{customdata[0]}<br>"
                    "base %{customdata[1]} (reference %{customdata[2]})<br>"
                    "read: %{customdata[3]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=2,
        )

        nonreference_calls = variants.loc[
            variants["variant_type"] != "reference"
        ]
        variant_symbols = {
            "substitution": ("circle", "non-reference allele"),
            "insertion": ("triangle-up", "insertion allele"),
            "deletion": ("triangle-down", "deletion allele"),
        }
        for variant_type, (symbol, hover_name) in variant_symbols.items():
            calls = nonreference_calls.loc[
                nonreference_calls["variant_type"] == variant_type
            ]
            alleles = pd.unique(calls["base"])
            if variant_type == "substitution":
                palette = _igv_nucleotide_palette(alleles)
            else:
                palette = {
                    allele: INDEL_COLORS[variant_type] for allele in alleles
                }
            _add_categorical_markers(
                figure,
                row,
                calls["pos2id"],
                calls["rname2id"],
                calls["base"],
                palette,
                4.5,
                symbol,
                hover_name,
                calls["rname"],
                loci=calls["pos"],
                column=2,
                reference_alleles=calls["refallele"],
            )

        figure.update_xaxes(
            range=[-0.75, 1.75],
            fixedrange=True,
            visible=False,
            row=row,
            col=1,
        )
        figure.update_yaxes(
            range=[-5, last_read_rows[sample] + 2], row=row
        )
        values = pvalues[sample]
        figure.add_trace(
            go.Scattergl(
                x=values["pos2id"], y=[-3] * len(values), mode="markers",
                marker={"size": 6, "symbol": "star", "color": "#6A3D9A",
                        "opacity": _pvalue_opacities(values[P_VALUE_COLUMN])},
                customdata=values[["pos", "least_methylated_allele",
                                   "most_methylated_allele", P_VALUE_COLUMN]].to_numpy(),
                hovertemplate=(f"locus {data.chrom}:%{{customdata[0]}}"
                               "<br>most unmet allele: %{customdata[1]}"
                               "<br>most met allele: %{customdata[2]}"
                               "<br>Kruskal–Wallis p=%{customdata[3]:.4g}<extra></extra>"),
                showlegend=False,
            ), row=row, col=2,
        )

    position_table = _position_table(data)
    position_table = position_table.loc[
        position_table["pos2id"].between(first_position, number_of_positions)
    ]
    tick_positions = position_table["pos2id"].tolist()
    tick_labels = position_table["pos"].tolist()
    stride = max(1, len(tick_positions) // 15)
    genomic_padding = max(1, (number_of_positions - first_position) * 0.01)
    figure.update_xaxes(
        range=[first_position - genomic_padding, number_of_positions + genomic_padding], col=2
    )
    tick_options = ({"tickmode": "auto", "tickformat": ",d"}
                    if getattr(data, "coordinate_mode", "compact") == "genomic"
                    else {"tickmode": "array", "tickvals": tick_positions[::stride],
                          "ticktext": tick_labels[::stride]})
    figure.update_xaxes(
        **tick_options,
        title_text=f"{data.chrom} position",
        row=2 * (len(data.names) - 1) + 1,
        col=2,
    )
    figure.update_xaxes(showgrid=False, zeroline=False)
    figure.update_yaxes(
        fixedrange=True,
        visible=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="",
    )
    requested_height = max(150, int(plot_area_height + 97))
    figure.update_layout(
        height=requested_height,
        width=1800,
        margin={"l": 0, "r": 0, "t": 42, "b": 55},
        template="plotly_white",
        showlegend=False,
        hovermode="closest",
        dragmode="zoom",
        uirevision=f"{data.chrom}:{data.start}-{data.end}",
        meta={
            "requested_height": requested_height,
            "webgl_height_capped": False,
            "webgl_pixel_ratio": webgl_pixel_ratio(requested_height, webgl_pixel_base, max_webgl_figure_height),
            "max_webgl_figure_height": max_webgl_figure_height,
        },
    )
    return figure


def _add_seaborn_legends(figure):
    def marker(label, symbol, color, size=6, alpha=1):
        return Line2D([], [], label=label, linestyle="none", marker=symbol,
                      markerfacecolor=color, markeredgewidth=0,
                      markersize=size, alpha=alpha)

    groups = [
        [Patch(facecolor="#eeeeee", edgecolor="none", label="selected region"),
         *[marker(label, "s", HAPLOTYPE_COLORS[hp])
           for hp, label in [(1, "HP1"), (2, "HP2"), (0, "untagged")]],
         marker("forward", "s", STRAND_COLORS[False]),
         marker("reverse", "s", STRAND_COLORS[True])],
        [marker(label, "d", MODIFICATION_COLORS[mod], size=5, alpha=0.3)
         for mod, label in [(1, "5mC"), (0.5, "5hmC"), (0, "unmodified")]],
        [marker("reference", "o", "grey"),
         *[marker(base, "o", IGV_NUCLEOTIDE_COLORS[base]) for base in "ACGT"],
         marker("insertion", "^", INDEL_COLORS["insertion"]),
         marker("deletion", "v", INDEL_COLORS["deletion"]),
         marker("methylation-associated", "*", "#6A3D9A", size=8)],
    ]
    width, height = figure.get_size_inches()
    height += 1.0
    figure.set_size_inches(width, height)
    figure.tight_layout(h_pad=1.2, rect=(0, 1.0 / height, 1, 1))
    for handles, bottom in zip(groups, [0.66, 0.38, 0.10]):
        figure.legend(handles=handles, loc="lower center",
                      bbox_to_anchor=(0.5, bottom / height),
                      ncol=len(handles), frameon=False, fontsize=9,
                      borderaxespad=0, columnspacing=1.8, handletextpad=0.5)


def plot_seaborn(data: PlotData, progress_callback=None, pvalue_cutoff=0.05):
    annotations = getattr(data, "annotations", {})
    last_read_rows = {
        sample: (max(2, 3 * (annotations[sample]["lane"].max() * GTF_ROW_SPACING + 1))
                 if sample in annotations and not annotations[sample].empty
                 else data.reads[sample][3].max()
                 if sample in data.reads and not data.reads[sample].empty else 2)
        for sample in data.names
    }
    first_position, number_of_positions = _view_xrange(data)
    position_span = max(1, number_of_positions - first_position)
    figure, axes = plt.subplots(
        len(data.names),
        1,
        figsize=[
            15,
            max(1, 0.1 * sum(last_read_rows.values())) + 0.5 * len(data.names),
        ],
        height_ratios=[max(7, last_read_rows[sample] + 7) for sample in data.names],
        sharex=True,
        squeeze=False,
    )
    panels = axes[:, 0]
    pvalues = {sample: _variant_pvalues(data, sample, pvalue_cutoff)
               for sample in data.names}
    selected_x0, selected_x1 = _selected_span(data)

    for index, sample in enumerate(data.names):
        if progress_callback is not None:
            progress_callback("Plotting " + sample)
        panel = panels[index]
        if sample in annotations:
            _plot_gtf_seaborn(panel, data, sample)
            continue
        panel.axvspan(
            selected_x0,
            selected_x1,
            facecolor="#eeeeee",
            edgecolor="none",
            zorder=1,
        )
        read_table = data.reads[sample]
        if read_table.empty:
            panel.set_ylim(-5, 2)
            panel.set_title(sample, pad=4)
            continue
        sns.scatterplot(
            x=first_position - 0.02 * position_span,
            y=read_table[3],
            hue=read_table[0],
            ax=panel,
            s=20,
            palette=HAPLOTYPE_COLORS,
            marker="s",
            linewidth=0,
            legend=False,
        )
        sns.scatterplot(
            x=first_position - 0.01 * position_span,
            y=read_table[3],
            hue=read_table[1],
            ax=panel,
            s=20,
            palette=STRAND_COLORS,
            marker="s",
            linewidth=0,
            legend=False,
        )
        modifications = _sample_calls(data, sample, "modification")
        if not modifications.empty:
            sns.scatterplot(
                x=modifications["pos2id"],
                y=modifications["rname2id"],
                hue=modifications["mod"],
                ax=panel,
                s=3,
                palette=MODIFICATION_COLORS,
                marker="d",
                linewidth=0,
                alpha=0.3,
                legend=False,
            )

        variants = _sample_calls(data, sample, "variant")
        reference_calls = variants.loc[
            variants["base"] == variants["refallele"]
        ]
        sns.scatterplot(
            x=reference_calls["pos2id"],
            y=reference_calls["rname2id"],
            ax=panel,
            s=20,
            marker="o",
            linewidth=0,
            legend=False,
            color="grey",
        )
        nonreference_calls = variants.loc[
            variants["variant_type"] != "reference"
        ]
        variant_markers = {
            "substitution": "o",
            "insertion": "^",
            "deletion": "v",
        }
        for variant_type, marker in variant_markers.items():
            calls = nonreference_calls.loc[
                nonreference_calls["variant_type"] == variant_type
            ]
            if calls.empty:
                continue
            if variant_type == "substitution":
                palette = _igv_nucleotide_palette(pd.unique(calls["base"]))
            else:
                palette = {
                    allele: INDEL_COLORS[variant_type]
                    for allele in pd.unique(calls["base"])
                }
            sns.scatterplot(
                x=calls["pos2id"],
                y=calls["rname2id"],
                hue=calls["base"],
                ax=panel,
                s=20,
                marker=marker,
                linewidth=0,
                legend=False,
                palette=palette,
            )

        panel.set_ylim(-5, last_read_rows[sample] + 1.5)
        panel.set_title(sample, pad=4)
        values = pvalues[sample]
        panel.scatter(values["pos2id"], [-3] * len(values), s=50,
                      marker="*", c=[to_rgba("#6A3D9A", alpha) for alpha in
                                     _pvalue_opacities(values[P_VALUE_COLUMN])], zorder=3)

    for panel in panels:
        panel.grid(False)
        panel.yaxis.set_visible(False)
        panel.xaxis.set_visible(False)
        for spine in panel.spines.values():
            spine.set_visible(False)
    panels[-1].set_xlim(first_position - max(1, 0.03 * position_span),
                        number_of_positions + max(1, 0.01 * position_span))
    _add_seaborn_legends(figure)
    return figure
