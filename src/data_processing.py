from dataclasses import dataclass, field
import gzip
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pysam
from scipy.stats import kruskal


DEFAULT_MAX_INDEL_LENGTH = 50
DEFAULT_MODIFICATION_CUTOFF = 0.8

CALL_COLUMNS = [
    "sample", "call_type", "chr", "pos", "pos2id", "rname", "hp", "rev",
    "rname2id", "mod", "ml", "base", "refallele", "variant_type",
    "methylation_average", "methylation_kruskal_pvalue",
    "least_methylated_allele", "most_methylated_allele",
]
ANNOTATION_COLUMNS = ["end", "end_pos2id", "feature", "strand", "gene_id",
                      "gene_name", "transcript_id", "lane"]


@dataclass(frozen=True)
class PlotData:
    chrom: str
    start: int
    end: int
    names: list[str]
    calls: dict[str, pd.DataFrame]
    reads: dict[str, pd.DataFrame]
    annotations: dict[str, pd.DataFrame] = field(default_factory=dict)
    region_xrange: tuple[float, float] | None = None
    annotation_region: tuple[int, int] | None = None
    coordinate_mode: str = "compact"
    view_xrange: tuple[float, float] | None = None
    position_table: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["pos", "pos2id"]))

    def to_source_dataframe(self) -> pd.DataFrame:
        frames = [frame.assign(sample=name) for name, frame in self.calls.items()]
        frames.extend(frame.assign(sample=name, call_type="annotation")
                      for name, frame in self.annotations.items())
        columns = CALL_COLUMNS + (ANNOTATION_COLUMNS if self.annotations else [])
        combined = (pd.concat(frames, ignore_index=True, sort=False)
                    if frames else pd.DataFrame()).reindex(columns=columns)
        return combined


def load_gtf_track(path, chrom, start, end):
    columns = ["chr", "pos", "end", "feature", "strand", "gene_id",
               "gene_name", "transcript_id", "lane"]
    records = []
    path = Path(path)
    indexed = Path(str(path) + ".tbi").is_file() or Path(str(path) + ".csi").is_file()
    if indexed:
        index_path = str(path) + (".tbi" if Path(str(path) + ".tbi").is_file() else ".csi")
        handle = pysam.TabixFile(str(path), index=index_path)
        lines = handle.fetch(chrom, start, end) if chrom in handle.contigs else []
    else:
        handle = gzip.open(path, "rt") if path.name.lower().endswith(".gz") else path.open()
        lines = handle
    try:
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"Invalid GTF line in {path.name}: expected 9 tab-separated columns")
            if fields[0] != chrom or fields[2] not in {"gene", "transcript", "exon", "CDS", "UTR", "five_prime_utr", "three_prime_utr"}:
                continue
            left, right = int(fields[3]), int(fields[4])
            if left < 1 or right < left:
                raise ValueError(f"Invalid GTF coordinates in {path.name}")
            if right <= start or left > end:
                continue
            attributes = dict(re.findall(r'(\w+)\s+"([^"]*)"', fields[8]))
            gene = attributes.get("gene_id", attributes.get("gene_name", "unknown"))
            records.append({
                "chr": chrom, "pos": left, "end": right, "feature": fields[2],
                "strand": fields[6], "gene_id": gene,
                "gene_name": attributes.get("gene_name", gene),
                "transcript_id": attributes.get("transcript_id", ""),
            })
    finally:
        handle.close()
    features = pd.DataFrame(records, columns=columns[:-1])
    if features.empty:
        return pd.DataFrame(columns=columns)
    transcript_genes = set(features.loc[features["feature"] != "gene", "gene_id"])
    features = features.loc[~((features["feature"] == "gene")
                              & features["gene_id"].isin(transcript_genes))].copy()
    features["transcript_id"] = features["transcript_id"].where(
        features["transcript_id"] != "", features["gene_id"]
    )
    spans = features.groupby(["gene_id", "transcript_id"], sort=False).agg(
        left=("pos", "min"), right=("end", "max")
    ).sort_values(["left", "right"], kind="stable")
    lane_ends, lanes = [], {}
    for identity, span in spans.iterrows():
        left, right = max(start + 1, span.left), min(end, span.right)
        lane = next((i for i, previous_end in enumerate(lane_ends)
                     if previous_end < left), len(lane_ends))
        if lane == len(lane_ends):
            lane_ends.append(right)
        else:
            lane_ends[lane] = right
        lanes[identity] = lane
    features["lane"] = [lanes[(row.gene_id, row.transcript_id)]
                        for row in features.itertuples()]
    return features.reset_index(drop=True)


def _classify_modifications(readmods, modification_cutoff):
    confident = readmods["ml"] >= modification_cutoff
    methylated = readmods.loc[
        (readmods["mod"] == "m") & confident
    ].copy()
    methylated["mod"] = 1

    hydroxymethylated = readmods.loc[
        (readmods["mod"] == "h") & confident
    ].copy()
    hydroxymethylated["mod"] = 0.5

    unmodified = (
        readmods.groupby(
            ["chr", "pos", "rname", "hp"], as_index=False, sort=False
        )["ml"].sum()
    )
    unmodified = unmodified.loc[
        unmodified["ml"] <= (1 - modification_cutoff)
    ].copy()
    unmodified["mod"] = 0

    return pd.concat(
        [methylated, hydroxymethylated, unmodified],
        ignore_index=True,
        sort=False,
    )


def _supported_alts(
    reference,
    alternate_alleles,
    max_indel_length=DEFAULT_MAX_INDEL_LENGTH,
):
    reference = reference.upper()
    supported = []
    for allele in alternate_alleles or ():
        if not allele or allele == "*" or allele.startswith("<"):
            continue
        allele = allele.upper()
        if (
            len(allele) == len(reference)
            or abs(len(allele) - len(reference)) <= max_indel_length
        ):
            supported.append(allele)
    return tuple(supported)


def _read_allele_at_variant(
    query_sequence,
    reference_positions,
    query_position,
    reference,
    has_indel,
):
    if not query_sequence:
        return None
    anchor_position = reference_positions[query_position]
    if anchor_position is None:
        return None
    end_position = anchor_position + len(reference)
    observed = []
    for index in range(query_position, len(reference_positions)):
        reference_position = reference_positions[index]
        if reference_position is not None and reference_position >= end_position:
            return "".join(observed)
        observed.append(query_sequence[index].upper())
        if not has_indel and reference_position == end_position - 1:
            return "".join(observed)
    return None


def _variant_type(reference, observed):
    reference = reference.upper()
    observed = observed.upper()
    if observed == reference:
        return "reference"
    if len(observed) > len(reference):
        return "insertion"
    if len(observed) < len(reference):
        return "deletion"
    return "substitution"


def _is_primary_read(read):
    return not (
        read.is_unmapped or read.is_secondary or read.is_supplementary
    )


def _annotate_methylation_kruskal(variants):
    variants = variants.copy()
    variants["methylation_kruskal_pvalue"] = float("nan")
    variants["least_methylated_allele"] = pd.Series(index=variants.index, dtype="object")
    variants["most_methylated_allele"] = pd.Series(index=variants.index, dtype="object")
    for _, locus in variants.groupby(["chr", "pos", "refallele"], sort=False):
        observations = locus.dropna(subset=["rname", "base", "methylation_average"])
        allele_counts = observations.groupby("rname")["base"].nunique()
        observations = observations.loc[
            observations["rname"].isin(allele_counts.index[allele_counts == 1])
        ].drop_duplicates("rname")
        groups = [
            group["methylation_average"].to_numpy(dtype=float)
            for _, group in observations.groupby("base", sort=False)
        ]
        if len(groups) < 2 or len(observations) <= len(groups):
            continue
        if observations["methylation_average"].nunique() == 1:
            pvalue = 1.0
        else:
            pvalue = float(kruskal(*groups).pvalue)
        variants.loc[locus.index, "methylation_kruskal_pvalue"] = pvalue
        allele_means = observations.groupby("base")["methylation_average"].mean()
        for column, target in [("least_methylated_allele", allele_means.min()),
                               ("most_methylated_allele", allele_means.max())]:
            variants.loc[locus.index, column] = ", ".join(
                sorted(allele_means.index[allele_means == target].astype(str))
            )
    return variants


def _ordered_read_table(
    readinfo, pileup, readmods, sort_bases, sort_position, methylation_region=None,
    sort_methylation_average=True, methylation_averages=None,
):
    table = pd.DataFrame.from_dict(readinfo, orient="index", columns=[0, 1, 2])
    if methylation_region is not None:
        start, end = methylation_region  # zero-based, half-open
        if methylation_averages is None:
            region_calls = readmods.loc[
                (readmods["pos"] > start) & (readmods["pos"] <= end)
            ].copy()
            region_calls["_methylated"] = region_calls["mod"].eq(1)
            per_cpg = region_calls.groupby(["rname", "pos"], sort=False)["_methylated"].max()
            averages = per_cpg.groupby(level="rname", sort=False).mean()
        else:
            averages = methylation_averages
        table["methylation_average"] = table.index.map(averages)
    if methylation_region is not None and sort_methylation_average:
        table = table.sort_values(
            ["methylation_average", 0, 1, 2],
            ascending=[False, True, True, True],
            na_position="last",
            kind="mergesort",
        )
    elif sort_position is None:
        table = table.sort_values([0, 1, 2], kind="mergesort")
    elif not table.empty:
        observed = {name: str(base).upper() for name, base in sort_bases.items()}
        calls = pileup.loc[pileup["pos"] == sort_position]
        observed.update(
            {row.rname: str(row.base).upper() for row in calls.itertuples(index=False)}
        )
        references = [
            str(reference).upper()
            for reference in calls["refallele"].dropna().tolist()
        ]
        reference = references[0] if references else None
        if reference is None:
            nucleotide_order = ("A", "T", "C", "G")
            counts = {
                base: sum(value == base for value in observed.values())
                for base in nucleotide_order
            }
            if any(counts.values()):
                reference = max(
                    nucleotide_order,
                    key=lambda base: (counts[base], -nucleotide_order.index(base)),
                )

        allele_order = {"A": 1, "T": 2, "C": 3, "G": 4}
        table["_allele_sort"] = [
            0 if reference is not None and observed.get(name) == reference
            else allele_order.get(observed.get(name), 5)
            for name in table.index
        ]
        modification_order = {1: 0, 0.5: 1, 0: 2}
        modification_calls = readmods.loc[readmods["pos"] == sort_position]
        modification_ranks = {
            name: min(modification_order.get(value, 3) for value in group["mod"])
            for name, group in modification_calls.groupby("rname", sort=False)
        }
        table["_modification_sort"] = [
            modification_ranks.get(name, 3) for name in table.index
        ]
        table = table.sort_values(
            ["_allele_sort", "_modification_sort", 0, 1, 2],
            kind="mergesort",
        ).drop(columns=["_allele_sort", "_modification_sort"])

    table[3] = range(len(table))
    return table


def load_region_data(chrom, start, end, bams, vcfs, names, cpg_table,
                     progress_callback=None):
    regional_reference = hasattr(cpg_table, "fetch_positions")
    cpgspos = None if regional_reference else cpg_table.loc[cpg_table[0] == chrom]
    samples = []
    spans = []
    for index, name in enumerate(names):
        if progress_callback:
            progress_callback("Loading " + name)
        with pysam.AlignmentFile(bams[index], "rb") as bam:
            alignments = ([read for read in bam.fetch(chrom, start, end)
                           if _is_primary_read(read)] if chrom in bam.references else [])
        long_start = min((read.reference_start for read in alignments), default=start)
        long_end = max((read.reference_end for read in alignments), default=end)
        if alignments:
            spans.append((long_start, long_end))
        records = []
        if vcfs[index] and alignments:
            with pysam.VariantFile(vcfs[index]) as variants:
                region_variants = variants.fetch(chrom, long_start, long_end) if chrom in variants.header.contigs else []
                for record in region_variants:
                    reference = record.ref.upper()
                    alts = _supported_alts(reference, record.alts, float("inf"))
                    if alts:
                        records.append((record.pos, reference, alts))
        if not alignments:
            positions = np.empty(0, dtype=np.int64)
        elif regional_reference:
            positions = cpg_table.fetch_positions(chrom, long_start, long_end)
        else:
            positions = cpgspos.loc[
                (cpgspos[1] >= long_start) & (cpgspos[1] <= long_end), 1
            ]
        cpgsubset = set(positions) | set(positions - 1)
        readinfo, retained_reads = {}, []
        query_chunks, probability_chunks = [], []
        modification_columns = {"pos": [], "mod": [], "rname": [], "hp": [], "rev": []}
        for read in alignments:
            modified_bases = read.modified_bases
            if modified_bases is None:
                continue
            hp = read.get_tag("HP") if read.has_tag("HP") else 0
            readinfo[read.query_name] = [hp, read.is_reverse, read.reference_start]
            retained_reads.append(read)
            reference_positions = read.get_reference_positions(full_length=True)
            for key, values in modified_bases.items():
                if not values:
                    continue
                values_array = np.asarray(values, dtype=np.int64)
                query_chunks.append(values_array[:, 0])
                probability_chunks.append(values_array[:, 1])
                modification_columns["pos"].extend(
                    reference_positions[query_position] for query_position in values_array[:, 0]
                )
                count = len(values)
                modification_columns["mod"].extend([key[-1]] * count)
                modification_columns["rname"].extend([read.query_name] * count)
                modification_columns["hp"].extend([hp] * count)
                modification_columns["rev"].extend([read.is_reverse] * count)
        if query_chunks:
            probabilities = np.concatenate(probability_chunks)
            raw_modifications = pd.DataFrame({
                0: np.concatenate(query_chunks), 1: probabilities, "chr": chrom,
                **modification_columns,
            })
            raw_modifications["pos"] = raw_modifications["pos"] + 1
            raw_modifications["ml"] = (probabilities + 0.5) / 256
        else:
            raw_modifications = pd.DataFrame(
                columns=["chr", "pos", "mod", "rname", "hp", "rev", "ml"]
            )
        raw_modifications = raw_modifications.loc[
            (raw_modifications["pos"] - 1).isin(cpgsubset)
        ].reset_index(drop=True)
        samples.append({
            "name": name, "alignments": retained_reads, "readinfo": readinfo,
            "raw_modifications": raw_modifications, "variants": records,
            "variant_cache": None,
            "long_start": long_start, "long_end": long_end,
            "has_reads": bool(alignments),
        })
    return {"chrom": chrom, "start": start, "end": end, "samples": samples,
            "long_start": min((span[0] for span in spans), default=start),
            "long_end": max((span[1] for span in spans), default=end)}


def _cached_variant_calls(sample, chrom, max_indel_length):
    cached = sample["variant_cache"]
    if cached is not None and cached[0] == max_indel_length:
        return cached[1].copy()
    sites = {}
    for position, reference, alternatives in sample["variants"]:
        alts = _supported_alts(reference, alternatives, max_indel_length)
        if alts:
            sites[position] = (reference, any(len(alt) != len(reference) for alt in alts))
    records = []
    for read in sample["alignments"] if sites else []:
        reference_positions = read.get_reference_positions(full_length=True)
        for query_position, reference_position in enumerate(reference_positions):
            if reference_position is None:
                continue
            position = reference_position + 1
            variant = sites.get(position)
            if variant is None:
                continue
            reference, has_indel = variant
            observed = _read_allele_at_variant(
                read.query_sequence, reference_positions, query_position,
                reference, has_indel,
            )
            if observed is None or abs(len(observed) - len(reference)) > max_indel_length:
                continue
            records.append({
                "chr": chrom, "pos": position, "base": observed,
                "refallele": reference, "variant_type": _variant_type(reference, observed),
                "rev": read.is_reverse, "hp": sample["readinfo"][read.query_name][0],
                "rname": read.query_name,
            })
    calls = pd.DataFrame(records, columns=[
        "chr", "pos", "base", "refallele", "variant_type", "rev", "hp", "rname",
    ])
    sample["variant_cache"] = (max_indel_length, calls)
    return calls.copy()


def process_region_data(loaded, max_indel_length=DEFAULT_MAX_INDEL_LENGTH,
                        modification_cutoff=DEFAULT_MODIFICATION_CUTOFF,
                        sort_position=None, progress_callback=None,
                        sort_methylation_average=False, coordinate_mode="compact"):
    if coordinate_mode not in {"compact", "genomic"}:
        raise ValueError("Coordinate mode must be compact or genomic")
    chrom, start, end = loaded["chrom"], loaded["start"], loaded["end"]
    sample_calls, reads = {}, {}
    for sample in loaded["samples"]:
        name = sample["name"]
        if progress_callback:
            progress_callback("Processing " + name)
        modification_key = ("modification-v1", chrom, start, end, modification_cutoff)
        modification_cache = sample.get("modification_cache")
        if modification_cache is None or modification_cache["key"] != modification_key:
            modifications = _classify_modifications(sample["raw_modifications"], modification_cutoff)
            base_table = _ordered_read_table(
                sample["readinfo"], None, modifications, {}, None,
                methylation_region=(start, end), sort_methylation_average=False,
            )
            modification_cache = {"key": modification_key, "calls": modifications,
                                  "averages": base_table["methylation_average"]}
            sample["modification_cache"] = modification_cache
        modifications = modification_cache["calls"].copy()
        averages = modification_cache["averages"]
        association_key = (modification_key, max_indel_length)
        association_cache = sample.get("association_cache")
        if association_cache is None or association_cache["key"] != association_key:
            pileup = _cached_variant_calls(sample, chrom, max_indel_length)
            pileup["methylation_average"] = pileup["rname"].map(averages)
            association_cache = {"key": association_key,
                                 "calls": _annotate_methylation_kruskal(pileup)}
            sample["association_cache"] = association_cache
        pileup = association_cache["calls"].copy()
        sort_bases = {}
        if sort_position is not None:
            sort_cache = sample.get("sort_base_cache")
            if sort_cache is None or sort_cache[0] != sort_position:
                for read in sample["alignments"]:
                    if not read.query_sequence:
                        continue
                    for query_position, reference_position in enumerate(
                        read.get_reference_positions(full_length=True)
                    ):
                        if reference_position == sort_position - 1:
                            sort_bases[read.query_name] = read.query_sequence[query_position]
                            break
                sample["sort_base_cache"] = (sort_position, sort_bases)
            else:
                sort_bases = sort_cache[1]
        table = _ordered_read_table(
            sample["readinfo"], pileup, modifications, sort_bases, sort_position,
            methylation_region=(start, end),
            sort_methylation_average=sort_methylation_average,
            methylation_averages=averages,
        )
        reads[name] = table
        for frame, call_type in [(modifications, "modification"), (pileup, "variant")]:
            frame["rname2id"] = frame["rname"].map(table[3])
            frame["methylation_average"] = frame["rname"].map(table["methylation_average"])
            frame["call_type"] = call_type
        sample_calls[name] = pd.concat([modifications, pileup], ignore_index=True, sort=False).reindex(
            columns=[column for column in CALL_COLUMNS if column != "sample"]
        )
    annotations = {name: features.copy() for name, features in loaded.get("annotations", {}).items()}
    names = loaded.get("track_order", [sample["name"] for sample in loaded["samples"]] + list(annotations))
    if not names:
        raise ValueError("Add at least one track")
    long_start, long_end = loaded.get("long_start", start), loaded.get("long_end", end)
    observed_positions = set()
    for frame in sample_calls.values():
        observed_positions.update(frame["pos"].dropna().unique())
    calls_empty = not any(not frame.empty for frame in [*sample_calls.values(), *annotations.values()])
    positions = observed_positions.copy()
    for features in annotations.values():
        positions.update(features["pos"].dropna().unique())
        positions.update(features["end"].dropna().unique())
    positions.update([long_start + 1, long_end])
    if annotations or calls_empty:
        positions.update([start + 1, end])
    position_ids = ({position: position for position in positions}
                    if coordinate_mode == "genomic"
                    else dict(zip(sorted(positions), range(len(positions)))))
    for frame in sample_calls.values():
        frame["pos2id"] = frame["pos"].map(position_ids)
    for features in annotations.values():
        features["pos2id"] = features["pos"].map(position_ids)
        features["end_pos2id"] = features["end"].map(position_ids)
    region_xrange = None
    if annotations or calls_empty:
        region_xrange = (position_ids[start + 1], position_ids[end])
    elif coordinate_mode == "genomic":
        region_xrange = (start + 1, end)
    else:
        closest_start = min((p for p in observed_positions if p >= start), default=long_start + 1)
        closest_end = max((p for p in observed_positions if p <= end), default=long_end)
        region_xrange = (position_ids[closest_start], position_ids[closest_end])
    annotation_region = (loaded.get("long_start", start), loaded.get("long_end", end))
    view_xrange = (position_ids[long_start + 1], position_ids[long_end])
    position_table = pd.DataFrame({"pos": sorted(positions)})
    position_table["pos2id"] = position_table["pos"].map(position_ids)
    return PlotData(chrom, start, end, names, sample_calls, reads, annotations, region_xrange,
                    annotation_region, coordinate_mode, view_xrange, position_table)


def prepare_plot_data(
    chrom, start, end, bams, vcfs, names, cpg_table,
    max_indel_length=DEFAULT_MAX_INDEL_LENGTH,
    modification_cutoff=DEFAULT_MODIFICATION_CUTOFF,
    sort_position=None, progress_callback=None, sort_methylation_average=False,
    coordinate_mode="compact",
):
    loaded = load_region_data(chrom, start, end, bams, vcfs, names, cpg_table,
                              progress_callback=progress_callback)
    return process_region_data(
        loaded, max_indel_length, modification_cutoff, sort_position,
        progress_callback, sort_methylation_average, coordinate_mode,
    )
