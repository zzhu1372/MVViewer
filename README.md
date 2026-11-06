# MVViewer

A genome browser for interactive visualization and statistical analysis of CpG methylation and small variants in long-read sequencing data.

## Install

Clone MVViewer and create its [conda](https://conda-forge.org/download/) environment:

```bash
git clone https://github.com/zzhu1372/MVViewer.git
cd MVViewer
conda env create -f environment.yml
conda activate mvviewer
```

The environment uses the Kaleido version 0.2.1, which bundles Chromium, so a separate Chrome installation is not needed. Please do not upgrade Plotly or Kaleido independently if you do not have Chromium/Chrome installed.


## Reference preparation

Reference preparation (generating the CpG table) should be performed before first use. You can use your own FASTA or FASTA.gz file:

```bash
python init.py /path/to/reference.fa
```

Alternatively, you can use a reference assembly from UCSC:

```bash
python init.py hg19
python init.py hg38
```

The prepared reference will be stored in `MVViewer/genomes/` unless a different reference cache directory is specified using the following command:

```bash
export MVVIEWER_CACHE_DIR=PATH
```

## Run

Launch MVViewer with the following command, then access the interface through a web browser at [http://localhost:8501](http://localhost:8501) (port might be different if 8501 is occupied, see the message from streamlit):

```bash
conda activate mvviewer
python -m streamlit run app.py
```

Alternatively, you can generate figures with the command-line version:

```bash
python mvvcli.py --tracks tracks.txt --reference hg38 \
  --region chr19:56837539-56840146 --output figure.pdf
```

`--tracks` accepts a file that contains a list of track definitions in the same format with the web-app version (see below). More information can be found with
```bash
python mvvcli.py --help
```

## Plot

Select a reference and a genomic region of interest, then enter one track per line in `Track definitions`. Columns must be separated by commas. A BAM file or a GTF is required; the VCF file and sample name are optional. Each BAM or VCF file must have an accompanying index file. To specify a BAM file and sample name without a VCF file, leave the second column empty. The filename is used when the name is omitted. Examples:

```text
/data/sample1.bam,/data/sample1.vcf.gz,sample-name-1
/data/sample2.bam,/data/sample2.vcf.gz
/data/sample3.bam,,sample-name-3
/data/sample4.bam,
/data/annotation.gtf,genes
```


