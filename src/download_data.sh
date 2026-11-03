#!/usr/bin/env bash
# Download all public inputs. No credentials required.
set -euo pipefail
mkdir -p data && cd data
echo "GEO GSE346736 (GUIDE-seq-2, 1115 sites x 95 donors) ..."
curl -sSL -O "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE346nnn/GSE346736/suppl/GSE346736_RAW.tar"
tar -xf GSE346736_RAW.tar
echo "GEO GSE286299 (rhAmpSeq targeted validation) ..."
curl -sSL -O "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE286nnn/GSE286299/suppl/GSE286299_rhamp_seq_merged_table.csv.gz"
echo "1000 Genomes sample panel ..."
curl -sSL -o panel.txt "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"
echo "Done."
