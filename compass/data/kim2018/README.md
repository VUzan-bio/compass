# Kim et al. 2018 — Seq-deepCpf1 Training Data

## Citation
Kim HK, Min S, Song M, et al. Deep learning improves prediction of CRISPR-Cpf1
guide RNA activity. *Nature Biotechnology* 36:239–241 (2018).
DOI: 10.1038/nbt.4061. PMID: 29431740.

## Data Source
- GitHub: https://github.com/MyungjaeSong/Paired-Library (branch: DeepCpf1-code)
- Nature Biotech Supplementary Data tables

## Download
```bash
git clone https://github.com/MyungjaeSong/Paired-Library.git
cd Paired-Library
git checkout DeepCpf1-code
```

## Format
- ~15,000 AsCpf1 target sequences
- 34-nt window: context upstream (4 nt) + PAM (4 nt) + protospacer (20 nt) + context downstream (6 nt)
- Label: indel frequencies (cis-cleavage activity) from integrated lentiviral libraries in HEK293T cells

## Splits
- HT1: training
- HT2: validation
- HT3: held-out test (never seen during training or hyperparameter tuning)

## Preprocessing
Labels: log2(indel_freq + 1) then min-max to [0, 1] (see `compass.scoring.preprocessing.normalise_labels`).
