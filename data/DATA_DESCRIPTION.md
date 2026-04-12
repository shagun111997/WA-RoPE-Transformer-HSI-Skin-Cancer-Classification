# Dataset Description
NIR-SC-UFES: A portable NIR spectral dataset to skin cancer in vivo

## Source
The dataset is publicly available at: https://data.mendeley.com/datasets/j9773cyr3k/1

## Structure
- Metadata columns:
  - N
  - Sample
  - Class
  - yClass

- Remaining columns:
  - Hyperspectral spectral bands

## Labels
Binary classification:
- 0 → Non-cancer
- 1 → Cancer

## Preprocessing
- SNV normalization
- Removal of:
  - NaN values
  - Flat spectra
  - Outliers
