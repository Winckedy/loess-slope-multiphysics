# loess-slope-multiphysics

Reproducible analysis code for:

> **Rainfall does not instantaneously or proportionally drive loess-slope
> deformation** — a five-layer method-ladder audit (WGCNA-style networks,
> convergent cross mapping, lagged transfer entropy, wavelet coherence,
> hidden Markov regimes + interpretable machine learning) on three
> instrumented loess slopes (Xiji, Qingyang, Tianshui; Chinese Loess
> Plateau, 2023–2025), with an out-of-sample test on three French
> landslide sites (Séchilienne, Viella, Villerville).

## Repository layout

```
loess-slope-multiphysics/
├── src/                                  method library (importable modules)
│   ├── common.py                         IO, daily panels, event table, style
│   ├── ccm.py                            convergent cross mapping
│   ├── transfer_entropy.py               lagged transfer entropy
│   ├── wavelet_coherence.py              Morlet wavelet coherence
│   ├── hmm_regimes.py                    2-state hidden Markov regimes
│   ├── ml_response.py                    interpretable gradient boosting
│   └── six_methods.py                    cross-audit of six method families
├── scripts/                              entry points (run in order)
│   ├── build_daily_dataset.py            raw records -> daily panels (results/)
│   ├── profile_raw_records.py            raw-record profile (spans, gaps)
│   ├── run_network_analysis.py           layer 1: WGCNA-style sensor network
│   ├── run_beta_sensitivity.py           network robustness (beta sweep)
│   ├── run_main_pipeline.py              layers 2-5 + synthetic validation (~80 s)
│   ├── run_cross_audit.py                cross-audit of six method families
│   ├── run_early_warning_backcalc.py     early-warning back-calculation
│   ├── run_french_site_profile.py        French sites: record profile
│   ├── run_french_validation.py          French out-of-sample pipeline test
│   ├── make_main_figures.py              main-text figures
│   ├── make_supplementary_figures.py     supplementary figures
│   ├── make_french_validation_figure.py  validation figure
│   ├── make_site_maps.py                 site/terrain maps
│   └── make_technical_framework_figure.py workflow schematic
├── data/                                 (git-ignored; see data/README.md)
├── results/                              (populated by runs)
├── requirements.txt
└── LICENSE
```

## Quick start

```bash
pip install -r requirements.txt
# 1. obtain the data (see data/README.md) and place it under data/
python scripts/build_daily_dataset.py        # raw -> daily panels
python scripts/run_main_pipeline.py          # ~80 s, all quantitative claims
python scripts/run_network_analysis.py       # network layer
python scripts/run_early_warning_backcalc.py # warning back-calculation
python scripts/run_french_validation.py      # external validation
```

All quantitative claims in the paper are machine-checkable against the JSON
files written to `results/` (`loess_results.json`, `loess_advanced.json`,
`loess_six_methods.json`, `loess_ewi_backcalc.json`,
`french_pipeline_test.json`).

## Method ladder

Every layer is first validated against **synthetic ground truth with a known
answer** (results stored in the `validation` block of each output JSON), then
applied to the records with **pre-registered parameter choices** (declared in
the script headers, not tuned post hoc). Significance is calibrated by
surrogates throughout: random-phase (Fourier) surrogates for CCM, circular
shifts for transfer entropy, AR(1) surrogate pairs for wavelet coherence.

## Data

The loess monitoring dataset is published by the National Cryosphere Desert
Data Center, DOI [10.12072/ncdc.loess.db7336.2026](https://doi.org/10.12072/ncdc.loess.db7336.2026)
(CC BY 4.0). French landslide records: OMIV / THEIA EasyData (CC BY 4.0).
See `data/README.md` for the expected folder layout.

## License

MIT — see [LICENSE](LICENSE).
