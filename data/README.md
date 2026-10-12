# Data

All analyses run on the published dataset:

> Zhang, F. (2026). *2023-2025 Multi-physics Observation Dataset of Loess Slopes
> at Huoxiangou in Qingyang, the Xiaocha Formation in Xiji, and Luoyugou in
> Tianshui, Gansu*. National Cryosphere Desert Data Center.
> DOI: https://doi.org/10.12072/ncdc.loess.db7336.2026 (CC BY 4.0)

Download the dataset and arrange it under `data/` as follows (file and
sub-folder names are kept exactly as delivered by the data center):

```
data/
├── Xiji/       Xiji xlsx record (2023-07 - 2024-10)
├── Qingyang/   weather-station, hydrology (10-depth soil moisture) and
│               slope-logger records (2023-07 - 2025-10)
├── Tianshui/   weather station, slope logger and liquid-level records
│               (2024-10 - 2025-10)
└── FrenchLandslides/   external validation sites (Sechilienne, Viella,
                        Villerville), OMIV / THEIA EasyData, CC BY 4.0
```

`data/` is git-ignored: the repository ships code only.
