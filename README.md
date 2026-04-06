# DisasterShield
AI-based system for predicting flood risk in rural Bangladesh.

## Overview
DisasterShield-X uses satellite data, hydrology records, and historical events to predict:
- Flood risk

The system is designed to support **decision-making** for communities and planners.

## Project Structure
```
DisasterShield/
├── README.md # Project overview
├── data/
│ ├── Raw_data/ # GeoTIFF satellite images (2019–2023)
│ │ ├── Bangladesh_RGB_YYYY.tif # 4-band Sentinel-2 imagery (flood season)
│ │ └── Bangladesh_WaterMask_YYYY.tif # Binary water masks (ground truth labels)
│ └── Data_script_19_23/
│ └── Earth_engine_data_2019_2023 # Google Earth Engine script
├── models/
│ └── flood_model_2019_2023.ipynb # Main ML training notebook
├── data_reports/ # Annual PDF flood reports (2010–2021)
│ └── 2010.pdf … 2021.pdf
└── papers/
└── knn_flood.pdf # Reference paper
```
## License
MIT License
