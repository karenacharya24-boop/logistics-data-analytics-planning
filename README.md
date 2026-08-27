# logistics-data-analytics-planning
# Logistics Data Analytics - Strategic Planning & Exploration

## Overview
This repository contains the strategic planning framework and exploratory scripts for optimizing last-mile delivery operations, transit time estimations, and zone allocation.

## Key Performance Indicators (KPIs)
- **On-Time Delivery (OTD) Rate:** Target >= 95%
- **Cost Per Mile (CPM):** Direct transit expenditure efficiency
- **Average Transit Deviation:** Discrepancy between estimated vs. actual delivery times

## Methodology
1. **Data Preprocessing & EDA:** Cleans transit logs and maps traffic density metrics.
2. **Spatial Clustering (K-Means):** Groups drop-off coordinates into micro-zones for optimized routing.
3. **Predictive Modeling (Random Forest):** Forecasts delivery durations based on load weight, distance, and congestion scores.

## Execution
```bash
python main.py
