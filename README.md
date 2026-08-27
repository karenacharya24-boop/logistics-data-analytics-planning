# Logistics Data Analytics Internship

Repository containing milestones, scripts, and analytical workflows for the Logistics Data Analyst Internship track.

---

## Week 1: Strategic Planning & Data Exploration
* **Objective:** Establishing operational KPIs (OTD, CPM, Transit Deviation) and initial analytical roadmaps.
* **Code:** `main.py` — Delivery zone clustering (K-Means) and baseline transit time regression (Random Forest).

---

## Week 2: Data Collection, Cleaning & Preprocessing
* **Objective:** Building automated pipelines to clean raw multi-modal shipment telemetry.
* **Key Techniques:**
  * **Missing Value Imputation:** Median imputation for continuous variables and mode imputation for categorical fields.
  * **Outlier Handling:** Interquartile Range (IQR) boundary clipping to handle sensor anomalies.
  * **Feature Engineering & Normalization:** Ordinal/One-Hot encoding and Min-Max scaling (`[0, 1]`).
* **Code:** `week2_preprocessing.py`
