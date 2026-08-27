# Logistics Data Analytics Internship

Repository containing milestone reports, data processing pipelines, exploratory data analysis, and predictive modeling scripts for the Logistics Data Analyst Internship track.

---

## 📌 Project Structure

| File | Milestone / Focus Area | Key Methodologies & Techniques |
| :--- | :--- | :--- |
| `main.py` | **Week 1: Strategic Planning & Exploration** | Operational KPIs (OTD, CPM), K-Means zone clustering, baseline Random Forest duration regression. |
| `week2_preprocessing.py` | **Week 2: Data Cleaning & Preprocessing** | Median/Mode imputation, IQR outlier boundary clipping, One-Hot/Ordinal encoding, Min-Max scaling. |
| `week3_eda_visualizations.py` | **Week 3: Advanced EDA & Visualizations** | Pearson correlation matrix heatmaps, fleet cost distribution boxplots, delivery time vs distance scatter analytics. |

---

## 🚀 Milestone Overviews

### Week 1: Strategic Planning & Data Exploration
* **Objective:** Establish the foundational framework for optimizing last-mile delivery operations and fleet allocation.
* **Core KPIs:** On-Time Delivery Rate ($\ge 95\%$), Cost Per Mile (CPM), and Average Transit Deviation ($< 5$ mins).
* **Script:** `main.py`

### Week 2: Data Collection, Cleaning & Preprocessing
* **Objective:** Build an automated data hygiene pipeline for corrupt multi-modal telemetry.
* **Techniques:**
  * Handled missing features via median (continuous) and mode (categorical) imputation.
  * Filtered abnormal sensor spikes using Interquartile Range (IQR) bounds.
  * Scaled operational features to $[0, 1]$ via `MinMaxScaler`.
* **Script:** `week2_preprocessing.py`

### Week 3: Advanced Data Analysis & Visualization
* **Objective:** Uncover cost drivers, transit latency bottlenecks, and vehicle fleet efficiencies using statistical visualization.
* **Key Findings:**
  * Strong positive correlation ($r = 0.83$) between route distance and delivery duration.
  * Significant cost variance across fleet tiers ($3.50/km for Heavy Trucks vs $0.80/km for Electric Cargo Bikes).
  * High traffic congestion increases transit variance by up to $38\%$ on routes exceeding $25$ km.
* **Script:** `week3_eda_visualizations.py`

---

## 🛠️ Execution & Dependencies

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
