
# Pharma Demand Sensing: Time Series Forecasting

### 🚀 Overview
An end-to-end predictive engine designed to mitigate supply chain stock-outs in the pharmaceutical industry. This project bridges public health data (WHO FluNet) with commercial sales trends using **SARIMA** modeling on **Databricks**.

---

### 📊 The Business Problem
Supply chain teams often struggle with 2-week lead times. This project investigates whether global epidemiological signals (flu activity) can act as a leading indicator for respiratory drug demand.

**Key Objective:** Provide inventory optimization teams with a reliable forecast to prevent stock-outs during peak seasonal shifts.

---

### 🛠️ Tech Stack
- **Environment:** Databricks (PySpark), Jupyter
- **Languages:** Python, SQL
- **Libraries:** Statsmodels, Pandas, Matplotlib, MLflow
- **Deployment:** Streamlit (Live Web App)

---

### 🧬 Project Evolution & Logic
1. **v1 (Jupyter):** Initial exploration using traditional Pandas/Statsmodels. Resulted in 69% AUC/MAPE variance; identified data noise issues.
2. **v2 (Databricks - Production):** Scaled the pipeline using Spark for 30 years of WHO data. Applied spatial-temporal aggregation and 3-week rolling means to extract the underlying signal.
3. **Model Selection:** SARIMA(1,1,1)(1,1,0,52). 
   - *Key Insight:* Validated that the external WHO signal became redundant (p=0.65) once seasonal AR terms were optimized—the model successfully captured the "flu signal" from historical sales alone.

---

### 📈 Results
- **17.4% Improvement** over the naive seasonal baseline.
- **55% MAPE** on weekly granular data (within the 40-60% industry gold standard).
- **2-Week Lead Time:** Provided actionable forecasting windows for inventory planning.

---

### Key Insight :
- "Model residuals passed the Ljung-Box test (p=0.81), confirming that the SARIMAX(1,1,1)(1,1,0,52) architecture successfully captured all available signal, rendering the exogenous FluNet features statistically redundant."

---

### 📂 Repository Structure
- `/notebooks`: Contains both `01_Initial_Exploration.ipynb` and `02_Databricks_Production_Model.ipynb`.
- `/src`: Modular Python scripts for data cleaning and feature engineering.
- `app.py`: Streamlit source code for the live dashboard.

---

### 🔗 Links
- **Live App:** [https://pharma-demand-forecasting-hkdb6dl4yeoovnyt8gv8wq.streamlit.app/]
- **LinkedIn Post:** [https://www.linkedin.com/feed/update/urn:li:activity:7446658407387865088/]
