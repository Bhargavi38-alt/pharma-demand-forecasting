import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Pharma Demand Forecasting", layout="wide")

st.title("🏥 Pharma Demand Sensing — R03 Respiratory Sales")
st.markdown("SARIMA model on weekly respiratory pharma sales — WHO flu index removed after finding it added noise vs the seasonal AR term")

# ── Sidebar ──
st.sidebar.header("Model Parameters")
train_pct = st.sidebar.slider("Train size (%)", 60, 90, 80)
p = st.sidebar.selectbox("AR order (p)", [1, 2], index=0)
q = st.sidebar.selectbox("MA order (q)", [1, 2], index=0)
show_ci = st.sidebar.checkbox("Show 95% CI", value=True)

# ── Load data ──
@st.cache_data
def load_data():
    return pd.read_csv("data/pharma_demand_sensing_gold.csv",
                       index_col=0, parse_dates=True)

df = load_data()

# ── Split ──
split_idx = int(len(df) * train_pct / 100)
df_train  = df.iloc[:split_idx]
df_test   = df.iloc[split_idx:]

# ── Fit SARIMA (no exogenous — flu index removed) ──
with st.spinner("Fitting SARIMA model..."):
    model = SARIMAX(
        np.log1p(df_train['R03_sales']),
        order=(p, 1, q),
        seasonal_order=(1, 1, 0, 52),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

# ── Forecast ──
fc      = model.get_forecast(steps=len(df_test))
pred    = np.expm1(fc.predicted_mean)
ci      = np.expm1(fc.conf_int())
actuals = df_test['R03_sales']

# ── Metrics ──
mape        = np.mean(np.abs((actuals - pred) / actuals)) * 100
naive_pred  = df_train['R03_sales'].iloc[-len(df_test):].values
naive_mape  = np.mean(np.abs(
    (actuals.values - naive_pred) / actuals.values
)) * 100
improvement = naive_mape - mape

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("MAPE",           f"{mape:.1f}%")
col2.metric("Accuracy",       f"{100-mape:.1f}%")
col3.metric("AIC",            f"{model.aic:.1f}")
col4.metric("Naive baseline", f"{naive_mape:.1f}%")
col5.metric("vs Baseline",    f"+{improvement:.1f}pp better")

# ── Plot ──
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_train.index, df_train['R03_sales'],
        color='blue', label='Train')
ax.plot(df_test.index, actuals,
        color='green', label='Actual')
ax.plot(df_test.index, pred,
        color='red', linestyle='--', label='Forecast')
if show_ci:
    ax.fill_between(df_test.index,
                    ci.iloc[:, 0], ci.iloc[:, 1],
                    alpha=0.2, color='red', label='95% CI')
ax.axvline(df_test.index[0], color='gray', linestyle=':', linewidth=1)
ax.set_title("SARIMA Forecast vs Actual — exogenous flu index removed")
ax.set_xlabel("Date")
ax.set_ylabel("R03 Sales")
ax.legend()
st.pyplot(fig)

# ── Finding ──
st.info("""
**Key Finding:** WHO flu index was removed as an exogenous variable.
The seasonal AR term (ar.S.L52) already captures flu seasonality 
from historical patterns, making the external signal redundant and 
adding noise. Pure SARIMA outperformed SARIMAX on this dataset.
""")

# ── Forecast table ──
st.subheader("Forecast vs Actual")
result_tbl = pd.DataFrame({
    'Actual':   actuals.values.round(1),
    'Forecast': pred.values.round(1),
    'Error %':  np.abs((actuals.values - pred.values)
                       / actuals.values * 100).round(1)
}, index=df_test.index)
st.dataframe(result_tbl.style.highlight_max(
    subset=['Error %'], color='#ffcccc'))