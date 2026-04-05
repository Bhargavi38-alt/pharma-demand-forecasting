# app.py — run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Pharma Demand Forecasting", layout="wide")

st.title("🏥 Pharma Demand Sensing — R03 Respiratory Sales")
st.markdown("SARIMAX model with CDC Flu Index as exogenous demand signal")

# ── Sidebar controls ──
st.sidebar.header("Model Parameters")
train_pct  = st.sidebar.slider("Train size (%)", 60, 90, 80)
p = st.sidebar.selectbox("AR order (p)", [1, 2], index=0)
q = st.sidebar.selectbox("MA order (q)", [1, 2], index=0)
show_ci = st.sidebar.checkbox("Show 95% CI", value=True)

# ── Load data ──
@st.cache_data
def load_data():
    # Replace with your actual path
    return pd.read_csv("data/pharma_demand_sensing_gold.csv",
                   index_col=0, parse_dates=True)

df = load_data()

# ── Split & fit ──
split_idx = int(len(df) * train_pct / 100)
df_train  = df.iloc[:split_idx]
df_test   = df.iloc[split_idx:]

scaler    = StandardScaler()
X_train   = scaler.fit_transform(df_train[['flu_index']])
X_test    = scaler.transform(df_test[['flu_index']])

with st.spinner("Fitting SARIMAX model..."):
    model = SARIMAX(
        np.log1p(df_train['R03_sales']),
        exog=X_train,
        order=(p, 1, q),
        seasonal_order=(1, 1, 0, 52),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

fc       = model.get_forecast(steps=len(df_test), exog=X_test)
pred     = np.expm1(fc.predicted_mean)
ci       = np.expm1(fc.conf_int())
actuals  = df_test['R03_sales']
mape     = np.mean(np.abs((actuals - pred) / actuals)) * 100

# ── Metrics row ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAPE",     f"{mape:.1f}%")
col2.metric("Accuracy", f"{100-mape:.1f}%")
col3.metric("AIC",      f"{model.aic:.1f}")
col4.metric("Train weeks", len(df_train))

# ── Plot ──
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df_train.index, df_train['R03_sales'],
        color='blue', label='Train')
ax.plot(df_test.index,  actuals,
        color='green', label='Actual')
ax.plot(df_test.index,  pred,
        color='red', linestyle='--', label='Forecast')
if show_ci:
    ax.fill_between(df_test.index,
                    ci.iloc[:, 0], ci.iloc[:, 1],
                    alpha=0.2, color='red', label='95% CI')
ax.axvline(df_test.index[0], color='gray',
           linestyle=':', linewidth=1)
ax.set_title("SARIMAX Forecast vs Actual")
ax.legend()
st.pyplot(fig)

# ── Data table ──
st.subheader("Forecast vs Actual")
result_tbl = pd.DataFrame({
    'Actual':   actuals.values,
    'Forecast': pred.values,
    'Error %':  np.abs((actuals.values - pred.values)
                       / actuals.values * 100).round(1)
}, index=df_test.index)
st.dataframe(result_tbl.style.highlight_max(
    subset=['Error %'], color='#ffcccc'))