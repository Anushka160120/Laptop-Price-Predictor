import streamlit as st
from preprocessing import load_and_train, predict_price, convert_memory

#st.set_page_config(page_title="Laptop Price Predictor 💻", layout="centered")
st.title("💻 Laptop Price Predictor (in Euros)")

model, cat_cols, num_cols, r2, rmse = load_and_train()

st.sidebar.header("📊 Model Metrics")
st.sidebar.metric("R² Score", f"{r2:.2f}")
st.sidebar.metric("RMSE", f"€ {rmse:.2f} ")


st.markdown("## 🔧 Enter Laptop Specifications")

company = st.selectbox("Company", ['Dell', 'HP', 'Apple', 'Lenovo', 'Acer', 'Asus', 'MSI', 'Toshiba', 'Samsung', 'LG', 'Other'])
laptop_type = st.selectbox("Type", ['Ultrabook', 'Gaming', 'Notebook', 'Netbook', 'Workstation'])
inches = st.slider("Screen Size (Inches)", 10.0, 20.0, 15.6)
resolution = st.selectbox("Screen Resolution", ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 'IPS Panel Full HD 1920x1200', 'IPS Panel Full HD 2560x1440'])
cpu = st.selectbox("CPU", ['Intel Core i7', 'Intel Core i5', 'Intel Core i3', 'AMD Ryzen', 'Other'])
ram = st.slider("RAM (GB)", 2, 64, 8)
memory = st.selectbox("Storage (GB)", ["128", "256", "512", "1024","128+128", "128+256", "256+256", "256+512", "512+512",
    "512+1024", "1024+1024"])
gpu = st.selectbox("GPU", ['Intel HD', 'Nvidia GTX', 'AMD Radeon', 'Other'])
os = st.selectbox("Operating System", ['Windows', 'Mac', 'Linux', 'Chrome OS', 'No OS'])
weight = st.slider("Weight (kg)", 0.5, 5.0, 2.0)

if st.button(" Predict Price"):
    input_data = {
        'Company': company,
        'TypeName': laptop_type,
        'Inches': inches,
        'ScreenResolution': resolution,
        'Cpu': cpu,
        'Ram': ram,
        'Memory': convert_memory(memory),
        'Gpu': gpu,
        'OpSys': os,
        'Weight': weight
    }

    price = predict_price(model, input_data)
    st.success(f" Estimated Laptop Price: **€{price:.2f}**")
