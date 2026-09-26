from pathlib import Path

import joblib
import numpy as np
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "knn_dbscan_proxy.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"

CLUSTER_LABELS = {
    0: "Tipe Menengah",
    1: "Tipe Premium",
    2: "Tipe Compact / Ekonomis",
    3: "Tipe Premium",
    4: "Tipe Compact / Ekonomis",
    5: "Tipe Premium",
}

SEGMENT_DESCRIPTIONS = {
    "Tipe Menengah": "Properti dengan karakteristik seimbang untuk pasar menengah dan keluarga.",
    "Tipe Premium": "Properti dengan nilai tinggi, luas besar, dan harga premium.",
    "Tipe Compact / Ekonomis": "Properti berskala lebih kecil namun tetap layak untuk kebutuhan rumah sederhana dan efisien.",
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler tidak ditemukan: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_model()


def predict_segment(price, bedrooms, bathrooms, sqft_living, sqft_lot):
    if any(value <= 0 for value in [price, bedrooms, bathrooms, sqft_living, sqft_lot]):
        raise ValueError("Semua nilai harus lebih besar dari 0.")

    features = np.array(
        [[float(price), float(bedrooms), float(bathrooms), float(sqft_living), float(sqft_lot)]],
        dtype=float,
    )
    scaled_features = scaler.transform(features)
    cluster_id = int(model.predict(scaled_features)[0])
    label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
    description = SEGMENT_DESCRIPTIONS.get(label, "Properti ini memiliki karakteristik khusus.")

    if label == "Tipe Compact / Ekonomis":
        badge = "🏠 Ekonomis"
        accent = "#2ecc71"
    elif label == "Tipe Menengah":
        badge = "🏡 Menengah"
        accent = "#3498db"
    else:
        badge = "✨ Premium"
        accent = "#f39c12"

    return label, cluster_id, description, badge, accent


st.set_page_config(page_title="Dashboard Segmentasi Properti", page_icon="🏠", layout="wide")

st.title("🏠 Dashboard Segmentasi Properti")
st.caption("Prediksi tipe properti berdasarkan 5 fitur utama: harga, kamar, luas bangunan, dan luas tanah.")

with st.form("property_form"):
    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("Harga Properti", min_value=1, value=500000, step=10000)
        bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=1, value=3, step=1)
        bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=1, value=2, step=1)

    with col2:
        sqft_living = st.number_input("Luas Bangunan (sqft_living)", min_value=1, value=2500, step=100)
        sqft_lot = st.number_input("Luas Tanah (sqft_lot)", min_value=1, value=5000, step=100)

    submitted = st.form_submit_button("Prediksi Segmentasi", use_container_width=True)

if submitted:
    try:
        label, cluster_id, description, badge, accent = predict_segment(
            price, bedrooms, bathrooms, sqft_living, sqft_lot
        )

        st.markdown(
            f"""
            <div style="padding:20px; border-radius:12px; background:linear-gradient(135deg,#f8fafc,#eef2ff); border:1px solid #deb887;">
                <h3 style="margin-top:0;">✅ Hasil Prediksi</h3>
                <p><b>Kategori:</b> <span style="color:{accent}; font-weight:700;">{label}</span></p>
                <p><b>Cluster ID:</b> {cluster_id}</p>
                <p><b>Keterangan:</b> {description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"<div style='margin-top:16px; padding:10px 14px; border-radius:999px; background:#e0f2fe; display:inline-block; font-weight:700;'>{badge}</div>", unsafe_allow_html=True)

    except ValueError as exc:
        st.error(str(exc))

st.markdown("---")

st.subheader("Contoh Input Properti")
example_data = [
    {"Harga": 450000, "Kamar Tidur": 3, "Kamar Mandi": 2, "Luas Bangunan": 1800, "Luas Tanah": 2400},
    {"Harga": 900000, "Kamar Tidur": 4, "Kamar Mandi": 3, "Luas Bangunan": 2600, "Luas Tanah": 3200},
    {"Harga": 320000, "Kamar Tidur": 2, "Kamar Mandi": 1, "Luas Bangunan": 1200, "Luas Tanah": 1800},
    {"Harga": 1500000, "Kamar Tidur": 5, "Kamar Mandi": 4, "Luas Bangunan": 3600, "Luas Tanah": 4200},
]

st.dataframe(example_data, use_container_width=True)

st.markdown("### Keterangan Segmentasi")
st.info("- Tipe Compact / Ekonomis: properti dengan skala kecil atau hemat biaya\n- Tipe Menengah: properti dengan keseimbangan harga dan kebutuhan keluarga\n- Tipe Premium: properti dengan nilai tinggi, luas besar, dan kualitas premium")
