import os
from pathlib import Path

import gradio as gr
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "knn_dbscan_proxy.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler tidak ditemukan: {SCALER_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

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

EXAMPLES = [
    [450000, 3, 2, 1800, 2400],
    [900000, 4, 3, 2600, 3200],
    [320000, 2, 1, 1200, 1800],
    [1500000, 5, 4, 3600, 4200],
]


def predict_segment(price, bedrooms, bathrooms, sqft_living, sqft_lot):
    if any(value <= 0 for value in [price, bedrooms, bathrooms, sqft_living, sqft_lot]):
        return (
            "### ⚠️ Input tidak valid\nSemua nilai harus lebih besar dari 0.",
            "-",
            "Pastikan semua field diisi dengan angka yang valid.",
        )

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

    result_md = f"""
### ✅ Hasil Prediksi Segmentasi Properti
- Kategori: <span style='color:{accent}; font-weight:700'>{label}</span>
- Cluster ID: {cluster_id}
- Keterangan: {description}
"""

    summary = (
        f"Properti ini tergolong ke dalam <b>{label}</b>. "
        f"Berdasarkan model clustering, karakteristiknya cocok untuk segmen {description.lower()}"
    )
    return result_md, badge, summary


with gr.Blocks(theme=gr.themes.Soft(), css="""
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .subtitle {
        text-align: center;
        color: #4b5563;
        margin-bottom: 20px;
    }
    .card {
        border-radius: 16px;
        padding: 18px;
        background: linear-gradient(135deg, #f8fafc, #eef2ff);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .badge {
        font-size: 1rem;
        font-weight: 700;
        padding: 8px 14px;
        border-radius: 999px;
        display: inline-block;
        background: #e0f2fe;
        color: #0f172a;
    }
""") as demo:
    gr.Markdown(
        """
        <div class="main-container">
            <div class="title">🏠 Dashboard Segmentasi Properti</div>
            <div class="subtitle">Prediksi tipe properti berdasarkan 5 fitur utama: harga, jumlah kamar, luas bangunan, dan luas tanah.</div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Box():
                gr.Markdown("### 🧮 Input Properti")
                price = gr.Number(label="Harga Properti", value=500000, precision=0)
                bedrooms = gr.Number(label="Jumlah Kamar Tidur", value=3, precision=0)
                bathrooms = gr.Number(label="Jumlah Kamar Mandi", value=2, precision=0)
                sqft_living = gr.Number(label="Luas Bangunan (sqft_living)", value=2500, precision=0)
                sqft_lot = gr.Number(label="Luas Tanah (sqft_lot)", value=5000, precision=0)
                submit_btn = gr.Button("Prediksi Segmentasi", variant="primary")

        with gr.Column(scale=1):
            with gr.Box():
                gr.Markdown("### 📊 Hasil Analisis")
                result_md = gr.Markdown("### ⏳ Silakan masukkan data properti")
                result_badge = gr.Markdown("<div class='badge'>Belum ada hasil</div>")
                result_summary = gr.Markdown("Hasil prediksi akan tampil di sini.")

    gr.Markdown("### 🧪 Contoh Input")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[price, bedrooms, bathrooms, sqft_living, sqft_lot],
        label="Contoh dataset properti",
        examples_per_page=4,
    )

    submit_btn.click(
        fn=predict_segment,
        inputs=[price, bedrooms, bathrooms, sqft_living, sqft_lot],
        outputs=[result_md, result_badge, result_summary],
    )

    gr.Markdown(
        """
        ### 🔎 Keterangan segmentasi
        - Tipe Compact / Ekonomis: properti dengan skala kecil atau hemat biaya
        - Tipe Menengah: properti dengan keseimbangan harga dan kebutuhan keluarga
        - Tipe Premium: properti dengan nilai tinggi, luas besar, dan kualitas premium
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
