"""Streamlit voice-sentiment app — pastel blue-purple UI (v4)
──────────────────────────────────────────────────────────────
• Loại bỏ label "Chọn file âm thanh" để tránh trùng lặp.
• Ẩn hoàn toàn cụm "Drag and drop file here" và dòng giới hạn dung lượng.
• Tinh chỉnh drop-zone: viền dashed pastel, hover subtle.
• Typography và khoảng cách mềm mại hơn.
"""

from __future__ import annotations

import os
import sys

import streamlit as st
import librosa
import os
import sys
import tempfile

MODULE_DIR = os.path.abspath(os.path.dirname(__file__))
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)

from modelcenn import SpeechEmotionRecognizer, CeNNBlock
from tensorflow.keras.layers import InputLayer

# ─────────────────────────────────────────────────────────────
# Legacy InputLayer patch
# ─────────────────────────────────────────────────────────────
class LegacyInputLayer(InputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape and len(batch_shape) > 1:
            kwargs["shape"] = tuple(x for x in batch_shape[1:] if x is not None)
        super(LegacyInputLayer, self).__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        if "batch_shape" in config:
            batch_shape = config.pop("batch_shape")
            if batch_shape and len(batch_shape) > 1:
                config["shape"] = tuple(x for x in batch_shape[1:] if x is not None)
        if "batch_input_shape" in config:
            batch_input_shape = config.pop("batch_input_shape")
            if batch_input_shape and len(batch_input_shape) > 1:
                config["shape"] = tuple(x for x in batch_input_shape[1:] if x is not None)
        return super(LegacyInputLayer, cls).from_config(config)

# Đường dẫn model CeNN đã huấn luyện
MODEL_PATH = os.path.join(MODULE_DIR, "saved_models", "emotion_model.keras")

# Load model CeNN
@st.cache_resource(show_spinner=False)
def load_cenn():
    try:
        model = SpeechEmotionRecognizer.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Lỗi tải mô hình CeNN: {e}")
        raise

model = load_cenn()

# ----------------------------------------------------------------------------
# Streamlit page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nhận dạng cảm xúc qua tiếng nói",
    page_icon="🎤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
html, body, [data-testid="stApp"] {
    height: 100%;
    background: radial-gradient(circle at 15% 15%, #cfe4ff 0%, #e8d9ff 70%) !important;
}

[data-testid="stApp"] .block-container {
    max-width: 800px;
    background: rgba(255, 255, 255, 0.50);
    backdrop-filter: blur(14px) saturate(170%);
    -webkit-backdrop-filter: blur(14px) saturate(170%);
    margin-top: 2rem; /* Giảm margin để mềm mại hơn */
    padding: 2rem 2rem 2.5rem; /* Giảm padding cho khoảng cách mềm mại */
    border: 1px solid rgba(255, 255, 255, 0.38);
    border-radius: 1.5rem;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05); /* Giảm bóng để nhẹ nhàng hơn */
}

h1 {
    font-size: 2.2rem; /* Giảm font-size cho typography mềm mại */
    font-weight: 700; /* Giảm weight để nhẹ nhàng hơn */
    color: #2e324d;
    margin-bottom: 0.5rem; /* Giảm margin-bottom */
    text-align: center;
}

p, label, .stFileUploader, button, div, span {
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 0.95rem; /* Giảm font-size cho typography mềm mại */
    line-height: 1.5; /* Tăng line-height để dễ đọc */
}

/* ---------- Dropzone refinements ---------- */
[data-testid="stFileDropzone"] {
    border: 2px dashed #a9c9ff !important;
    background: rgba(169, 201, 255, 0.18) !important;
    border-radius: 12px !important;
    transition: background 0.3s ease, border-color 0.3s ease; /* Thêm hiệu ứng hover mượt hơn */
    padding: 1.5rem 1rem 2.5rem !important; /* Điều chỉnh padding cho cân đối */
    position: relative;
}

[data-testid="stFileDropzone"]:hover {
    background: rgba(213, 202, 255, 0.3) !important; /* Hover subtle */
    border-color: #8ab4f8 !important; /* Viền pastel khi hover */
}

/* Ẩn hoàn toàn các dòng mặc định */
[data-testid="stFileDropzoneInstructions"],
[data-testid="stFileUploaderSizeLimit"],
[data-testid="stFileUploaderLabel"] { /* Ẩn label để tránh trùng lặp */
    display: none !important;
}

/* Gắn dòng mô tả tùy chỉnh */
[data-testid="stFileDropzone"]::after {
    content: "🎧 Kéo thả hoặc chọn tệp WAV (≤ 200MB)";
    position: absolute;
    top: 1rem;
    left: 50%;
    transform: translateX(-50%);
    color: #2e324d;
    font-size: 1rem; /* Giảm font-size cho mềm mại */
    font-weight: 500;
    font-family: 'Inter', sans-serif;
}

.stAlert {
    border-radius: 12px;
    padding: 1rem 1.25rem;
}

footer {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <h1>Nhận dạng Cảm Xúc</h1>
    <p style="text-align:center; font-size:0.95rem;">Kéo thả hoặc chọn tệp <b>.wav</b> để hệ thống nhận dạng cảm xúc của giọng nói.</p>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# File uploader (label collapsed)
# ----------------------------------------------------------------------------
col_left, col_center, col_right = st.columns([1, 5, 1])
with col_center:
    uploaded_file = st.file_uploader(
        label="Chọn file âm thanh",
        type=["wav"],
        label_visibility="collapsed",
        key="uploader",
    )

if not uploaded_file:
    st.info("Vui lòng thêm tệp .wav để bắt đầu.")
    st.stop()

# ----------------------------------------------------------------------------
# Audio preview & inference
# ----------------------------------------------------------------------------
st.audio(uploaded_file, format="audio/wav")

try:
    # Lưu file tạm để truyền đường dẫn cho CeNN
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpf:
        tmpf.write(uploaded_file.read())
        tmp_path = tmpf.name

    if not os.path.exists(tmp_path):
        st.error("Không thể lưu file tạm để dự đoán.")
        st.stop()

    with st.spinner("🔎 Đang phân tích…"):
        result = model.predict(tmp_path)

    os.remove(tmp_path)

    st.markdown("---")
    if result['class'] == 'positive':
        st.success(f"Cảm xúc **TÍCH CỰC**\n\nĐộ tin cậy: **{result['confidence']:.2f}**")
    else:
        st.warning(f"Cảm xúc **TIÊU CỰC**\n\nĐộ tin cậy: **{result['confidence']:.2f}**")

except Exception as exc:
    st.error(f"❌ Lỗi xử lý âm thanh: {exc}")

