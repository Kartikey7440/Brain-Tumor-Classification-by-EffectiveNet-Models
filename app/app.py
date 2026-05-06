import streamlit as st
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import os
import base64
import time

def rgba(r, g, b, a=1):
    return (r/255, g/255, b/255, a)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Dark Medical Theme ── */
.stApp {
    background: #080d14;
    color: #e8edf5;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem; max-width: 100%; }

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 40%, #091220 100%);
    border-bottom: 1px solid rgba(0,200,255,0.15);
    padding: 2.5rem 3rem 2rem;
    margin: 0 -2rem 2.5rem -2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -10%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(0,180,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-header::after {
    content: '';
    position: absolute;
    bottom: -30%;
    right: 5%;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(100,0,255,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00c8ff, #ffffff 50%, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: rgba(180,210,255,0.65);
    margin: 0.6rem 0 0 0;
    font-weight: 300;
    letter-spacing: 0.04em;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0,200,255,0.08);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-size: 0.72rem;
    color: #00c8ff;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
    margin-top: 1.5rem;
}
.hero-stat {
    display: flex;
    flex-direction: column;
}
.hero-stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
}
.hero-stat-label {
    font-size: 0.72rem;
    color: rgba(180,210,255,0.5);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}

/* ── Upload Zone ── */
.upload-zone {
    background: rgba(255,255,255,0.02);
    border: 2px dashed rgba(0,200,255,0.2);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    transition: all 0.3s ease;
    margin-bottom: 1.5rem;
}
.upload-zone:hover { border-color: rgba(0,200,255,0.5); background: rgba(0,200,255,0.03); }

/* ── Stagger Cards ── */
.result-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #00c8ff);
    opacity: 0.8;
}

/* ── Diagnosis Banner ── */
.diagnosis-banner {
    background: linear-gradient(135deg, var(--bg1), var(--bg2));
    border: 1px solid var(--accent);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.diagnosis-banner::after {
    content: '';
    position: absolute;
    top: -40%;
    right: -5%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, var(--glow) 0%, transparent 70%);
}
.diagnosis-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.diagnosis-name {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.diagnosis-confidence {
    font-size: 1rem;
    color: rgba(255,255,255,0.6);
    font-weight: 300;
}
.diagnosis-confidence span {
    color: var(--accent);
    font-weight: 600;
}

/* ── Prob Bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.75rem;
}
.prob-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: rgba(255,255,255,0.7);
    width: 110px;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: var(--fill, #00c8ff);
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.prob-pct {
    font-size: 0.82rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: rgba(255,255,255,0.8);
    width: 48px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Severity Pill ── */
.severity-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.75rem;
}

/* ── Info Cards ── */
.info-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.75rem;
}
.info-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(180,210,255,0.5);
    margin-bottom: 0.5rem;
}
.info-card-body {
    font-size: 0.88rem;
    color: rgba(220,235,255,0.75);
    line-height: 1.6;
    font-weight: 300;
}

/* ── Model Badge ── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(100,0,255,0.08);
    border: 1px solid rgba(120,60,255,0.25);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.8rem;
    color: #a78bfa;
    font-weight: 500;
    margin-bottom: 1rem;
}

/* ── Section Title ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: rgba(180,210,255,0.4);
    margin: 1.5rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* ── Warning Box ── */
.warning-box {
    background: rgba(255,180,0,0.06);
    border: 1px solid rgba(255,180,0,0.25);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.82rem;
    color: rgba(255,200,80,0.85);
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    margin: 0.75rem 0;
}

/* ── Sidebar Styling ── */
section[data-testid="stSidebar"] {
    background: #060b12 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown {
    color: rgba(180,210,255,0.7) !important;
}

/* ── Streamlit widget overrides ── */
.stFileUploader > div {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(0,200,255,0.2) !important;
    border-radius: 14px !important;
}
.stFileUploader > div:hover {
    border-color: rgba(0,200,255,0.45) !important;
}
div[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0078d4, #0050a0) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0090f0, #0065c0) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,120,212,0.35) !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8edf5 !important;
}
.stSlider > div > div { color: rgba(180,210,255,0.7) !important; }
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}
div[data-testid="metric-container"] label {
    color: rgba(180,210,255,0.5) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 10px !important;
    padding: 0.2rem !important;
    gap: 0.2rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: rgba(180,210,255,0.5) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,200,255,0.1) !important;
    color: #00c8ff !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
img { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

CLASS_INFO = {
    'Glioma': {
        'desc': 'Originates from glial cells in the brain or spine. Can range from slow-growing to highly aggressive. Represents ~33% of all brain tumors.',
        'color': '#FF4757', 'bg1': 'rgba(255,71,87,0.08)', 'bg2': 'rgba(255,71,87,0.03)',
        'glow': 'rgba(255,71,87,0.15)', 'severity': 'High Risk', 'sev_color': '#FF4757', 'sev_bg': 'rgba(255,71,87,0.12)',
        'icon': '🔴', 'action': 'Immediate neurosurgical consultation recommended.'
    },
    'Meningioma': {
        'desc': 'Arises from the meninges surrounding the brain and spinal cord. Most are benign but can cause symptoms due to compression.',
        'color': '#FFA502', 'bg1': 'rgba(255,165,2,0.08)', 'bg2': 'rgba(255,165,2,0.03)',
        'glow': 'rgba(255,165,2,0.15)', 'severity': 'Moderate Risk', 'sev_color': '#FFA502', 'sev_bg': 'rgba(255,165,2,0.12)',
        'icon': '🟡', 'action': 'Neurology follow-up and monitoring recommended.'
    },
    'Pituitary': {
        'desc': 'Forms in the pituitary gland at the base of the brain. Most are non-cancerous adenomas that may affect hormone production.',
        'color': '#2ED573', 'bg1': 'rgba(46,213,115,0.08)', 'bg2': 'rgba(46,213,115,0.03)',
        'glow': 'rgba(46,213,115,0.15)', 'severity': 'Low–Moderate Risk', 'sev_color': '#2ED573', 'sev_bg': 'rgba(46,213,115,0.12)',
        'icon': '🟢', 'action': 'Endocrinology and neurology evaluation advised.'
    },
    'No Tumor': {
        'desc': 'No tumor detected. Brain tissue appears within normal limits on MRI scan. No abnormal mass or lesion identified.',
        'color': '#1E90FF', 'bg1': 'rgba(30,144,255,0.08)', 'bg2': 'rgba(30,144,255,0.03)',
        'glow': 'rgba(30,144,255,0.15)', 'severity': 'Normal', 'sev_color': '#1E90FF', 'sev_bg': 'rgba(30,144,255,0.12)',
        'icon': '🔵', 'action': 'Routine follow-up as clinically indicated.'
    },
}

PROB_COLORS = ['#FF4757', '#FFA502', '#2ED573', '#1E90FF']


# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_choice):
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
    from tensorflow.keras import Model, Input

    paths_b0 = ['best_model.keras', 'best_brain_tumor_b0.keras', 'best_glioma_model.h5', 'brain_tumor_final.tflite']
    paths_b3 = ['best_brain_tumor_b3.h5', 'best_model.keras']

    if model_choice == 'EfficientNetB0':
        for p in paths_b0:
            if os.path.exists(p):
                m = tf.keras.models.load_model(p)
                return m, f'EfficientNetB0 (loaded from {p})', True

        # Build fresh functional model (demo mode — random weights)
        base = EfficientNetB0(input_shape=(224,224,3), include_top=False, weights='imagenet')
        inp = Input(shape=(224,224,3))
        x = base(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        out = tf.keras.layers.Dense(4, activation='softmax')(x)
        m = Model(inp, out)
        return m, 'EfficientNetB0 (ImageNet weights — no fine-tuned .keras found)', False

    else:  # EfficientNetB3
        for p in paths_b3:
            if os.path.exists(p):
                m = tf.keras.models.load_model(p)
                return m, f'EfficientNetB3 (loaded from {p})', True

        base = EfficientNetB3(input_shape=(224,224,3), include_top=False, weights='imagenet')
        inp = Input(shape=(224,224,3))
        x = base(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        out = tf.keras.layers.Dense(4, activation='softmax')(x)
        m = Model(inp, out)
        return m, 'EfficientNetB3 (ImageNet weights — no fine-tuned .h5 found)', False


# ─── Image Processing ─────────────────────────────────────────────────────────
def crop_center(img, crop_size=180):
    h, w = img.shape[:2]
    sx = max(0, w//2 - crop_size//2)
    sy = max(0, h//2 - crop_size//2)
    return img[sy:sy+crop_size, sx:sx+crop_size]


def preprocess(pil_img):
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img_np = np.array(pil_img.convert('RGB'))
    original = img_np.copy()
    cropped  = crop_center(img_np)
    resized  = cv2.resize(cropped, (224, 224))
    arr = preprocess_input(resized.astype(np.float32))
    return original, resized, np.expand_dims(arr, 0)


# ─── Grad-CAM ────────────────────────────────────────────────────────────────
def make_gradcam(img_array, model, conf_threshold=0.75):
    import tensorflow as tf

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # ── Try proper layer-based Grad-CAM ──
    target_layers = ['top_conv', 'top_activation', 'block7a_project_bn',
                     'block7a_project_conv', 'block6d_project_bn']

    def try_nested_gradcam(base_name, layer_name):
        try:
            base    = model.get_layer(base_name)
            clayer  = base.get_layer(layer_name)
            cmodel  = tf.keras.Model(inputs=base.input, outputs=clayer.output)

            ci = tf.keras.Input(shape=clayer.output.shape[1:])
            x  = ci
            for lyr in model.layers[2:]:
                x = lyr(x)
            clf = tf.keras.Model(ci, x)

            with tf.GradientTape() as tape:
                co = cmodel(img_tensor)
                tape.watch(co)
                preds = clf(co)
                idx   = tf.argmax(preds[0])
                loss  = preds[:, idx]

            grads   = tape.gradient(loss, co)
            weights = tf.reduce_mean(grads, axis=(0,1,2))
            cam     = tf.nn.relu(tf.reduce_sum(weights * co[0], axis=-1)).numpy()
            cam     = cv2.resize(cam, (224, 224))
            if cam.max() > 0:
                cam /= cam.max()
            return cam
        except Exception:
            return None

    for base_name in ['efficientnetb0', 'efficientnetb3']:
        for lname in target_layers:
            result = try_nested_gradcam(base_name, lname)
            if result is not None:
                return result

    # ── Try direct model Grad-CAM (Functional API) ──
    for lname in target_layers:
        try:
            clayer = model.get_layer(lname)
            gm = tf.keras.Model(inputs=model.input,
                                outputs=[clayer.output, model.output])
            with tf.GradientTape() as tape:
                co, preds = gm(img_tensor)
                tape.watch(co)
                idx  = tf.argmax(preds[0])
                loss = preds[:, idx]
            grads   = tape.gradient(loss, co)
            weights = tf.reduce_mean(grads, axis=(0,1,2))
            cam     = tf.nn.relu(tf.reduce_sum(weights * co[0], axis=-1)).numpy()
            cam     = cv2.resize(cam, (224, 224))
            if cam.max() > 0:
                cam /= cam.max()
            return cam
        except Exception:
            continue

    # ── Fallback: Saliency map ──
    try:
        img_var = tf.Variable(img_tensor)
        with tf.GradientTape() as tape:
            preds = model(img_var)
            idx   = tf.argmax(preds[0])
            loss  = preds[:, idx]
        grads   = tape.gradient(loss, img_var)
        saliency = tf.reduce_mean(tf.abs(grads), axis=-1)[0].numpy()
        saliency = cv2.resize(saliency, (224, 224))
        saliency = cv2.GaussianBlur(saliency, (15,15), 0)
        if saliency.max() > 0:
            saliency /= saliency.max()
        return saliency
    except Exception:
        return np.zeros((224, 224), dtype=np.float32)


def suppress_edges(heatmap, margin=0.15):
    h, w = heatmap.shape
    mask = np.zeros_like(heatmap)
    m = int(margin * h)
    mask[m:h-m, m:w-m] = 1
    return heatmap * mask


def get_bbox(heatmap, threshold=0.5):
    h8 = np.uint8(255 * heatmap)
    _, thr = cv2.threshold(h8, int(255*threshold), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 200]
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(c)


def render_gradcam_figure(original_rgb, heatmap, pred_class, confidence,
                          show_bbox=True, alpha=0.4, colormap='jet'):
    info  = CLASS_INFO[pred_class]
    color_hex = info['color']
    color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16)/255 for i in (0,2,4))

    hm_resized = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
    cmap = plt.get_cmap(colormap)
    hm_colored = (cmap(hm_resized)[:,:,:3] * 255).astype(np.uint8)
    overlay    = (original_rgb * (1 - alpha) + hm_colored * alpha).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor('#080d14')

    titles = ['Original MRI', 'Grad-CAM Heatmap', f'Overlay — {pred_class}']
    imgs   = [original_rgb, hm_colored, overlay]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, color='#8ab4f8', fontsize=11, fontweight='bold',
                     fontfamily='DejaVu Sans', pad=10)
        ax.set_facecolor('#080d14')
        for spine in ax.spines.values():
            spine.set_edgecolor((1, 1, 1, 0.08))
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Draw bounding box on overlay panel
    if show_bbox and pred_class != 'No Tumor':
        bbox = get_bbox(hm_resized / 255 if hm_resized.max() > 1 else hm_resized)
        if bbox:
            x, y, w, h = bbox
            rect = patches.Rectangle((x, y), w, h,
                                      linewidth=2.5, edgecolor=color_rgb,
                                      facecolor='none', linestyle='-')
            axes[2].add_patch(rect)
            axes[2].text(x, y - 8, f'{pred_class} ({confidence:.1%})',
                         color=color_hex, fontsize=8.5, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#080d14',
                                   edgecolor=color_hex, alpha=0.85))

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='#080d14', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf


# ─── Confidence Bar HTML ──────────────────────────────────────────────────────
def prob_bar_html(label, prob, color, is_top=False):
    pct = f"{prob*100:.1f}%"
    weight = 700 if is_top else 400
    opacity = 1.0 if is_top else 0.6
    return f"""
    <div class="prob-row">
        <span class="prob-label" style="font-weight:{weight};opacity:{opacity}">{label}</span>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{prob*100:.1f}%;--fill:{color};background:{color};
                 {'box-shadow:0 0 8px ' + color + '66' if is_top else ''}">
            </div>
        </div>
        <span class="prob-pct" style="opacity:{opacity}">{pct}</span>
    </div>"""


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0 1rem">
        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
                    color:#ffffff;letter-spacing:-0.01em">⚙️ Configuration</div>
        <div style="font-size:0.75rem;color:rgba(180,210,255,0.4);margin-top:0.2rem">
            Model & analysis settings
        </div>
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Model Architecture",
        ["EfficientNetB0", "EfficientNetB3"],
        help="B0: 96.07% val accuracy, lightweight.\nB3: 95.45% val accuracy, larger."
    )

    st.markdown('<div class="section-title">Grad-CAM Settings</div>', unsafe_allow_html=True)

    show_gradcam = st.toggle("Enable Grad-CAM", value=True)
    show_bbox    = st.toggle("Show Bounding Box", value=True)

    grad_alpha = st.slider("Heatmap Opacity", 0.1, 0.9, 0.4, 0.05,
                           help="Transparency of Grad-CAM overlay")
    colormap   = st.selectbox("Colormap", ["jet", "hot", "plasma", "inferno", "RdYlGn"],
                              help="Grad-CAM heatmap color scheme")
    bbox_thresh = st.slider("BBox Threshold", 0.3, 0.8, 0.5, 0.05,
                            help="Heatmap activation threshold for bounding box")

    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <div class="info-card-title">Project</div>
        <div class="info-card-body">4th Semester Macro Project — Brain Tumor Multi-Class Detection using Deep Learning & Transfer Learning.</div>
    </div>
    <div class="info-card">
        <div class="info-card-title">Classes</div>
        <div class="info-card-body">🔴 Glioma &nbsp; 🟡 Meningioma<br>🟢 Pituitary &nbsp; 🔵 No Tumor</div>
    </div>
    <div class="info-card">
        <div class="info-card-title">Dataset</div>
        <div class="info-card-body">5,604 training images · 1,604 test images across 4 classes.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        ⚕️ <span>For academic/research purposes only. Not a substitute for professional medical diagnosis.</span>
    </div>
    """, unsafe_allow_html=True)


# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🧠 Deep Learning · Medical Imaging · XAI</div>
    <h1 class="hero-title">NeuroScan AI</h1>
    <p class="hero-subtitle">Brain Tumor Multi-Class Detection · EfficientNet Transfer Learning · Grad-CAM Explainability</p>
    <div class="hero-stats">
        <div class="hero-stat">
            <span class="hero-stat-value">96.07%</span>
            <span class="hero-stat-label">B0 Val Accuracy</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-value">95.45%</span>
            <span class="hero-stat-label">B3 Val Accuracy</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-value">4</span>
            <span class="hero-stat-label">Tumor Classes</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-value">5,604</span>
            <span class="hero-stat-label">Training Images</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Load Model ───────────────────────────────────────────────────────────────
with st.spinner(f"Loading {model_choice} model..."):
    model, model_label, is_finetuned = load_model(model_choice)

status_color = "#2ED573" if is_finetuned else "#FFA502"
status_text  = "Fine-tuned Weights Loaded" if is_finetuned else "ImageNet Weights (place .keras/.h5 file here)"
st.markdown(f"""
<div class="model-badge">
    <span style="width:8px;height:8px;border-radius:50%;background:{status_color};
                 box-shadow:0 0 6px {status_color};display:inline-block"></span>
    {model_label} &nbsp;·&nbsp;
    <span style="color:{status_color}">{status_text}</span>
</div>
""", unsafe_allow_html=True)


# ─── Main Layout ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.55], gap="large")

with left_col:
    st.markdown('<div class="section-title">Upload MRI Scan</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your brain MRI image here",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Uploaded MRI", use_container_width=True)

        st.markdown(f"""
        <div class="info-card" style="margin-top:0.75rem">
            <div class="info-card-title">Image Info</div>
            <div class="info-card-body">
                📄 {uploaded.name}<br>
                📐 {pil_img.size[0]} × {pil_img.size[1]} px<br>
                🎨 {pil_img.mode} · {round(uploaded.size/1024,1)} KB
            </div>
        </div>
        """, unsafe_allow_html=True)

        analyze_btn = st.button("🔬 Analyze MRI Scan", use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:2rem 1rem;color:rgba(180,210,255,0.35);">
            <div style="font-size:3rem;margin-bottom:0.75rem">🧠</div>
            <div style="font-size:0.88rem;line-height:1.6">
                Upload a brain MRI scan (JPG, PNG)<br>
                to begin AI-powered tumor detection
            </div>
        </div>
        """, unsafe_allow_html=True)
        analyze_btn = False

    # ── Model Comparison Panel ──
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.metric("EfficientNetB0", "96.07%", "Val Acc")
        st.metric("Parameters", "~4.05M", "")
    with m2:
        st.metric("EfficientNetB3", "95.45%", "Val Acc")
        st.metric("Parameters", "~10.78M", "")


# ─── Results Column ───────────────────────────────────────────────────────────
with right_col:
    if uploaded and analyze_btn:
        # ── Inference ──
        with st.spinner("Running inference..."):
            t0 = time.time()
            original_rgb, display_rgb, arr = preprocess(pil_img)
            preds      = model.predict(arr, verbose=0)[0]
            pred_idx   = int(np.argmax(preds))
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(preds[pred_idx])
            inf_time   = time.time() - t0

        info = CLASS_INFO[pred_class]

        # ── Diagnosis Banner ──
        st.markdown(f"""
        <div class="diagnosis-banner"
             style="--accent:{info['color']};--bg1:{info['bg1']};--bg2:{info['bg2']};--glow:{info['glow']}">
            <div class="diagnosis-label">🔬 AI Diagnosis Result</div>
            <div class="diagnosis-name">{info['icon']} {pred_class}</div>
            <div class="diagnosis-confidence">
                Confidence: <span>{confidence:.1%}</span>
                &nbsp;·&nbsp; Inference: {inf_time*1000:.0f}ms
            </div>
            <div class="severity-pill"
                 style="background:{info['sev_bg']};color:{info['sev_color']};
                        border:1px solid {info['sev_color']}33">
                ⚡ {info['severity']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Low confidence warning
        if confidence < 0.75:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ <span>Low confidence ({confidence:.1%}). The model is uncertain — 
                Grad-CAM may be less reliable. Consider re-scanning or consulting a specialist.</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Tabs ──
        tab1, tab2, tab3 = st.tabs(["📊 Probabilities", "🔥 Grad-CAM", "📋 Clinical Info"])

        with tab1:
            st.markdown('<div class="result-card" style="--accent:#00c8ff">', unsafe_allow_html=True)
            st.markdown('<div class="info-card-title">Class Probability Distribution</div>', unsafe_allow_html=True)

            bars_html = ""
            sorted_idx = np.argsort(preds)[::-1]
            for i in sorted_idx:
                bars_html += prob_bar_html(
                    CLASS_NAMES[i], preds[i],
                    PROB_COLORS[i], is_top=(i == pred_idx)
                )
            st.markdown(bars_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Matplotlib bar chart
            fig, ax = plt.subplots(figsize=(7, 2.8))
            fig.patch.set_facecolor('#0d1520')
            ax.set_facecolor('#0d1520')
            bars = ax.bar(CLASS_NAMES, preds * 100,
                          color=[PROB_COLORS[i] if i == pred_idx
                                 else PROB_COLORS[i] + '55' for i in range(4)],
                          width=0.55, zorder=3)
            ax.set_ylim(0, 115)
            ax.set_ylabel('Probability (%)', color=(180/255, 210/255, 255/255, 0.6), fontsize=9)            
            ax.tick_params(colors=(180/255, 210/255, 255/255, 0.6), labelsize=9)
            ax.set_facecolor('#0d1520')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.yaxis.grid(True, color=(1, 1, 1, 0.05), linewidth=0.8)
            ax.set_axisbelow(True)
            for bar, p in zip(bars, preds):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 2, f'{p*100:.1f}%',
                        ha='center', va='bottom',
                        color=rgba(255,255,255,0.8), fontsize=8.5, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with tab2:
            if show_gradcam:
                with st.spinner("Computing Grad-CAM..."):
                    heatmap = make_gradcam(arr, model)
                    heatmap = suppress_edges(heatmap)

                if pred_class != 'No Tumor':
                    buf = render_gradcam_figure(
                        display_rgb, heatmap, pred_class, confidence,
                        show_bbox=show_bbox, alpha=grad_alpha, colormap=colormap
                    )
                    st.image(buf, use_container_width=True)

                    # Activation stats
                    act_mean  = float(heatmap.mean())
                    act_max   = float(heatmap.max())
                    focus_pct = float((heatmap > 0.5).mean() * 100)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Peak Activation", f"{act_max:.3f}")
                    c2.metric("Mean Activation", f"{act_mean:.3f}")
                    c3.metric("Focus Area", f"{focus_pct:.1f}%")

                    st.markdown(f"""
                    <div class="info-card" style="margin-top:0.75rem">
                        <div class="info-card-title">How to read this</div>
                        <div class="info-card-body">
                            🔴 <b>Red/Yellow regions</b> — highest model attention (most influential for prediction)<br>
                            🔵 <b>Blue/Dark regions</b> — low attention<br>
                            🟩 <b>Green box</b> — tumor localization estimate (≥{bbox_thresh:.0%} activation threshold)<br>
                            Layer used: <code>top_conv</code> / <code>block7a_project_bn</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="text-align:center;padding:2rem;color:rgba(180,210,255,0.4)">
                        <div style="font-size:2rem">🔵</div>
                        <div style="margin-top:0.5rem;font-size:0.88rem">
                            No Grad-CAM generated for 'No Tumor' predictions.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Enable Grad-CAM in the sidebar to view heatmaps.")

        with tab3:
            st.markdown(f"""
            <div class="result-card" style="--accent:{info['color']}">
                <div class="info-card-title">Tumor Description</div>
                <div class="info-card-body" style="font-size:0.92rem">{info['desc']}</div>
            </div>
            <div class="result-card" style="--accent:{info['color']}">
                <div class="info-card-title">⚕️ Recommended Action</div>
                <div class="info-card-body" style="font-size:0.92rem">{info['action']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Per-class breakdown
            st.markdown('<div class="section-title">All Class Details</div>', unsafe_allow_html=True)
            for cname, cinfo in CLASS_INFO.items():
                with st.expander(f"{cinfo['icon']} {cname} — {cinfo['severity']}"):
                    st.markdown(f"""
                    <div style="font-size:0.88rem;color:rgba(220,235,255,0.75);line-height:1.7">
                        {cinfo['desc']}<br><br>
                        <b>Recommended:</b> {cinfo['action']}
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("""
            <div class="warning-box" style="margin-top:1rem">
                ⚕️ <span><b>Medical Disclaimer:</b> This AI tool is for research and educational 
                purposes only. Always consult a qualified medical professional for diagnosis and treatment decisions.</span>
            </div>
            """, unsafe_allow_html=True)

    elif not uploaded:
        # ── Landing state ──
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;min-height:420px;text-align:center;
                    padding:2rem;opacity:0.5">
            <div style="font-size:5rem;margin-bottom:1.5rem;
                        filter:drop-shadow(0 0 30px rgba(0,200,255,0.3))">🧠</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;
                        font-weight:700;color:#ffffff;margin-bottom:0.5rem">
                Upload an MRI to Begin
            </div>
            <div style="font-size:0.88rem;color:rgba(180,210,255,0.6);max-width:300px;line-height:1.7">
                The model will classify the tumor type and generate a Grad-CAM heatmap 
                showing which brain regions influenced the prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded and not analyze_btn:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:rgba(180,210,255,0.4)">
            <div style="font-size:2.5rem;margin-bottom:1rem">👆</div>
            <div style="font-size:0.92rem">Click <b style="color:rgba(180,210,255,0.7)">
            Analyze MRI Scan</b> to run the model</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1.5rem 0;border-top:1px solid rgba(255,255,255,0.06);
            text-align:center;color:rgba(180,210,255,0.3);font-size:0.78rem;line-height:2">
    <b style="color:rgba(180,210,255,0.5)">NeuroScan AI</b> · 4th Semester Macro Project ·
    EfficientNetB0 (96.07%) &amp; EfficientNetB3 (95.45%) · 
    Grad-CAM Explainability · TensorFlow 2.20 · Built with Streamlit<br>
    ⚕️ For academic &amp; research purposes only — not a clinical diagnostic tool
</div>
""", unsafe_allow_html=True)
