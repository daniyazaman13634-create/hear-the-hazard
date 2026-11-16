import time
from datetime import datetime
from collections import deque
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

st.set_page_config(page_title="Hear the Hazard", page_icon="🔔", layout="wide")

# AudioSet/YAMNet class names and bilingual alert text.
ALERTS = {
    "doorbell": {"en": "DOORBELL DETECTED", "ur": "دروازے کی گھنٹی بج رہی ہے", "icon": "🔔", "color": "#2563eb",
                 "labels": ["Doorbell"]},
    "fire": {"en": "FIRE ALARM DETECTED", "ur": "آگ کا الارم بج رہا ہے", "icon": "🔥", "color": "#dc2626",
             "labels": ["Fire alarm", "Smoke detector, smoke alarm"]},
    "siren": {"en": "EMERGENCY SIREN DETECTED", "ur": "ایمرجنسی سائرن بج رہا ہے", "icon": "🚨", "color": "#b91c1c",
              "labels": ["Siren", "Civil defense siren"]},
    "smoke": {"en": "SMOKE ALARM DETECTED", "ur": "دھوئیں کا الارم بج رہا ہے", "icon": "⚠️", "color": "#ea580c",
              "labels": ["Smoke detector, smoke alarm"]},
    "glass": {"en": "GLASS BREAKING SOUND", "ur": "شیشہ ٹوٹنے کی آواز", "icon": "💥", "color": "#7c3aed",
              "labels": ["Glass", "Shatter"]},
}

st.markdown("""
<style>
.stApp {background: #f5f7fb;}
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {background:linear-gradient(120deg,#10243a,#174a67);padding:26px 30px;border-radius:20px;color:white;margin-bottom:18px;}
.hero h1 {font-size:2.15rem;margin:0;color:white;}
.hero p {color:#dbeafe;margin:6px 0 0;}
.panel {background:white;border:1px solid #e6eaf0;border-radius:16px;padding:18px 20px;min-height:125px;}
.alert {padding:18px;border-radius:16px;color:white;margin:10px 0;font-weight:700;}
.muted {color:#64748b;font-size:.9rem;}
.metric {font-size:1.7rem;font-weight:800;color:#10243a;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the sound recognition model…")
def load_model():
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = tf.keras.utils.get_file(
        "yamnet_class_map.csv",
        "https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv"
    )
    class_names = pd.read_csv(class_map_path)["display_name"].tolist()
    return model, class_names

def classify(audio, model, class_names):
    # YAMNet expects mono float32 audio sampled at 16 kHz.
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return []
    if np.max(np.abs(x)) > 1.0:
        x = x / max(np.max(np.abs(x)), 1e-8)
    scores, embeddings, spectrogram = model(x)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    top = np.argsort(mean_scores)[::-1][:5]
    return [(class_names[i], float(mean_scores[i])) for i in top]

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = deque(maxlen=50)
        self.latest_audio = None

    def recv(self, frame):
        arr = frame.to_ndarray()
        if arr.ndim == 2:
            arr = np.mean(arr, axis=0)
        self.frames.append((arr.astype(np.float32), frame.sample_rate))
        return frame

def map_alert(predictions, threshold, enabled):
    for category, item in ALERTS.items():
        if category not in enabled:
            continue
        for label in item["labels"]:
            for predicted, score in predictions:
                if predicted.lower() == label.lower() and score >= threshold:
                    return category, score
    return None, 0.0

def resample_to_16k(audio, sample_rate):
    if sample_rate == 16000:
        return audio.astype(np.float32)
    duration = len(audio) / float(sample_rate)
    target_len = max(1, int(duration * 16000))
    old_t = np.linspace(0, duration, num=len(audio), endpoint=False)
    new_t = np.linspace(0, duration, num=target_len, endpoint=False)
    return np.interp(new_t, old_t, audio).astype(np.float32)

if "history" not in st.session_state:
    st.session_state.history = []
if "last_alert" not in st.session_state:
    st.session_state.last_alert = 0.0

st.markdown("""
<div class="hero">
  <h1>🔔 Hear the Hazard</h1>
  <p>AI-powered sound recognition · Visual safety alerts · English + اردو</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Configure the sounds you want the system to watch for.")
    enabled = st.multiselect("Sound categories", list(ALERTS.keys()),
                             default=["doorbell", "fire", "siren", "smoke"])
    threshold = st.slider("Detection sensitivity", 0.05, 0.95, 0.25, 0.05,
                          help="Lower values are more sensitive but may cause more false alerts.")
    st.markdown("---")
    st.markdown("**Alert language**")
    bilingual = st.checkbox("Show English + Urdu", value=True)
    st.markdown("**Alert style**")
    alert_style = st.selectbox("Visual style", ["Banner", "Compact"])

c1, c2, c3 = st.columns([1.15, 1, 1])
with c1:
    st.markdown('<div class="panel"><div class="muted">MONITORING</div><div class="metric">● Live microphone</div><div class="muted">Allow microphone access when prompted</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="panel"><div class="muted">SENSITIVITY</div><div class="metric">{int(threshold*100)}%</div><div class="muted">Model score threshold</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="panel"><div class="muted">ALERTS THIS SESSION</div><div class="metric">{len(st.session_state.history)}</div><div class="muted">Recent recognized events</div></div>', unsafe_allow_html=True)

st.subheader("🎙️ Stage 1 — Listen")
st.write("Start the microphone stream below. Keep this page open while monitoring.")
ctx = webrtc_streamer(
    key="hazard-audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

st.subheader("🧠 Stage 2 — Recognize")
model_ready = False
try:
    model, class_names = load_model()
    model_ready = True
    st.success("YAMNet sound-classification model loaded.")
except Exception as e:
    st.error("The AI model could not load. Check your internet connection and installed dependencies.")
    st.caption(str(e))

latest_predictions = []
if ctx.state.playing and ctx.audio_processor and model_ready:
    processor = ctx.audio_processor
    if processor.frames:
        chunks = list(processor.frames)
        samples = []
        sr = chunks[-1][1]
        for chunk, chunk_sr in chunks[-10:]:
            if chunk_sr == sr:
                samples.append(chunk.reshape(-1))
        if samples:
            audio = np.concatenate(samples)
            audio = resample_to_16k(audio, sr)
            if len(audio) >= 8000:
                try:
                    latest_predictions = classify(audio, model, class_names)
                except Exception as e:
                    st.warning(f"Audio analysis issue: {e}")

if latest_predictions:
    top_label, top_score = latest_predictions[0]
    category, score = map_alert(latest_predictions, threshold, enabled)
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Top model predictions**")
        st.dataframe(pd.DataFrame(latest_predictions, columns=["Sound class", "Confidence"])
                     .assign(Confidence=lambda d: (d.Confidence * 100).round(1).astype(str) + "%"),
                     use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Detected sound**")
        if category:
            item = ALERTS[category]
            now = datetime.now().strftime("%H:%M:%S")
            if time.time() - st.session_state.last_alert > 4:
                st.session_state.history.insert(0, {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sound": category.title(),
                    "confidence": round(score * 100, 1),
                    "status": "Detected"
                })
                st.session_state.history = st.session_state.history[:30]
                st.session_state.last_alert = time.time()
            msg = f"{item['icon']} {item['en']}"
            urdu = f"<div dir='rtl' style='font-size:1.3rem;margin-top:8px'>{item['ur']}</div>" if bilingual else ""
            if alert_style == "Banner":
                st.markdown(f"<div class='alert' style='background:{item['color']}'><div style='font-size:1.4rem'>{msg}</div>{urdu}<div style='margin-top:10px'>Confidence: {score:.0%} · {now}</div></div>", unsafe_allow_html=True)
            else:
                st.warning(f"{msg} — {score:.0%} — {now}")
                if bilingual:
                    st.markdown(f"<div dir='rtl'>{item['ur']}</div>", unsafe_allow_html=True)
        else:
            st.info(f"No configured hazard above threshold. Top sound: {top_label} ({top_score:.0%})")
else:
    st.info("Waiting for enough microphone audio. Start monitoring and make a supported sound nearby to test.")

st.subheader("🚨 Stage 3 — Alert history")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
else:
    st.caption("No alerts detected yet.")

st.download_button("Download alert history (CSV)",
                   pd.DataFrame(st.session_state.history).to_csv(index=False).encode("utf-8"),
                   file_name="hear_the_hazard_alerts.csv", mime="text/csv")

st.markdown("---")
st.header("📚 How the science works")
tab1, tab2, tab3, tab4 = st.tabs(["Sound waves", "Frequency & amplitude", "FFT", "AI recognition"])
with tab1:
    st.write("Sound is a vibration that travels through a medium such as air. A microphone converts changing air pressure into an electrical/digital waveform.")
with tab2:
    st.write("**Frequency (Hz)** is the number of vibrations per second and is related to pitch. **Amplitude** describes signal strength and is related to loudness.")
with tab3:
    st.write("The Fast Fourier Transform (FFT) converts a waveform from the time domain into frequency components. Frequency patterns can help distinguish alarms, bells, and other sounds.")
with tab4:
    st.write("Hear the Hazard uses **YAMNet**, a pretrained deep-learning audio event classifier trained on AudioSet. It produces scores for sound classes. This prototype maps selected model classes to visual alerts; a locally collected Pakistani sound dataset and fine-tuning are recommended before real-world deployment.")

st.header("🇵🇰 Pakistan-focused development")
st.write("The prototype can be extended with locally recorded doorbells, school bells, mosque/public-address announcements, generator warnings, vehicle horns, and regional emergency sirens. Collect consented recordings from varied homes and public spaces, label them, and fine-tune a classifier. Do not treat this prototype as a certified fire or life-safety alarm.")

st.warning("Safety note: This is an educational prototype, not a certified life-safety device. Model predictions can be wrong or delayed. Keep certified smoke/fire alarms and other required safety systems active.")
