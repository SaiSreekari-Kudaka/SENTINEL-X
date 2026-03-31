
import streamlit as st
import cv2
import numpy as np
import os
import json
import hashlib
import shutil
import time
import uuid
from datetime import datetime
from collections import deque
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# ── Optional heavy deps (graceful fallback) ──────────────────────────────────
try:
    from deepface import DeepFace
    from sklearn.metrics.pairwise import cosine_similarity
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & PATHS
# ─────────────────────────────────────────────────────────────────────────────
FACE_SIZE        = (200, 200)
USERS_FILE       = "users.json"
INTRUDER_MODEL   = "intruder_lbph.yml"

DIRS = {
    "gallery"           : "gallery",
    "intruder_faces"    : "intruder_faces",
    "normal_faces"      : "normal_faces",
    "snap_intruder"     : "snap_intruder",
    "snap_safe"         : "snap_safe",
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# Age / Gender model paths (Caffe)
AGE_PROTO    = "age_deploy.prototxt"
AGE_MODEL    = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

AGE_BUCKETS  = ['(0-2)','(4-6)','(8-12)','(15-20)',
                '(25-32)','(38-43)','(48-53)','(60-100)']
GENDER_LIST  = ['Male', 'Female']
MODEL_MEAN   = (78.426, 87.768, 114.895)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SENTINEL-X",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;900&display=swap');

:root {
  --bg-deep:      #060a0f;
  --bg-panel:     #0c1118;
  --bg-card:      #111820;
  --accent:       #00e5ff;
  --accent2:      #ff3c5f;
  --accent3:      #39ff14;
  --text-primary: #ddeeff;
  --text-muted:   #5a7fa0;
  --border:       rgba(0,229,255,0.15);
  --glow:         0 0 20px rgba(0,229,255,0.25);
  --glow-red:     0 0 20px rgba(255,60,95,0.35);
}

html, body, [class*="css"] {
  font-family: 'Exo 2', 'Rajdhani', sans-serif;
  background-color: var(--bg-deep);
  color: var(--text-primary);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #070d14 0%, #0a1520 100%) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Main area ── */
.main .block-container { padding-top: 1.2rem; padding-bottom: 3rem; }

/* ── Headings ── */
h1,h2,h3,h4 { font-family: 'Rajdhani', sans-serif !important; letter-spacing: 2px; }
h1 { color: var(--accent) !important; text-shadow: 0 0 24px rgba(0,229,255,0.5); font-weight:700; }
h2 { color: var(--text-primary) !important; font-weight:600; }
h3 { color: var(--accent) !important; font-size:1.1rem !important; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #001f2e 60%, #00303f);
  border: 1px solid var(--accent);
  color: var(--accent) !important;
  font-family: 'Rajdhani', sans-serif;
  font-weight: 600;
  letter-spacing: 1.5px;
  border-radius: 4px;
  padding: 0.55em 1.6em;
  transition: all .2s;
  text-transform: uppercase;
}
.stButton > button:hover {
  background: var(--accent);
  color: var(--bg-deep) !important;
  box-shadow: var(--glow);
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stPasswordInput > div > div > input {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px;
  color: var(--text-primary) !important;
  font-family: 'Share Tech Mono', monospace;
}
.stTextInput > div > div > input:focus,
.stPasswordInput > div > div > input:focus {
  border-color: var(--accent) !important;
  box-shadow: var(--glow) !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] { background: var(--border); }

/* ── Selectbox / Radio ── */
.stSelectbox > div > div, .stRadio > div {
  background: var(--bg-card);
  border-radius: 4px;
}

/* ── Alerts ── */
.stAlert { border-radius: 6px; border-left: 4px solid var(--accent); }

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  box-shadow: var(--glow);
}
[data-testid="metric-container"] label { color: var(--text-muted) !important; font-family: 'Share Tech Mono'; font-size:.8rem; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Rajdhani'; font-size: 2rem; font-weight:700; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: var(--bg-panel); border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { color: var(--text-muted); font-family:'Rajdhani'; letter-spacing:1px; font-weight:600; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

/* ── Expander ── */
.streamlit-expanderHeader { font-family:'Rajdhani'; font-weight:600; color: var(--accent) !important; letter-spacing:1px; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Banner ── */
.sentinel-banner {
  background: linear-gradient(90deg, #00060a 0%, #001822 50%, #00060a 100%);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.6rem 2rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
}
.sentinel-banner::before {
  content: '';
  position: absolute; top:0; left:0; right:0; height:2px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent);
  animation: scan 3s linear infinite;
}
@keyframes scan { 0%{transform:translateX(-100%)} 100%{transform:translateX(100%)} }

.sentinel-title {
  font-family: 'Rajdhani', sans-serif;
  font-size: 2.6rem; font-weight:700;
  color: var(--accent);
  letter-spacing: 6px;
  text-shadow: 0 0 30px rgba(0,229,255,.6);
  margin:0;
}
.sentinel-sub {
  font-family: 'Share Tech Mono';
  font-size: .85rem;
  color: var(--text-muted);
  letter-spacing: 3px;
  margin-top: .3rem;
}

/* ── Status badge ── */
.badge {
  display: inline-block;
  padding: .25em .9em;
  border-radius: 3px;
  font-family: 'Share Tech Mono';
  font-size: .8rem;
  letter-spacing: 1px;
  font-weight: 600;
}
.badge-green  { background: rgba(57,255,20,.12); color:#39ff14; border:1px solid #39ff1455; }
.badge-red    { background: rgba(255,60,95,.12);  color:#ff3c5f; border:1px solid #ff3c5f55; }
.badge-yellow { background: rgba(255,200,0,.12);  color:#ffc800; border:1px solid #ffc80055; }
.badge-blue   { background: rgba(0,229,255,.12);  color:#00e5ff; border:1px solid #00e5ff55; }

/* ── Info card ── */
.info-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem 1.2rem;
  margin: .5rem 0;
}
.info-card b { color: var(--accent); }

/* ── Gallery image card ── */
.img-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: .5rem;
  text-align: center;
  font-family: 'Share Tech Mono';
  font-size:.75rem;
  color: var(--text-muted);
}

/* ── Login form container ── */
.login-wrap {
  max-width: 420px;
  margin: 6vh auto 0;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 2.5rem 2.8rem;
  box-shadow: var(--glow);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    default = {"admin": hash_pw("admin123")}
    with open(USERS_FILE, "w") as f:
        json.dump(default, f)
    return default

def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def ts_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def count_files(folder: str) -> int:
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder)
                if f.lower().endswith((".jpg",".jpeg",".png"))])

def get_images(folder: str):
    if not os.path.exists(folder):
        return []
    return sorted(
        [f for f in os.listdir(folder)
         if f.lower().endswith((".jpg",".jpeg",".png"))],
        reverse=True
    )

# ── Face cascade ──────────────────────────────────────────────────────────────
HAAR = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR)

def detect_faces(gray):
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

def crop_face_gray(img_bgr):
    """Return resized grayscale face or None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
    return cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)

# ── LBPH ─────────────────────────────────────────────────────────────────────
def get_lbph_data():
    imgs, labels = [], []
    for fname in os.listdir(DIRS["intruder_faces"]):
        if not fname.lower().endswith((".png",".jpg","jpeg")):
            continue
        p = os.path.join(DIRS["intruder_faces"], fname)
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is not None:
            imgs.append(cv2.resize(g, FACE_SIZE))
            labels.append(0)
    return imgs, labels

def train_lbph():
    imgs, labels = get_lbph_data()
    if not imgs:
        return False
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train(imgs, np.array(labels))
    rec.save(INTRUDER_MODEL)
    return True

@st.cache_resource
def load_lbph():
    if not os.path.exists(INTRUDER_MODEL):
        return None
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(INTRUDER_MODEL)
    return rec

# ── DeepFace embeddings ───────────────────────────────────────────────────────
@st.cache_resource
def load_deepface_embeddings():
    if not DEEPFACE_AVAILABLE:
        return {"Intruder": [], "Normal": []}
    def embed(folder, tag):
        result = []
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            try:
                vec = DeepFace.represent(
                    img_path=p, model_name="Facenet512",
                    enforce_detection=False)[0]["embedding"]
                result.append((np.array(vec), tag))
            except Exception:
                pass
        return result
    return {
        "Intruder": embed(DIRS["intruder_faces"], "Intruder"),
        "Normal":   embed(DIRS["normal_faces"],   "Normal"),
    }

# ── Age / Gender Caffe models ─────────────────────────────────────────────────
@st.cache_resource
def load_age_gender():
    if not (os.path.exists(AGE_MODEL) and os.path.exists(GENDER_MODEL)):
        return None, None
    return (cv2.dnn.readNet(AGE_MODEL, AGE_PROTO),
            cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO))

def predict_age_gender(face_bgr, age_net, gender_net):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(face_bgr, (227, 227)),
        1.0, (227, 227), MODEL_MEAN, swapRB=False)
    gender_net.setInput(blob)
    g = GENDER_LIST[gender_net.forward()[0].argmax()]
    age_net.setInput(blob)
    a = AGE_BUCKETS[age_net.forward()[0].argmax()]
    return g, a

def age_gender_overlay(face_bgr):
    age_net, gender_net = load_age_gender()
    if age_net is None:
        return "N/A", "N/A"
    return predict_age_gender(face_bgr, age_net, gender_net)

# ── Snapshot helpers ─────────────────────────────────────────────────────────
def save_snapshot(img_bgr, is_intruder: bool) -> str:
    folder = DIRS["snap_intruder"] if is_intruder else DIRS["snap_safe"]
    path   = os.path.join(folder, f"snap_{ts_now()}_{uuid.uuid4().hex[:6]}.jpg")
    cv2.imwrite(path, img_bgr)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────
for k, v in {
    "logged_in"       : False,
    "username"        : "",
    "users"           : load_users(),
    "alert_count"     : 0,
    "last_status"     : "IDLE",
    "signup_mode"     : False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN / SIGNUP PAGE
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown("""
    <div style='text-align:center; margin-top:4vh; margin-bottom:2rem;'>
      <div style='font-family:Rajdhani; font-size:3rem; font-weight:700;
                  color:#00e5ff; letter-spacing:8px;
                  text-shadow: 0 0 30px rgba(0,229,255,.6);'>
        SENTINEL-X
      </div>
      <div style='font-family:Share Tech Mono; color:#5a7fa0;
                  letter-spacing:3px; font-size:.85rem; margin-top:.4rem;'>
        ADVANCED INTRUDER DETECTION SYSTEM
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        mode = st.radio("", ["Login", "Create Account"],
                        horizontal=True, label_visibility="collapsed")

        if mode == "Login":
            with st.form("login_form"):
                st.markdown("##### ACCESS CREDENTIALS")
                uname = st.text_input("Username", placeholder="username")
                pw    = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("AUTHENTICATE", use_container_width=True)
            if submitted:
                if (uname in st.session_state.users and
                        st.session_state.users[uname] == hash_pw(pw)):
                    st.session_state.logged_in = True
                    st.session_state.username  = uname
                    st.success(f"Access granted. Welcome, **{uname.upper()}**.")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("⚠️ Invalid credentials.")
        else:
            with st.form("signup_form"):
                st.markdown("##### NEW OPERATOR REGISTRATION")
                nu = st.text_input("New Username", placeholder="choose a username")
                np1 = st.text_input("Password", type="password", placeholder="••••••••")
                np2 = st.text_input("Confirm Password", type="password", placeholder="repeat password")
                reg = st.form_submit_button("REGISTER", use_container_width=True)
            if reg:
                if not nu or not np1:
                    st.warning("All fields required.")
                elif np1 != np2:
                    st.error("Passwords do not match.")
                elif nu in st.session_state.users:
                    st.warning("Username already exists.")
                else:
                    st.session_state.users[nu] = hash_pw(np1)
                    save_users(st.session_state.users)
                    st.success("Account created. Switch to Login to continue.")

        st.markdown("""
        <div style='text-align:center; margin-top:2rem;
                    font-family:Share Tech Mono; font-size:.75rem; color:#2a4a6a;'>
          DEFAULT: admin / admin123
        </div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem;'>
      <div style='font-family:Rajdhani; font-size:1.6rem; font-weight:700;
                  color:#00e5ff; letter-spacing:4px;
                  text-shadow:0 0 16px rgba(0,229,255,.5);'>
        SENTINEL-X
      </div>
      <div style='font-family:Share Tech Mono; font-size:.7rem;
                  color:#2a5a7a; letter-spacing:2px;'>
        v2.0 HYBRID ENGINE
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("NAVIGATION", [
        "📊  Dashboard",
        "🎥  Live Detection",
        "🧠  Train Model",
        "🖼️  Gallery",
        "👤  Age & Gender",
    ], label_visibility="collapsed")

    st.markdown("---")
    # Runtime stats in sidebar
    st.markdown(f"""
    <div style='font-family:Share Tech Mono; font-size:.78rem; color:#2a5a7a; line-height:2;'>
      OPERATOR : <span style='color:#00e5ff'>{st.session_state.username.upper()}</span><br>
      INTRUDER DB : <span style='color:#ff3c5f'>{count_files(DIRS["intruder_faces"])} FACES</span><br>
      SAFE DB : <span style='color:#39ff14'>{count_files(DIRS["normal_faces"])} FACES</span><br>
      ALERTS : <span style='color:#ffc800'>{count_files(DIRS["snap_intruder"])}</span><br>
      MODEL : <span style='color:#00e5ff'>{"LBPH+DF" if DEEPFACE_AVAILABLE else "LBPH"}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🔓 LOGOUT", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if "Dashboard" in page:
    st.markdown("""
    <div class='sentinel-banner'>
      <p class='sentinel-title'>📊 DASHBOARD</p>
      <p class='sentinel-sub'>SYSTEM OVERVIEW · REAL-TIME METRICS · SENTINEL-X v2.0</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric row ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intruder Faces (DB)",  count_files(DIRS["intruder_faces"]))
    c2.metric("Safe Faces (DB)",      count_files(DIRS["normal_faces"]))
    c3.metric("Intruder Alerts",      count_files(DIRS["snap_intruder"]))
    c4.metric("Safe Snapshots",       count_files(DIRS["snap_safe"]))

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🛡️ SYSTEM STATUS")
        model_ok = os.path.exists(INTRUDER_MODEL)
        age_ok   = os.path.exists(AGE_MODEL) and os.path.exists(GENDER_MODEL)
        st.markdown(f"""
        <div class='info-card'>
          Detection Engine &nbsp;
          <span class='badge {"badge-green" if DEEPFACE_AVAILABLE else "badge-yellow"}'>
            {"HYBRID (LBPH + DeepFace)" if DEEPFACE_AVAILABLE else "LBPH ONLY"}
          </span>
        </div>
        <div class='info-card'>
          LBPH Model File &nbsp;
          <span class='badge {"badge-green" if model_ok else "badge-red"}'>
            {"LOADED" if model_ok else "NOT TRAINED"}
          </span>
        </div>
        <div class='info-card'>
          Age/Gender Model &nbsp;
          <span class='badge {"badge-green" if age_ok else "badge-yellow"}'>
            {"READY" if age_ok else "MODELS NOT FOUND"}
          </span>
        </div>
        <div class='info-card'>
          DeepFace (Facenet512) &nbsp;
          <span class='badge {"badge-green" if DEEPFACE_AVAILABLE else "badge-red"}'>
            {"AVAILABLE" if DEEPFACE_AVAILABLE else "NOT INSTALLED"}
          </span>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("### 📂 RECENT INTRUDER ALERTS")
        snaps = get_images(DIRS["snap_intruder"])[:6]
        if snaps:
            g_cols = st.columns(3)
            for i, fn in enumerate(snaps):
                p = os.path.join(DIRS["snap_intruder"], fn)
                with g_cols[i % 3]:
                    st.image(p, use_container_width=True)
        else:
            st.info("No alerts yet. Start Live Detection to begin monitoring.")

    st.markdown("---")
    st.markdown("### 📖 QUICK-START GUIDE")
    st.markdown("""
    <div class='info-card'>
      <b>STEP 1</b> — Go to <b>Train Model</b>. Upload or capture intruder face images, then press <b>Train</b>.
    </div>
    <div class='info-card'>
      <b>STEP 2</b> — (Optional) Add known-safe faces to improve accuracy.
    </div>
    <div class='info-card'>
      <b>STEP 3</b> — Open <b>Live Detection</b>. The hybrid engine (LBPH + DeepFace) will classify every face.
    </div>
    <div class='info-card'>
      <b>STEP 4</b> — Review snapshots in <b>Gallery</b> or run <b>Age & Gender</b> analysis on saved intruder shots.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LIVE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
elif "Live Detection" in page:
    st.markdown("""
    <div class='sentinel-banner'>
      <p class='sentinel-title'>🎥 LIVE DETECTION</p>
      <p class='sentinel-sub'>HYBRID ENGINE · LBPH + DEEPFACE FACENET512</p>
    </div>
    """, unsafe_allow_html=True)

    if not os.path.exists(INTRUDER_MODEL):
        st.warning("⚠️ LBPH model not trained yet. Go to **Train Model** first.")

    # ── Detection parameters ──
    with st.sidebar.expander("⚙️ DETECTION PARAMS", expanded=True):
        lbph_thresh     = st.slider("LBPH Confidence Threshold", 30, 150, 80,
                                    help="Lower = stricter. Match if confidence < threshold")
        df_thresh       = st.slider("DeepFace Similarity %", 40, 95, 58,
                                    help="Match if cosine similarity ≥ this value")
        snap_interval   = st.slider("Snapshot Interval (s)", 1, 10, 3)
        show_conf       = st.checkbox("Show Confidence Values", True)
        enable_deepface = st.checkbox("Enable DeepFace Layer", DEEPFACE_AVAILABLE,
                                      disabled=not DEEPFACE_AVAILABLE)

    status_ph = st.empty()
    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("#### LIVE STATS")
        stat_ph      = st.empty()
        intruder_ph  = st.empty()

    class HybridProcessor(VideoProcessorBase):
        found             = False
        confidence_lbph   = None
        confidence_df     = None
        last_snap         = 0
        face_count        = 0
        method_used       = "—"
        age_info          = "—"
        gender_info       = "—"

        def __init__(self):
            self.recognizer = load_lbph()
            self.embeddings = load_deepface_embeddings() if enable_deepface else {"Intruder":[],"Normal":[]}
            self.age_net, self.gender_net = load_age_gender()
            self._snap_interval = snap_interval
            self._lbph_thresh   = lbph_thresh
            self._df_thresh     = df_thresh
            self._show_conf     = show_conf
            self._age_history   = deque(maxlen=7)
            self._gender_history= deque(maxlen=7)

        def _hybrid_classify(self, face_gray, face_bgr):
            """Return (label, conf_str, method)"""
            label, conf_str, method = "Unknown", "", "—"

            # ── Layer 1: LBPH ──
            if self.recognizer is not None:
                lbl, conf = self.recognizer.predict(cv2.resize(face_gray, FACE_SIZE))
                HybridProcessor.confidence_lbph = conf
                if lbl == 0 and conf < self._lbph_thresh:
                    label = "Intruder"
                    conf_str = f"LBPH:{conf:.0f}"
                    method   = "LBPH"
                elif conf >= self._lbph_thresh:
                    label = "Safe"
                    conf_str = f"LBPH:{conf:.0f}"
                    method   = "LBPH"

            # ── Layer 2: DeepFace (confirm / override) ──
            if enable_deepface and (label == "Unknown" or self.recognizer is None):
                try:
                    probe = np.array(
                        DeepFace.represent(face_bgr, model_name="Facenet512",
                                           enforce_detection=False)[0]["embedding"]
                    ).reshape(1, -1)
                    best_score, best_tag = 0, None
                    for tag in ["Intruder", "Normal"]:
                        for emb, _ in self.embeddings[tag]:
                            score = cosine_similarity(probe, emb.reshape(1,-1))[0][0] * 100
                            if score > best_score:
                                best_score, best_tag = score, tag
                    HybridProcessor.confidence_df = best_score
                    if best_score >= self._df_thresh:
                        label    = "Intruder" if best_tag == "Intruder" else "Safe"
                        conf_str = f"DF:{best_score:.0f}%"
                        method   = "DeepFace" if method == "—" else method+"+DF"
                except Exception:
                    pass

            return label, conf_str, method

        def recv(self, frame):
            img  = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)

            HybridProcessor.found      = False
            HybridProcessor.face_count = len(faces)

            for (x, y, w, h) in faces:
                face_gray = gray[y:y+h, x:x+w]
                face_bgr  = img [y:y+h, x:x+w]

                label, conf_str, method = self._hybrid_classify(face_gray, face_bgr)
                HybridProcessor.method_used = method

                # ── Age / Gender ──
                if self.age_net is not None:
                    try:
                        gender, age = predict_age_gender(
                            cv2.resize(face_bgr,(227,227)),
                            self.age_net, self.gender_net)
                        self._gender_history.append(gender)
                        HybridProcessor.gender_info = max(
                            set(self._gender_history),
                            key=self._gender_history.count)
                        HybridProcessor.age_info = age
                    except Exception:
                        pass

                # ── Draw ──
                COLORS = {"Intruder":(0,0,255),"Safe":(57,255,20),"Unknown":(0,229,255)}
                col = COLORS.get(label,(255,255,255))
                cv2.rectangle(img, (x,y), (x+w,y+h), col, 2)

                tag_parts = [label]
                if self._show_conf and conf_str:
                    tag_parts.append(conf_str)
                if HybridProcessor.age_info != "—":
                    tag_parts.append(f"{HybridProcessor.gender_info} {HybridProcessor.age_info}")

                cv2.putText(img, " | ".join(tag_parts), (x, max(y-10,14)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, col, 1, cv2.LINE_AA)

                if label == "Intruder":
                    HybridProcessor.found = True
                    now = time.time()
                    if now - HybridProcessor.last_snap > self._snap_interval:
                        save_snapshot(img, is_intruder=True)
                        HybridProcessor.last_snap = now
                elif label == "Safe":
                    now = time.time()
                    if now - HybridProcessor.last_snap > self._snap_interval + 1:
                        save_snapshot(img, is_intruder=False)
                        HybridProcessor.last_snap = now

            # ── Border ──
            border_col = (0,0,255) if HybridProcessor.found else (57,255,20)
            img = cv2.copyMakeBorder(img, 8,8,8,8,
                                     cv2.BORDER_CONSTANT, value=border_col)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    with col1:
        ctx = webrtc_streamer(
            key="sentinel_live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=HybridProcessor,
            media_stream_constraints={"video":True,"audio":False},
            async_processing=True,
        )

    if ctx.state.playing:
        is_threat = HybridProcessor.found
        if is_threat:
            status_ph.error("🚨 **INTRUDER DETECTED** — ALERT TRIGGERED")
        else:
            status_ph.success("✅ **AREA SECURE** — No threats detected")

        lbph_c = HybridProcessor.confidence_lbph
        df_c   = HybridProcessor.confidence_df
        stat_ph.markdown(f"""
        <div class='info-card' style='font-family:Share Tech Mono; font-size:.82rem;'>
          FACES DETECTED : <b>{HybridProcessor.face_count}</b><br>
          METHOD : <b>{HybridProcessor.method_used}</b><br>
          LBPH CONF : <b>{f"{lbph_c:.1f}" if lbph_c else "—"}</b><br>
          DEEPFACE SIM : <b>{f"{df_c:.1f}%" if df_c else "—"}</b><br>
          GENDER : <b>{HybridProcessor.gender_info}</b><br>
          AGE : <b>{HybridProcessor.age_info}</b><br>
          ALERTS SAVED : <b>{count_files(DIRS["snap_intruder"])}</b>
        </div>
        """, unsafe_allow_html=True)

        badge = '<span class="badge badge-red">⬤ INTRUDER</span>' if is_threat \
                else '<span class="badge badge-green">⬤ SECURE</span>'
        intruder_ph.markdown(badge, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────
elif "Train Model" in page:
    st.markdown("""
    <div class='sentinel-banner'>
      <p class='sentinel-title'>🧠 TRAIN MODEL</p>
      <p class='sentinel-sub'>REGISTER FACES · BUILD RECOGNITION DATABASE</p>
    </div>
    """, unsafe_allow_html=True)

    tab_intr, tab_safe = st.tabs(["🔴  Intruder Faces", "🟢  Safe/Known Faces"])

    for tab, folder, label, prefix in [
        (tab_intr, DIRS["intruder_faces"], "Intruder", "intruder"),
        (tab_safe, DIRS["normal_faces"],   "Safe",     "safe"),
    ]:
        with tab:
            col_a, col_b = st.columns(2)

            # ───────── Upload Section ─────────
            with col_a:
                st.markdown(f"#### 📤 Upload {label} Images")
                uploads = st.file_uploader(
                    f"Upload {label} face images",
                    type=["jpg","jpeg","png"],
                    accept_multiple_files=True,
                    key=f"upload_{prefix}",
                )

                if uploads:
                    saved = 0
                    for u in uploads:
                        try:
                            img_pil = Image.open(u).convert("RGB")
                            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                            face_g = crop_face_gray(img_bgr)

                            if face_g is not None:
                                fn = os.path.join(
                                    folder,
                                    f"{prefix}_{ts_now()}_{uuid.uuid4().hex[:4]}.png"
                                )
                                cv2.imwrite(fn, face_g)
                                saved += 1
                        except Exception as e:
                            st.warning(f"Error processing image: {e}")

                    if saved:
                        st.success(f"✅ {saved} face(s) saved to {label} database.")
                    else:
                        st.warning("⚠️ No faces detected in uploaded images.")

            # ───────── Camera Capture ─────────
            with col_b:
                st.markdown(f"#### 📷 Capture from Camera")
                cam_img = st.camera_input(f"Capture {label} face", key=f"cam_{prefix}")

                if cam_img:
                    try:
                        img_pil = Image.open(cam_img).convert("RGB")
                        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                        face_g = crop_face_gray(img_bgr)

                        if face_g is not None:
                            fn = os.path.join(folder, f"{prefix}_{ts_now()}.png")
                            cv2.imwrite(fn, face_g)
                            st.success(f"✅ {label} face captured and saved.")
                        else:
                            st.warning("⚠️ No face detected. Try again.")
                    except Exception as e:
                        st.error(f"Camera error: {e}")

            # ───────── Count + Preview ─────────
            st.markdown(f"**Faces in {label} database:** `{count_files(folder)}`")

            faces_in_db = get_images(folder)[:9]
            if faces_in_db:
                with st.expander(f"Preview {label} database ({count_files(folder)} images)"):
                    g = st.columns(3)
                    for i, fn in enumerate(faces_in_db):
                        with g[i % 3]:
                            st.image(
                                os.path.join(folder, fn),
                                use_container_width=True,
                                caption=fn[:18]
                            )

    # ─────────────────────────────────────────
    # TRAIN SECTION
    # ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 TRAIN / RETRAIN MODEL")

    c1, c2 = st.columns(2)

    # ───────── LBPH TRAINING ─────────
    with c1:
        if st.button("⚡ TRAIN LBPH MODEL", use_container_width=True):

            # 🔥 Check OpenCV face module BEFORE training
            if not hasattr(cv2, "face"):
                st.error("❌ OpenCV 'face' module not found.")
                st.info("👉 Install correct version: pip install opencv-contrib-python==4.8.0.76")
            else:
                with st.spinner("Training LBPH model... please wait ⏳"):
                    try:
                        result = train_lbph()

                        if result:
                            st.cache_resource.clear()
                            st.success("✅ LBPH model trained successfully. Cache refreshed.")
                        else:
                            st.warning("⚠️ No faces found. Upload faces first.")

                    except Exception as e:
                        st.error(f"❌ Training failed: {e}")

    # ───────── DEEPFACE RELOAD ─────────
    with c2:
        if DEEPFACE_AVAILABLE:
            if st.button("🔄 RELOAD DEEPFACE EMBEDDINGS", use_container_width=True):
                with st.spinner("Reloading embeddings..."):
                    st.cache_resource.clear()
                    st.success("✅ DeepFace embeddings will reload on next run.")
        else:
            st.info("ℹ️ DeepFace not installed. Only LBPH will be used.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: GALLERY
# ─────────────────────────────────────────────────────────────────────────────
elif "Gallery" in page:
    st.markdown("""
    <div class='sentinel-banner'>
      <p class='sentinel-title'>🖼️ GALLERY</p>
      <p class='sentinel-sub'>SNAPSHOT ARCHIVE · INTRUDER EVIDENCE LOG</p>
    </div>
    """, unsafe_allow_html=True)

    tab_a, tab_b, tab_c = st.tabs([
        f"🔴  Intruder Alerts ({count_files(DIRS['snap_intruder'])})",
        f"🟢  Safe Snapshots ({count_files(DIRS['snap_safe'])})",
        f"📂  Gallery Captures ({count_files(DIRS['gallery'])})",
    ])

    def render_gallery(folder, key_prefix):
        files = get_images(folder)
        if not files:
            st.info("No images in this folder yet.")
            return
        # Bulk delete
        if st.button(f"🗑️ CLEAR ALL ({len(files)} images)",
                     key=f"clear_{key_prefix}"):
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)
            st.success("Cleared.")
            st.rerun()
        cols = st.columns(4)
        for i, fn in enumerate(files):
            p = os.path.join(folder, fn)
            with cols[i % 4]:
                st.image(p, use_container_width=True)
                short = fn[:16] + "…" if len(fn) > 16 else fn
                st.caption(short)
                with open(p,"rb") as f:
                    st.download_button("⬇️", f, file_name=fn,
                                       key=f"dl_{key_prefix}_{fn}",
                                       use_container_width=True)
                if st.button("✕", key=f"del_{key_prefix}_{fn}",
                             use_container_width=True):
                    os.remove(p)
                    st.rerun()

    with tab_a:
        render_gallery(DIRS["snap_intruder"], "intr")
    with tab_b:
        render_gallery(DIRS["snap_safe"], "safe")
    with tab_c:
        render_gallery(DIRS["gallery"], "gal")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: AGE & GENDER
# ─────────────────────────────────────────────────────────────────────────────
elif "Age" in page:
    st.markdown("""
    <div class='sentinel-banner'>
      <p class='sentinel-title'>👤 AGE & GENDER ANALYSIS</p>
      <p class='sentinel-sub'>DEEP LEARNING DEMOGRAPHIC PROFILING</p>
    </div>
    """, unsafe_allow_html=True)

    age_net, gender_net = load_age_gender()
    if age_net is None:
        st.error("""
        ⚠️ Age/Gender Caffe models not found.

        Download and place these 4 files in the project root:
        - `age_net.caffemodel` + `age_deploy.prototxt`
        - `gender_net.caffemodel` + `gender_deploy.prototxt`
        """)
        st.stop()

    # Source selection
    source = st.radio("Analyze images from:",
                      ["Intruder Alerts", "Safe Snapshots", "Upload Image"],
                      horizontal=True)

    images_to_analyze = []
    if source == "Intruder Alerts":
        images_to_analyze = [os.path.join(DIRS["snap_intruder"], f)
                             for f in get_images(DIRS["snap_intruder"])]
    elif source == "Safe Snapshots":
        images_to_analyze = [os.path.join(DIRS["snap_safe"], f)
                             for f in get_images(DIRS["snap_safe"])]
    else:
        uploaded = st.file_uploader("Upload image(s)", type=["jpg","jpeg","png"],
                                    accept_multiple_files=True, key="ag_upload")
        for u in (uploaded or []):
            pil = Image.open(u)
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            tmp = os.path.join(DIRS["gallery"], f"ag_{ts_now()}_{uuid.uuid4().hex[:4]}.jpg")
            cv2.imwrite(tmp, bgr)
            images_to_analyze.append(tmp)

    if not images_to_analyze:
        st.info("No images to analyze. Detect intruders first or upload images above.")
        st.stop()

    if st.button(f"🔍 ANALYZE {len(images_to_analyze)} IMAGES", use_container_width=True):
        results = []
        prog = st.progress(0)
        for idx, path in enumerate(images_to_analyze):
            prog.progress((idx+1)/len(images_to_analyze))
            img = cv2.imread(path)
            if img is None:
                continue
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)
            for (x, y, w, h) in faces:
                face_bgr = img[y:y+h, x:x+w]
                try:
                    gender, age = predict_age_gender(
                        cv2.resize(face_bgr,(227,227)), age_net, gender_net)
                    results.append((path, gender, age, face_bgr))
                except Exception:
                    pass
        prog.empty()

        if not results:
            st.warning("No faces detected in selected images.")
        else:
            st.success(f"✅ Analyzed {len(results)} face(s) across {len(images_to_analyze)} image(s).")
            st.markdown("---")
            for path, gender, age, face_bgr in results:
                try:
                    dt_str = os.path.basename(path).split("_")[1]
                    dt = datetime.strptime(dt_str, "%Y%m%d").strftime("%Y-%m-%d")
                except Exception:
                    dt = "—"

                cc1, cc2 = st.columns([1, 3])
                with cc1:
                    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                    st.image(rgb, use_container_width=True)
                with cc2:
                    gender_badge = ('badge-blue' if gender == 'Male' else 'badge-red')
                    st.markdown(f"""
                    <div class='info-card'>
                      <b>FILE</b> : {os.path.basename(path)}<br>
                      <b>DATE</b> : {dt}<br>
                      <b>GENDER</b> : <span class='badge {gender_badge}'>{gender}</span><br>
                      <b>AGE RANGE</b> : <span class='badge badge-blue'>{age}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; margin-top:3rem; padding: 1.5rem;
            border-top:1px solid rgba(0,229,255,0.1);
            font-family:Share Tech Mono; font-size:.75rem; color:#1a3a5a;'>
  SENTINEL-X v2.0 &nbsp;·&nbsp; HYBRID DETECTION ENGINE &nbsp;·&nbsp;
  LBPH + DEEPFACE FACENET512 &nbsp;·&nbsp; © 2025
</div>
""", unsafe_allow_html=True)
