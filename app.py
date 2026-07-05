import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import pickle
import textwrap
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# PAGE CONFIG


st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS — light dashboard theme


st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F1F4FB;
    background-image:
        radial-gradient(circle at 8% 8%, rgba(99,102,241,.10) 0%, transparent 40%),
        radial-gradient(circle at 92% 15%, rgba(6,182,212,.10) 0%, transparent 42%),
        radial-gradient(circle at 50% 100%, rgba(236,72,153,.06) 0%, transparent 45%);
    background-attachment: fixed;
}

body, .stMarkdown p {
    color: #1F2937;
    line-height: 1.6;
}

h1, h2, h3, h4 { color: #0B1120; letter-spacing: -.01em; }

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Sidebar: dark panel against light app for contrast */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #111827 0%, #1E1B4B 100%);
    border-right: 1px solid #1F2937;
}
section[data-testid="stSidebar"] * { color: #E5E7EB !important; }
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,.12) !important;
    margin: 16px 0 !important;
}
section[data-testid="stSidebar"] img {
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,.15);
    box-shadow: 0 4px 14px rgba(0,0,0,.3);
}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #9CA3AF !important;
}

/* Top header bar */
.header-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, #FFFFFF 60%, #F5F7FF 100%);
    border-radius: 18px;
    padding: 22px 30px;
    margin-bottom: 22px;
    box-shadow: 0 4px 18px rgba(17,24,39,.07);
    border: 1px solid #EEF0F5;
    border-left: 4px solid #4F46E5;
    position: relative;
}
.header-title {
    font-size: 27px;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #111827, #4F46E5);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.header-sub {
    font-size: 14px;
    color: #6B7280;
    margin-top: 4px;
}
.header-badge {
    background: #EEF2FF;
    color: #4F46E5;
    font-size: 12px;
    font-weight: 700;
    padding: 6px 14px;
    border-radius: 999px;
    border: 1px solid #E0E7FF;
    display: flex;
    align-items: center;
    gap: 6px;
}
.header-badge .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 0 0 rgba(16,185,129,.5);
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(16,185,129,.5); }
    70%  { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
    100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #9CA3AF; }

/* Expander */
details {
    border-radius: 14px !important;
    border: 1px solid #EEF0F5 !important;
    background: #FFFFFF !important;
    box-shadow: 0 2px 12px rgba(17,24,39,.05);
}
summary {
    font-weight: 600 !important;
    color: #111827 !important;
}

/* Download button accent */
.stDownloadButton>button {
    background: #111827 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: .2s ease;
}
.stDownloadButton>button:hover {
    background: #1F2937 !important;
    box-shadow: 0 6px 16px rgba(17,24,39,.25);
}

/* Real panel containers (st.container(border=True, key=...)) */
.st-key-panel_input, .st-key-panel_result,
.st-key-panel_stats, .st-key-panel_words, .st-key-panel_history {
    background: linear-gradient(165deg, #FFFFFF 0%, #FAFBFF 100%) !important;
    border-radius: 18px !important;
    padding: 22px !important;
    box-shadow: 0 1px 2px rgba(17,24,39,.04), 0 8px 20px rgba(17,24,39,.06) !important;
    border: 1px solid #EEF0F5 !important;
    position: relative;
    transition: box-shadow .25s ease, transform .25s ease;
    animation: rise .35s ease both;
}
.st-key-panel_input:hover, .st-key-panel_result:hover,
.st-key-panel_stats:hover, .st-key-panel_words:hover, .st-key-panel_history:hover {
    box-shadow: 0 2px 4px rgba(17,24,39,.05), 0 16px 32px rgba(17,24,39,.11) !important;
    transform: translateY(-3px);
}
.st-key-panel_input::before, .st-key-panel_result::before,
.st-key-panel_stats::before, .st-key-panel_words::before, .st-key-panel_history::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 18px 18px 0 0;
    background: linear-gradient(90deg, #4F46E5, #06B6D4);
    opacity: .55;
}

@keyframes rise {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.panel-title {
    font-size: 15px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid #F3F4F6;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -.01em;
}

.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    min-height: 260px;
    color: #9CA3AF;
    text-align: center;
    gap: 8px;
    border: 1.5px dashed #E5E7EB;
    border-radius: 14px;
    background: #FAFBFF;
}
.empty-state .emoji {
    font-size: 40px;
    animation: float 2.4s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}

textarea:focus {
    border: 1px solid #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,.15) !important;
}

textarea {
    font-size: 15px !important;
    border-radius: 12px !important;
    background: #F9FAFB !important;
    border: 1px solid #E5E7EB !important;
}

label, .stMarkdown p, .stMarkdown li { color: #374151; }

.stButton>button {
    width: 100%;
    height: 46px;
    font-size: 14px;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    background: linear-gradient(135deg, #4F46E5, #6366F1);
    color: white;
    transition: .2s ease;
    position: relative;
    overflow: hidden;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #4338CA, #4F46E5);
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(79,70,229,.35);
}
.stButton>button:active { transform: translateY(0); }

/* Secondary (non-primary) buttons via nth-of-type in a row look identical by default;
   style Clear/Sample buttons a bit lighter using data-testid on their container class */
div[data-testid="column"]:nth-of-type(2) .stButton>button,
div[data-testid="column"]:nth-of-type(3) .stButton>button,
div[data-testid="column"]:nth-of-type(4) .stButton>button {
    background: #F3F4F6;
    color: #374151;
}
div[data-testid="column"]:nth-of-type(2) .stButton>button:hover,
div[data-testid="column"]:nth-of-type(3) .stButton>button:hover,
div[data-testid="column"]:nth-of-type(4) .stButton>button:hover {
    background: #E5E7EB;
    box-shadow: none;
}

/* Result badge */
.badge-result {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 20px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 16px;
    box-shadow: 0 4px 10px rgba(17,24,39,.05);
}
.badge-result.positive {
    background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
    color: #047857;
    border: 1px solid #A7F3D0;
}
.badge-result.negative {
    background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
    color: #B91C1C;
    border: 1px solid #FECACA;
}

/* Confidence bar (linear, not circular this time) */
.conf-track {
    width: 100%;
    height: 12px;
    border-radius: 999px;
    background: #EEF0F5;
    box-shadow: inset 0 1px 3px rgba(17,24,39,.08);
    overflow: hidden;
    margin: 6px 0 4px 0;
}
.conf-fill {
    height: 100%;
    border-radius: 999px;
    transition: width .4s ease;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
}
.conf-fill.positive { background: linear-gradient(90deg, #10B981, #34D399); }
.conf-fill.negative { background: linear-gradient(90deg, #EF4444, #F87171); }

.stat-number {
    font-size: 28px;
    font-weight: 800;
    background: linear-gradient(135deg, #111827, #4F46E5);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    line-height: 1.2;
}
.stat-label {
    font-size: 11px;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: .06em;
    font-weight: 600;
}

.chip {
    display: inline-block;
    padding: 5px 13px;
    margin: 3px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    transition: transform .15s ease, box-shadow .15s ease;
    box-shadow: 0 2px 6px rgba(17,24,39,.04);
}
.chip:hover { transform: translateY(-2px); box-shadow: 0 6px 14px rgba(17,24,39,.1); }
.chip.pos { background: linear-gradient(135deg, #ECFDF5, #D1FAE5); color: #047857; border: 1px solid #A7F3D0; }
.chip.neg { background: linear-gradient(135deg, #FEF2F2, #FEE2E2); color: #B91C1C; border: 1px solid #FECACA; }

.history-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 10px;
    border-bottom: 1px solid #F3F4F6;
    border-left: 3px solid transparent;
    font-size: 13px;
    border-radius: 8px;
    transition: background .15s ease, border-color .15s ease;
}
.history-row.positive { border-left-color: #A7F3D0; }
.history-row.negative { border-left-color: #FECACA; }
.history-row:hover { background: #F9FAFB; }
.history-row:last-child { border-bottom: none; }
.history-text { color: #374151; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; padding-right: 10px; }
.history-tag { font-weight: 600; font-size: 12px; }
.history-tag.positive { color: #047857; }
.history-tag.negative { color: #B91C1C; }
.history-conf { color: #9CA3AF; font-size: 12px; width: 60px; text-align: right; }

div[data-testid="stAlert"] { border-radius: 12px; }

hr { border-color: #E5E7EB !important; }

.footer {
    margin-top: 24px;
    text-align: center;
    color: #9CA3AF;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# LOAD MODEL

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sentiment_model.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

MAX_LEN = 200  # matches maxlen used in training (pad_sequences(..., maxlen=200))

try:
    model = load_model()
    tokenizer = load_tokenizer()
except Exception as e:
    st.error("Unable to load model files. Make sure 'sentiment_model.keras' and 'tokenizer.pkl' are in the same folder as this app.")
    st.exception(e)
    st.stop()

# SESSION STATE


if "review_input" not in st.session_state:
    st.session_state.review_input = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

SAMPLE_POSITIVE = (
    "This movie was absolutely wonderful. The acting was superb, "
    "the story kept me engaged from start to finish, and I would "
    "happily watch it again."
)
SAMPLE_NEGATIVE = (
    "This movie was a complete waste of time. The plot made no "
    "sense, the acting was terrible, and I regret watching it."
)
POSITIVE_WORDS = [
    "great", "excellent", "amazing", "wonderful", "love", "loved",
    "best", "fantastic", "brilliant", "enjoyed", "beautiful", "perfect",
    "good", "superb", "favorite"
]
NEGATIVE_WORDS = [
    "bad", "worst", "terrible", "awful", "boring", "hate", "hated",
    "waste", "poor", "disappointing", "horrible", "dull", "annoying",
    "mess", "weak"
]

def clear_review():
    st.session_state.review_input = ""
    st.session_state.last_result = None
    st.session_state.history = []

def load_sample_positive():
    st.session_state.review_input = SAMPLE_POSITIVE

def load_sample_negative():
    st.session_state.review_input = SAMPLE_NEGATIVE


# SIDEBAR


with st.sidebar:
    st.image(
        "https://media.springernature.com/lw685/springer-static/image/chp%3A10.1007%2F978-3-032-08246-6_38/MediaObjects/664033_1_En_38_Fig1_HTML.png",
        width=80
    )
    st.markdown("### Sentiment Analyzer")
    st.caption("Embedding + GRU · TensorFlow")
    st.markdown("---")
    st.markdown("""
**How it works**

1. Type or paste a review
2. Click **Analyze**
3. See prediction, confidence & word breakdown
""")
    st.markdown("---")
    st.caption("Model trained on IMDB 50K movie reviews.")

# HEADER


st.markdown("""
<div class="header-bar">
    <div>
        <p class="header-title">🎬 Movie Review Sentiment Analyzer</p>
        <p class="header-sub">Positive or negative — powered by a GRU neural network</p>
    </div>
    <div class="header-badge"><span class="dot"></span> Model Ready</div>
</div>
""", unsafe_allow_html=True)

# MAIN LAYOUT: input (left) + result (right), side by side


left, right = st.columns([1, 1], gap="medium")

with left:
    with st.container(border=True, key="panel_input"):
        st.markdown('<p class="panel-title">📝 Your Review</p>', unsafe_allow_html=True)

        st.text_area(
            "Review text",
            height=200,
            placeholder="e.g. I really enjoyed this movie — the performances were great and the story stayed engaging all the way through.",
            key="review_input",
            label_visibility="collapsed"
        )

        b1, b2, b3, b4 = st.columns(4)
        analyze = b1.button("🚀 Analyze", use_container_width=True)
        b2.button("🗑 Clear", use_container_width=True, on_click=clear_review)
        b3.button("😊 Sample +", use_container_width=True, on_click=load_sample_positive)
        b4.button("😞 Sample −", use_container_width=True, on_click=load_sample_negative)

        review = st.session_state.review_input

        if analyze:
            if review.strip() == "":
                st.warning("⚠ Please enter a review first.")
            else:
                with st.spinner("Analyzing review..."):
                    time.sleep(0.4)
                    try:
                        sequence = tokenizer.texts_to_sequences([review])
                        padded = pad_sequences(sequence, maxlen=MAX_LEN)
                        probability = float(model.predict(padded, verbose=0)[0][0])
                    except Exception as e:
                        st.error("Prediction failed.")
                        st.exception(e)
                        st.stop()

                is_positive = probability > 0.5
                confidence = (probability if is_positive else 1 - probability) * 100

                st.session_state.last_result = {
                    "review": review,
                    "probability": probability,
                    "confidence": confidence,
                    "is_positive": is_positive,
                }
                st.session_state.history.insert(0, {
                    "Review": review[:50] + "..." if len(review) > 50 else review,
                    "is_positive": is_positive,
                    "Confidence": f"{confidence:.1f}%"
                })
                st.session_state.history = st.session_state.history[:8]

with right:
    result = st.session_state.last_result
    with st.container(border=True, key="panel_result"):
        st.markdown('<p class="panel-title">📊 Result</p>', unsafe_allow_html=True)

        if not result:
            st.markdown(textwrap.dedent("""
            <div class="empty-state">
                <div class="emoji">🎯</div>
                <div><b>No prediction yet</b></div>
                <div style="font-size:13px;">Enter a review on the left and click Analyze</div>
            </div>
            """), unsafe_allow_html=True)
        else:
            is_positive = result["is_positive"]
            probability = result["probability"]
            confidence = result["confidence"]
            pos_score = probability * 100
            neg_score = (1 - probability) * 100
            css_class = "positive" if is_positive else "negative"
            icon = "😊" if is_positive else "😞"
            label = "Positive Review" if is_positive else "Negative Review"

            st.markdown(textwrap.dedent(f"""
            <div class="badge-result {css_class}">{icon} {label}</div>
            <p style="font-size:13px; color:#6B7280; margin-bottom:2px;">Confidence</p>
            <div class="conf-track"><div class="conf-fill {css_class}" style="width:{confidence}%;"></div></div>
            <p style="font-size:13px; color:#111827; font-weight:700; margin-top:2px;">{confidence:.2f}%</p>
            """), unsafe_allow_html=True)

            st.write("")
            st.markdown(textwrap.dedent(f"""
            <p style="font-size:13px; color:#6B7280; margin-bottom:2px;">Positive vs Negative</p>
            <div class="conf-track" style="height:22px; display:flex;">
                <div style="width:{pos_score}%; background:#10B981; height:100%;"></div>
                <div style="width:{neg_score}%; background:#EF4444; height:100%;"></div>
            </div>
            <p style="font-size:12px; color:#6B7280; margin-top:6px;">
                😊 {pos_score:.1f}% &nbsp;·&nbsp; 😞 {neg_score:.1f}%
            </p>
            """), unsafe_allow_html=True)

            report = f"""SENTIMENT REPORT
==================
Prediction : {'POSITIVE' if is_positive else 'NEGATIVE'}
Confidence : {confidence:.2f}%
Positive   : {pos_score:.2f}%
Negative   : {neg_score:.2f}%

Review:
{result['review']}
"""
            st.download_button("📥 Download Report", data=report, file_name="sentiment_report.txt",
                                mime="text/plain", use_container_width=True)

st.write("")


# DASHBOARD GRID — replaces tabs with permanent cards


c1, c2, c3 = st.columns([1, 1, 1], gap="medium")

with c1:
    with st.container(border=True, key="panel_stats"):
        st.markdown('<p class="panel-title">🔢 Text Stats</p>', unsafe_allow_html=True)
        if review.strip() == "":
            st.markdown('<p style="color:#9CA3AF; font-size:13px;">Enter a review to see stats.</p>', unsafe_allow_html=True)
        else:
            words = review.split()
            stats = [("Words", len(words)), ("Characters", len(review)),
                      ("Sentences", review.count(".") + review.count("!") + review.count("?")),
                      ("Spaces", review.count(" "))]
            s1, s2 = st.columns(2)
            for i, (lbl, val) in enumerate(stats):
                target = s1 if i % 2 == 0 else s2
                target.markdown(f'<div class="stat-number">{val}</div><div class="stat-label">{lbl}</div><br>', unsafe_allow_html=True)

with c2:
    with st.container(border=True, key="panel_words"):
        st.markdown('<p class="panel-title">🔍 Sentiment Words</p>', unsafe_allow_html=True)
        if review.strip() == "":
            st.markdown('<p style="color:#9CA3AF; font-size:13px;">Enter a review to see detected words.</p>', unsafe_allow_html=True)
        else:
            lower = review.lower()
            pos_found = [w for w in POSITIVE_WORDS if w in lower]
            neg_found = [w for w in NEGATIVE_WORDS if w in lower]
            if not pos_found and not neg_found:
                st.markdown('<p style="color:#9CA3AF; font-size:13px;">No strong sentiment words detected — likely context-based.</p>', unsafe_allow_html=True)
            else:
                chips = "".join(f'<span class="chip pos">{w}</span>' for w in pos_found)
                chips += "".join(f'<span class="chip neg">{w}</span>' for w in neg_found)
                st.markdown(chips, unsafe_allow_html=True)

with c3:
    with st.container(border=True, key="panel_history"):
        st.markdown('<p class="panel-title">🕘 Recent Predictions</p>', unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown('<p style="color:#9CA3AF; font-size:13px;">No predictions yet.</p>', unsafe_allow_html=True)
        else:
            rows = ""
            for item in st.session_state.history:
                is_pos = item.get("is_positive", False)
                tag_class = "positive" if is_pos else "negative"
                tag_text = "Positive" if is_pos else "Negative"
                row_html = textwrap.dedent(f"""
                <div class="history-row {tag_class}">
                    <div class="history-text">{item.get('Review', '')}</div>
                    <div class="history-tag {tag_class}">{tag_text}</div>
                    <div class="history-conf">{item.get('Confidence', '—')}</div>
                </div>
                """)
                rows += row_html
            st.markdown(rows, unsafe_allow_html=True)


# MODEL INFO / ABOUT — collapsed by default, out of the way


st.write("")
with st.expander("🤖 Model & Project Details"):
    colA, colB = st.columns(2)
    with colA:
        st.markdown("""
**Architecture**
- Embedding layer (5,000 vocab, 128-dim)
- GRU (128 units, tanh)
- Dense (sigmoid)

**Input / Output**
- Raw text → tokenized, padded to 200 tokens
- Output: probability of positive sentiment
""")
    with colB:
        st.markdown("""
**About**
Trained on the IMDB 50K movie review dataset. Confidence
reflects the model's raw sigmoid output — short or sarcastic
reviews can be harder to classify correctly.
""")
    st.write("")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.caption("Positive example")
        st.code(SAMPLE_POSITIVE, language="text")
    with sc2:
        st.caption("Negative example")
        st.code(SAMPLE_NEGATIVE, language="text")

st.markdown("""
<div class="footer">Built with TensorFlow · Streamlit · Keras</div>
""", unsafe_allow_html=True)