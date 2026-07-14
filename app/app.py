import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import predict_review

# Page config - controls browser tab title, icon, and layout width
st.set_page_config(
    page_title="FakeGuard - Review Detector",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS for nicer styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🛡️ FakeGuard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Fake Review Detector</p>', unsafe_allow_html=True)

st.divider()

review_text = st.text_area(
    "Paste a product review below:",
    height=150,
    placeholder="e.g. 'This product is absolutely amazing! Best purchase ever!'"
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_clicked = st.button("🔍 Analyze Review", use_container_width=True)

if analyze_clicked:
    if review_text.strip() == "":
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_review(review_text)

        st.divider()

        # Result display with columns
        col1, col2 = st.columns([1, 1])
        with col1:
            if result['label'] == "Fake":
                st.error(f"🚨 **Likely FAKE**")
            else:
                st.success(f"✅ **Likely REAL**")
        with col2:
            st.metric("Confidence", f"{result['confidence']}%")

        # Confidence as a progress bar
        st.progress(result['confidence'] / 100)

        st.subheader("🔎 What influenced this prediction")
        for word, coef in result['top_words']:
            direction = "🔴 Fake signal" if coef > 0 else "🟢 Real signal"
            st.write(f"**{word}** — {direction} (weight: {round(coef, 3)})")

st.divider()
st.caption("Built with scikit-learn + Streamlit | Model: TF-IDF + Logistic Regression (89% accuracy)")