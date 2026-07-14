import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import predict_review

st.title("🛡️ FakeGuard - AI Fake Review Detector")
st.write("Paste a product review below to check if it's likely genuine or AI-generated.")

review_text = st.text_area("Review text:", height=150)

if st.button("Analyze Review"):
    if review_text.strip() == "":
        st.warning("Please enter a review first.")
    else:
        result = predict_review(review_text)
        
        if result['label'] == "Fake":
            st.error(f"🚨 Likely FAKE ({result['confidence']}% confidence)")
        else:
            st.success(f"✅ Likely REAL ({result['confidence']}% confidence)")
        
        st.subheader("Top words influencing this prediction:")
        for word, coef in result['top_words']:
            direction = "→ Fake" if coef > 0 else "→ Real"
            st.write(f"**{word}** ({direction}, weight: {round(coef, 3)})")