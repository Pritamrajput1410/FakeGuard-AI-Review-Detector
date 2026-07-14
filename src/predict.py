import joblib
from preprocess import clean_text
import os

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'lr_tfidf_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'tfidf_vectorizer.pkl'))

def predict_review(text, top_n=5):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0]
    
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    present_indices = vec.nonzero()[1]
    
    word_contributions = [(feature_names[i], coefs[i]) for i in present_indices]
    word_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    label = "Fake" if prediction == 1 else "Real"
    confidence = probability[1] if prediction == 1 else probability[0]
    
    return {
        'label': label,
        'confidence': round(confidence * 100, 2),
        'top_words': word_contributions[:top_n]
    }