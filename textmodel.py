import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

@st.cache(allow_output_mutation=True)

def load_models():
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    logreg_selected = joblib.load('logreg_selected_model.pkl')
    neural_net = joblib.load('neural_net_model.pkl')
    nb_classifier = joblib.load('naive_bayes_model.pkl')
    selector_weights = joblib.load('selector.pkl')
    return tfidf_vectorizer, logreg_selected, neural_net, nb_classifier, selector_weights

def make_predictions(tfidf_vectorizer, logreg_selected, neural_net, nb_classifier, selector_weights, text):
    custom_text_tfidf = tfidf_vectorizer.transform([text])
    custom_text_selected = selector_weights.transform(custom_text_tfidf)
    logreg_pred = logreg_selected.predict(custom_text_selected)
    neural_net_pred = neural_net.predict(custom_text_selected)
    nb_pred = nb_classifier.predict(custom_text_selected)
    return logreg_pred[0], neural_net_pred[0], nb_pred[0]

# Streamlit app
st.title('Text Classification App')
st.markdown("# High School Subject Classification App by Faraz")
# Load models
tfidf_vectorizer, logreg_selected, neural_net, nb_classifier, selector_weights = load_models()

# Text input
text_input = st.text_area('Enter text here:', '')

# Make predictions when button is clicked
if st.button('Predict'):
    if text_input:
        logreg_pred, neural_net_pred, nb_pred = make_predictions(tfidf_vectorizer, logreg_selected, neural_net, nb_classifier, selector_weights, text_input)
        st.write("Logistic Regression Prediction:", logreg_pred)
        st.write("Neural Network Prediction:", neural_net_pred)
        st.write("Naive Bayes Prediction:", nb_pred)
    else:
        st.write("Please enter some text before predicting.")
