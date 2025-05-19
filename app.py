import joblib
import streamlit as st

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_sentiment(text):
    # Transform the input text using the TF-IDF vectorizer
    X_input = vectorizer.transform([text])
    # Predict the sentiment (returns "Positive" or "Negative" as a string)
    prediction = model.predict(X_input)[0]
    return prediction

# Quick console test
sample_text = "i love it"
print("Predicted Sentiment:", predict_sentiment(sample_text))

# Streamlit app
st.title("E-commerce Sentiment Analysis")
st.write("Enter a product review, and I'll predict the sentiment!")

user_input = st.text_area("Enter a review:")
if st.button("Analyze Sentiment"):
    if user_input:
        # Get the raw prediction string from the model
        prediction = predict_sentiment(user_input)

        # If the model returns "Positive" or "Negative"
        if prediction == "Positive":
            st.success("Sentiment: Positive 😊")
        else:
            st.error("Sentiment: Negative 😠")
