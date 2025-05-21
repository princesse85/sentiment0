import streamlit as st
import pickle

st.markdown(
    """
    <style>
    /* Style for the text area */
    div.stTextArea > div > textarea {
        font-size: 16px;
        padding: 12px;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    /* Style for the button */
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        transition: background-color 0.3s ease;
        justify-content: center;
    }
    div.stButton > button:hover {
        background-color:rgba(69, 80, 160, 0.34);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load the model
#with open('sentiment_model.pkl', 'rb') as f:
   # sentiment_model = pickle.load(f)
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

## App Layout and Title
st.title("  Ecommerce Customer Reviews Sentiment Analysis App  ")

st.write("""
        Welcome to our Ecommerce Customer Reviews Analysis App. 
        This simple tool analyzes customer reviews to show you the sentiment content of the review. 
        It uses machine learning to quickly highlight positive and negative feedback, helping you make better business decisions.""")

# Text input
user_input = st.text_area("Input The Text That You Want Analyzed Down Below: ", height=100)


# Prediction button
if st.button("Predict"):
    # Transform user input to TF-IDF features
    input_features = tfidf_vectorizer.transform([user_input])
    #  Predict sentiment
    prediction = sentiment_model.predict(input_features)

    # Display the result
    st.write(f"The Text inputed is a ", prediction[0] , "Sentiment")
