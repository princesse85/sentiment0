import streamlit as st


# 🌿 Custom CSS
st.markdown(
    """
    <style>
    div.stTextArea > div > textarea {
        font-size: 16px;
        padding: 12px;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: rgba(69, 80, 160, 0.34);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 📦 Load model and vectorizer
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.pkl")

model = load_model()
vectorizer = load_vectorizer()

# 📋 App Layout
st.title("🛍️ Ecommerce Customer Reviews Sentiment Analysis App")

st.write("""
Welcome to our Ecommerce Customer Reviews Analysis App. 
This tool analyzes customer feedback to detect **positive** or **negative** sentiment using machine learning.
""")

# 📝 User input
user_input = st.text_area("✍️ Enter a customer review below:", height=100)

# 🔍 Prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😠"
        st.success(f"Predicted sentiment: **{sentiment}**")


    # Display the result
    st.write(f"The Text inputed is a ", prediction[0] , "Sentiment")
