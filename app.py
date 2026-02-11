import streamlit as st
import pickle

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Twitter Sentiment Analysis")

text = st.text_area("Enter Tweet")

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter text")

    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]

        if pred == 0:
            st.error("Negative Sentiment 😠")
        else:
            st.success("Positive Sentiment 😊")
