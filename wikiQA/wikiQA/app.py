import streamlit as st
from src.preprocessing import load_data, clean_data
from src.model import train_model, predict
from src.bert_model import train_bert, predict_bert

@st.cache_resource
def load_and_prepare_data():
    """Load and clean data once"""
    df = load_data("data/train.tsv")
    df = clean_data(df)
    return df

@st.cache_resource
def load_tfidf_model():
    """Train TF-IDF model once"""
    df = load_and_prepare_data()
    model, vectorizer = train_model(df)
    return model, vectorizer

@st.cache_resource
def load_bert_model():
    """Train BERT model once"""
    df = load_and_prepare_data()
    bert_df = df.sample(300, random_state=42)
    bert_model, tokenizer = train_bert(bert_df, batch_size=2)
    return bert_model, tokenizer

# Load data and models
df = load_and_prepare_data()
tfidf_model, vectorizer = load_tfidf_model()
bert_model, tokenizer = load_bert_model()

# UI
st.title("WikiQA Question Answering System")

question = st.text_input("Enter your question:")

if st.button("Get Answers"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        subset = df[df['Question'].str.contains(question.lower(), na=False)]

        if subset.empty:
            st.warning("No matching question found in dataset")
        else:
            sentences = subset['Sentence'].tolist()

            # TF-IDF
            tfidf_results = predict(tfidf_model, vectorizer, question, sentences)

            st.subheader("Top Answers (TF-IDF)")
            for s, score in tfidf_results[:5]:
                st.write(f"{score:.3f} → {s}")

            # BERT
            bert_results = predict_bert(bert_model, tokenizer, question, sentences)

            st.subheader("Top Answers (BERT)")
            for s, score in bert_results[:5]:
                st.write(f"{score:.3f} → {s}")