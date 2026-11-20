import streamlit as st
from similarity import compare_questions
from dataset import load_dataset

st.set_page_config(page_title="Text Similarity & Duplicate Detection", page_icon="🔍", layout="centered")
st.title("🔍 Text Similarity and Duplicate Detection")
st.write("Enter two questions to estimate whether they have the same meaning. The classifier is trained using labelled question pairs.")

st.subheader("Compare two questions")
question1 = st.text_area("Question 1", placeholder="Example: How can I learn Python?", height=110)
question2 = st.text_area("Question 2", placeholder="Example: What is the best way to learn Python?", height=110)

if st.button("Compare Questions", type="primary", use_container_width=True):
    if not question1.strip() or not question2.strip():
        st.error("Please enter both questions.")
    else:
        try:
            with st.spinner("Checking similarity..."):
                probability, cosine_score, label = compare_questions(question1, question2)
            st.divider()
            st.subheader("Result")
            c1, c2 = st.columns(2)
            c1.metric("Model duplicate probability", f"{probability}%")
            c2.metric("TF-IDF cosine similarity", f"{cosine_score}%")
            if probability >= 75:
                st.success(label)
            elif probability >= 45:
                st.warning(label)
            else:
                st.info(label)
            st.caption("These are model estimates, not proof of plagiarism or guaranteed semantic equivalence.")
        except Exception as error:
            st.error(str(error))

st.divider()
st.subheader("Dataset preview")
if st.button("Load dataset preview"):
    try:
        df = load_dataset()
        st.write(f"Usable question pairs: {len(df):,}")
        st.dataframe(df.head(100), use_container_width=True)
        st.bar_chart(df["is_duplicate"].value_counts().sort_index())
    except Exception as error:
        st.error(str(error))

st.caption("If this is your first run, train the model first using python model_training.py in the project terminal.")
