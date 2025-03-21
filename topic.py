import streamlit as st
import fitz  # PyMuPDF for PDFs
import docx
import openai
import faiss
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=openai_api_key)
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(file):

    text = ""
    if file.type == "application/pdf":
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.type == "text/plain":
        text = file.getvalue().decode("utf-8")
    return text


def get_hierarchical_topics(text, level=2):

    prompt = f"""Analyze the following text and generate hierarchical topics up to level {level}:
    {text[:4000]}"""
    response = llm(
        [SystemMessage(content="You are an AI that extracts multi-level topics."), HumanMessage(content=prompt)])
    return response.content.strip()


def format_topics_as_tree(topics):

    topic_lines = topics.split("\n")
    formatted_topics = ""
    for line in topic_lines:
        indent_level = line.count(" ") // 2  # Assume indentation by spaces
        formatted_topics += "&nbsp;&nbsp;&nbsp;&nbsp;" * indent_level + "🔹 " + line.strip() + "  \n"
    return formatted_topics


def query_topic(text, query):

    prompt = f"""
    Based on the document provided, answer the following query:
    Query: {query}
    Document: {text[:4000]}
    """
    response = llm(
        [SystemMessage(content="You are an AI answering document-based queries."), HumanMessage(content=prompt)])
    return response.content.strip()


def embed_and_store(text_chunks):

    embeddings = model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings


def search_similar_text(query, text_chunks, index, embeddings):

    query_embedding = model.encode([query])
    _, idxs = index.search(np.array(query_embedding), 1)
    return text_chunks[idxs[0][0]]


def generate_wordcloud(text):

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)


def perform_topic_clustering(text_chunks):

    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    text_matrix = vectorizer.fit_transform(text_chunks)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(text_matrix)
    topic_distributions = lda.transform(text_matrix)
    topic_labels = [f"Topic {i + 1}" for i in range(5)]

    # Extract top words per topic
    feature_names = vectorizer.get_feature_names_out()
    topic_words = {f"Topic {i + 1}": [feature_names[idx] for idx in lda.components_[i].argsort()[-10:]] for i in
                   range(5)}

    # Display topics and top words
    st.subheader("🔍 Topics and Top Words")
    for topic, words in topic_words.items():
        st.write(f"**{topic}:** {', '.join(words)}")

    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(topic_distributions[:10], cmap="Blues", annot=True, xticklabels=topic_labels, cbar=True)
    plt.xlabel("Topics")
    plt.ylabel("Documents")
    plt.title("Topic Clustering Heatmap")
    st.pyplot(plt)


# Streamlit UI
st.title("📄 Multi-Level Topic Modeling App")
uploaded_file = st.file_uploader("Upload a Document", type=["pdf", "docx", "txt"])

topic_depth = st.slider("Select Topic Depth Level", 1, 5, 2)

if uploaded_file:
    st.success("File uploaded successfully!")
    text = extract_text(uploaded_file)
    text_chunks = text.split("\n")
    index, embeddings = embed_and_store(text_chunks)
    topics = get_hierarchical_topics(text, level=topic_depth)
    formatted_topics = format_topics_as_tree(topics)

    st.subheader("📝 Extracted Multi-Level Topics")
    with st.expander("Click to expand topics"):
        st.markdown(formatted_topics, unsafe_allow_html=True)

    st.subheader("☁️ Word Cloud Visualization")
    generate_wordcloud(text)

    st.subheader("📊 Topic Clustering")
    perform_topic_clustering(text_chunks)

    user_query = st.text_input("Ask about the document:")
    if user_query:
        relevant_chunk = search_similar_text(user_query, text_chunks, index, embeddings)
        answer = query_topic(relevant_chunk, user_query)
        st.subheader("🤖 AI Response")
        st.write(answer)
