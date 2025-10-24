# Format Python code here
import io

import pandas as pd
import PyPDF2
import pytesseract
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from pdf2image import convert_from_bytes

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", google_api_key=st.secrets["GEMINI_API_KEY"]
)


# Ekstraksi teks PDF


def text_extraction_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    if not text.strip():
        images = convert_from_bytes(file.read())
        for image in images:
            text += pytesseract.image_to_string(image)
    return text


# Membuat ringkasan dari dokumen PDF
def summary(text):
    prompt = (
        f"Buat ringkasan singkat dari teks berikut:\n\n{text}\n\nRingkasan singkat:"
    )
    response = llm.predict(prompt)
    return response


# Mendapatkan kata-kata kunci dari dokumen PDF
def most_frequent_word_list(text):
    prompt = f"Dari teks berikut, berikan daftar kata-kata kunci yang paling sering muncul:\n\n{text}\n\nKata-kata kunci:"
    response = llm.predict(prompt)
    return response


# Streamlit UI


st.set_page_config(
    page_title="Mesin Penjawab Query PDF",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Mesin Penjawab Query PDF")

with st.sidebar:

    st.header("Hapus Obrolan")
    if st.button("Hapus Obrolan"):
        st.session_state["messages"] = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# File Upload:
upload_file = st.file_uploader("Unggah File PDF", type=["pdf"])

if upload_file:
    file_id = f"{upload_file.name}_{upload_file.size}"

    if (
        "last_uploaded_file" not in st.session_state
        or st.session_state["last_uploaded_file"] != file_id
    ):
        with st.spinner("Melakukan ekstraksi dan indeksasi PDF..."):
            pdf_text = text_extraction_pdf(upload_file)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = text_splitter.split_text(pdf_text)

            if not chunks:
                st.error(
                    "Tidak ada teks yang berhasil diekstrak dari PDF. Pastikan file tidak berupa hasil scan gambar."
                )
                st.stop()

            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            vectordb = FAISS.from_texts(chunks, embeddings)

            retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True,
            )

            # Creating session state variables to store vectordb, qa_chain, pdf_text, last_uploaded_file, doc_summary, keywords to avoid reprocessing

            st.session_state["vectordb"] = vectordb
            st.session_state["qa_chain"] = qa_chain
            st.session_state["pdf_text"] = pdf_text
            st.session_state["last_uploaded_file"] = file_id

            st.session_state["doc_summary"] = summary(pdf_text)
            st.session_state["keywords"] = most_frequent_word_list(pdf_text)

        st.success("PDF telah diproses")

    st.markdown("## Ringkasan Dokumen:")
    st.write(st.session_state["doc_summary"])

    st.markdown("---")

    st.markdown("## Kata-kata Kunci:")
    st.write(st.session_state["keywords"])

    query = st.text_input("Masukkan pertanyaan Anda:")
    if query:
        with st.spinner("Berpikir..."):
            qa_chain = st.session_state["qa_chain"]
            result = qa_chain({"query": query})
            st.subheader("Jawaban:")
            st.write(result["result"])

            with st.expander("Sumber Chunks"):
                for doc in result["source_documents"]:
                    st.markdown(doc.page_content)

    # st.success("PDF telah diproses")
    # st.markdown("Ringkasan Dokumen:")
    # doc_summary = summary(pdf_text)
    # st.write(doc_summary)
    # st.markdown("Kata-kata Kunci:")
    # keywords = most_frequent_word_list(pdf_text)
    # st.write(keywords)
    # query = st.text_input("Apa yang ingin ditanyakan?")
    # if query:
    #   with st.spinner("Berpikir..."):
    #      result = qa_chain({"query": query})
    #      st.subheader("Answer:")
    #      st.write(result["result"])

    #       with st.expander("Sumber Chunks"):
    #           for doc in result["source_documents"]:
    #               st.markdown(doc.page_content)


# Memperbaiki kode agar
