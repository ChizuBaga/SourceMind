import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM

import PyPDF2
import argparse
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

import requests
from fpdf import FPDF
from bs4 import BeautifulSoup

#to compatible with the streamlit version
import sqlite3
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

CHROMA_PATH = "chroma"
DATA_PATH = "data"

# List of websites to crawl
urls = []

@st.cache_data
def crawl_webpage(url):
    response = requests.get(url)
    print(f"Response Status Code: {response.status_code} for {url}")
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = ' '.join([para.get_text() for para in paragraphs])
        return text_content
    else:
        return None


if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


def save_to_pdf(text, filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    encoded_text = text.encode('utf-8', 'replace').decode('latin-1')
    pdf.add_page()
    pdf.multi_cell(0, 10, encoded_text)
    pdf.output(filename)


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks


def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3", show_progress=True, model_kwargs={"device":"cpu"},)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function())
    
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


db = Chroma(persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function(), collection_metadata={"hnsw:space": "cosine"})


def add_url_and_pdf_input():
    st.subheader("Add URLs and PDF Files")

    # Limit to 2 URLs
    url1 = st.text_input("Enter URL 1:")
    url2 = st.text_input("Enter URL 2:")

    # Limit to 2 PDF uploads
    pdf1 = st.file_uploader("Upload PDF 1", type="pdf", label_visibility='collapsed')
    pdf2 = st.file_uploader("Upload PDF 2", type="pdf", label_visibility='collapsed')

    if st.button("Submit"):
        urls = [u for u in [url1, url2] if u]
        pdf_files = [p for p in [pdf1, pdf2] if p]

        # Process URLs
        for i, url in enumerate(urls):
            website_text = crawl_webpage(url)
            if website_text:
                # Save a local PDF copy (optional)
                base_filename = os.path.join(DATA_PATH, f'url_content_{i+1}.pdf')
                pdf_path = base_filename
                count = 1
                while os.path.exists(pdf_path):
                    pdf_path = os.path.join(DATA_PATH, f'url_content_{i+1}_{count}.pdf')
                    count += 1
                save_to_pdf(website_text, pdf_path)

                # Wrap in Document, split, and embed
                doc = Document(page_content=website_text, metadata={"source": url, "page": 1})
                chunks = split_documents([doc])
                add_to_chroma(chunks)

                st.success(f"{url} content processed and added ✅")
            else:
                st.error(f"Could not access {url}. Please check the link.")

        # Process PDFs
        for i, pdf in enumerate(pdf_files):
            pdf_path = os.path.join(DATA_PATH, f'uploaded_pdf_{i+1}.pdf')
            count = 1
            while os.path.exists(pdf_path):
                pdf_path = os.path.join(DATA_PATH, f'uploaded_pdf_{i+1}_{count}.pdf')
                count += 1

            with open(pdf_path, "wb") as f:
                f.write(pdf.getbuffer())
            st.success(f"Uploaded PDF: {pdf_path}")

            # Extract text, split, and embed
            pdf_text = extract_text_from_pdf(pdf_path)
            doc = Document(page_content=pdf_text, metadata={"source": pdf_path, "page": 1})
            chunks = split_documents([doc])
            add_to_chroma(chunks)

            st.success(f"PDF content from {pdf_path} processed and added ✅")

@st.cache_resource
def load_llm():
    model_id = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,  # lower for speed
        temperature=0.7
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"  # Extract text from each page
    return text


#set the app title
st.title('Welcome to SourceMind')

# Initial text
st.subheader('Input your PDF files or URLS and ask away! 🤖')  

#URL and PDF input
add_url_and_pdf_input()

# User inputs the question
question = st.text_input("Enter your question:")


if st.button("Enter"):
    CHROMA_PATH = "chroma"
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    def main(query_text=question):
        # Load the Chroma DB
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function(),
            collection_metadata={"hnsw:space": "cosine"}
        )

    # Run similarity search
        results = db.similarity_search_with_relevance_scores(query_text, k=2)
        if not results or results[0][1] < 0.7:
            st.warning("It looks like your question might not be related to this content.")
            return

    # Create context from retrieved docs
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Format prompt
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context_text,
            question=query_text
        )

    # Generate answer
        response_text = llm.invoke(prompt)

    # Display
        sources = [doc.metadata.get("source", None) for doc, _ in results]
        st.markdown(f"**Answer:** {response_text}")
        st.markdown(f"**Sources:** {sources}")

    # Run the main function

main()


def main2():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",
                        action="store_true",
                        help="Reset the database.")
    args = parser.parse_args(args=[])
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main2()
