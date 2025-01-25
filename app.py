import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import dateparser
from datetime import datetime

import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import re

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf = PdfReader(pdf)
        for page in pdf.pages:
            text += page.extract_text()
    return text



def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks



def get_vector_stores(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss")



def get_conversational_chain():
    template = """
    Answer the question in as much detail as possible, from the provided context. If the answer is not available in the provided context, just say, "sorry, the answer is not available in the provided context.", don't provide the incorrect answer.\n\n
    Context: \n{context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def validate_email(email):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email)


def validate_phone(phone):
    pattern = r"^\d{10}$"
    return re.match(pattern, phone)


def extract_date(query):
    date = dateparser.parse(query)
    if date:
        return date.strftime("%Y-%m-%d")
    return None


def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True,
    )
    st.write("Response: ", response["output_text"])


def collect_user_info():
    st.subheader("Provide Your Contact Details")
    name = st.text_input("Name")
    phone = st.text_input("Phone Number (10 digits)")
    email = st.text_input("Email Address")
    date_query = st.text_input("Appointment Date (e.g., 'Next Monday')")

    if st.button("Submit Details"):
        errors = []
        appointment_date = extract_date(date_query)
        if not name:
            errors.append("Name is required.")
        if not validate_phone(phone):
            errors.append("Invalid phone number. Must be 10 digits.")
        if not validate_email(email):
            errors.append("Invalid email address format.")
        if not appointment_date:
            errors.append("Please provide a valid date.")

        if errors:
            st.error("\n".join(errors))
        else:
            st.success(
                f"Thank you, {name}! We will contact you at {phone} or {email} to confirm your appointment on {appointment_date}."
            )


def main():
    st.set_page_config("Question Answering Chatbot")
    st.title("AI Chatbot with PDF Query and Appointment Booking")


    with st.sidebar:
        st.header("Upload PDFs")
        pdfs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

        if st.button("Process PDFs"):
            if pdfs:
                with st.spinner("Processing..."):
                    text = get_text(pdfs)
                    chunks = get_chunks(text)
                    get_vector_stores(chunks)
                    st.success("PDFs processed successfully!")
            else:
                st.error("Please upload at least one PDF.")


    st.header("Ask Questions from Uploaded PDFs")
    question = st.text_input("Enter your question:")
    if st.button("Submit Question"):
        if question:
            user_input(question)
        else:
            st.error("Please enter a question.")

    st.header("Request a Call or Book an Appointment")
    collect_user_info()


if __name__ == "__main__":
    main()


