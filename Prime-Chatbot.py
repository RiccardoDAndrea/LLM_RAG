import streamlit as st
import openai
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from tempfile import NamedTemporaryFile
from streamlit_pdf_viewer import pdf_viewer

Open_api_token = st.sidebar.text_input("OpenAI API Token", "sk-", type="password")

example_pdfs = {
    "The economic potential of generative AI": "PDF_docs/the-economic-potential-of-generative-ai-the-next-productivity-frontier-vf.pdf",
    "Overcoming huge challenges in cancer": "PDF_docs/WIREs Mechanisms of Disease - 2013 - Roukos - Genome network medicine  innovation to overcome huge challenges in cancer.pdf",
}

# Streamlit Main
st.sidebar.title("Prime-Chatbot 🤖")

st.sidebar.markdown("### Example PDFs")
selected_example_pdfs = st.sidebar.selectbox('Choose your PDF example', options=["The economic potential of generative AI", "Overcoming huge challenges in cancer", "Upload your own data"])

st.title("Prime-Chatbot 🤖")
st.warning("To get started enter your API Token in the Sidebar.")
st.write("""
        This is a simple implementation of OpenAI's Retrieval Augmented Generation (RAG) model. 🧠 The model can generate long-form answers 
        to questions and is suitable for various tasks such as question answering, summarization, and translation. 📝🌐🔍

        > To get started, two different documents are provided, but you can also upload your own PDF documents. ⬆️

        1. The first document, "The Economic Potential of Generative AI," is an assessment by McKinsey & Company exploring the potential of 
        AI in the commercial sector. 📈🤖

        2. The second document, "Overcoming Huge Challenges in Cancer," discusses innovative approaches to combating cancer. 💊🧬""")


st.expander("About Prime-Chatbot").info("""
                                        The goal of the Prime-Chatbot 🤖 is to efficiently retrieve 
                                        information and make complex scientific work understandable 
                                        to everyone through interactive queries. Currently, the chatbot 
                                        exhibits inconsistencies in its responses and struggles to answer
                                        questions about documents accurately. 📄❓ However, when questions 
                                        are precisely formulated, the model provides solid answers. 💬👍 
                                        Nevertheless, there is still much fine-tuning needed to elevate its performance to a 'good' level. 🔧✨""")

st.divider()

# Function to load and split PDFs
def load_and_split_pdf(uploaded_file, selected_example):
    chunks = []
    # Load the uploaded file
    if uploaded_file:
        try:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
            loader = PyPDFLoader(temp_file.name)
            chunks += loader.load_and_split()
            os.unlink(temp_file.name)
        except Exception as e:
            st.error(f"Error loading and splitting the PDF: {e}")
    
    # Load the selected example PDF
    elif selected_example and example_pdfs[selected_example]:
        try:
            pdf_path = example_pdfs[selected_example]
            loader = PyPDFLoader(pdf_path)
            chunks += loader.load_and_split()
        except Exception as e:
            st.error(f"Error loading and splitting the example PDF {pdf_path}: {e}")

    return chunks

# Function to initialize embeddings
def initialize_embeddings():
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embedding_function
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        return None

# Function to initialize Chroma database
def initialize_chroma(chunks, embedding_function):
    try:
        db = Chroma.from_documents(chunks, embedding_function)
        return db
    except Exception as e:
        st.error(f"Error initializing Chroma database: {e}")
        return None

# Function to retrieve documents
def retrieve_documents(db, query):
    try:
        retriever = db.as_retriever(search_kwargs={"k": 2})
        retriever.invoke(query)
        return retriever
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return None

# Function to initialize LLM model
def initialize_llm_model():
    try:
        llm = ChatOpenAI(
            openai_api_key=Open_api_token,
            model_name="gpt-3.5-turbo",
            temperature=0.0,
            max_tokens=300
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing OpenAI model: {e}")
        return None

# Function to perform QA with sources
def qa_with_sources(llm_instance, chunks, embedding_instance, query):
    try:
        text_splitter_instance = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=10,
            length_function=len,
        )
        db = initialize_chroma(chunks, embedding_instance)
        if not db:
            return {"answer": "Error initializing Chroma database."}
        retriever_instance = retrieve_documents(db, query)
        if not retriever_instance:
            return {"answer": "Error retrieving documents."}
        qa_with_sources_result = RetrievalQAWithSourcesChain.from_chain_type(llm=llm_instance, chain_type="stuff", retriever=retriever_instance)
        return qa_with_sources_result.invoke(query)
    except Exception as e:
        st.error(f"Error processing QA: {e}")
        return {"answer": f"Error processing request: {e}"}

# Main execution

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Show file uploader only if "Upload your own data" is selected
uploaded_file = None
if selected_example_pdfs == "Upload your own data":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if "messages" not in st.session_state:
        st.session_state.messages = []
    with st.chat_message("assistant"):
        st.markdown("""
                    Welcome to Prime! 🤖 I am your personal document detective! Send me your PDFs and I will put them through their paces. From

                    - What is addressed in the uploaded document
                    - Can you summarize the document
                    - What are the key takeaways from the document

                    I'm your man! 🕵️‍♂️ Uh, your bot. Never mind, you know. Let us crack your PDFs! 💼 Don't forget to upload a PDF! 📎
                    """)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Ask a question about the uploaded or selected PDFs", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Thinking..."):  # Display spinner while processing
            llm_instance = initialize_llm_model()
            if not llm_instance:
                st.write("Error initializing the LLM model.")
            else:
                chunks = load_and_split_pdf(uploaded_file, selected_example_pdfs)
                embedding_instance = initialize_embeddings()
                if not embedding_instance:
                    st.write("Error initializing embeddings.")
                else:
                    answer = qa_with_sources(llm_instance, chunks, embedding_instance, prompt)
                    st.write(answer["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": answer["answer"]})
    
        
if selected_example_pdfs == "The economic potential of generative AI":
    st.sidebar.write("The document can be viewed under the following link")
    st.sidebar.info("Source documents: [McKinsey & Company](https://www.mckinsey.de/~/media/mckinsey/locations/europe%20and%20middle%20east/deutschland/news/presse/2023/2023-06-14%20mgi%20genai%20report%2023/the-economic-potential-of-generative-ai-the-next-productivity-frontier-vf.pdf)")
    with st.chat_message("assistant"):
        st.markdown("""
            Welcome to Prime! 🤖 I am your personal document detective! Send me your PDFs and I will put them through their paces.
            Let's dive into the study by McKinsey & Company on the economic potential of generative AI. 📈🤖
                    
            📚 I've already prepared a few questions to get us started:
            
            - Can you summarize the document comprehensively?
            - What are some specific use cases of generative AI in the retail and banking industries?
            - How can businesses and society prepare for the implications of integrating generative AI into their operations?
            What new quality checks must companies implement when transitioning from human labor to generative AI, particularly 
            regarding content generation like emails or assisting processes such as drug discovery? How can transparency and 
            traceability be ensured in generative AI systems to minimize risks and ensure quality?
            
            I'm your man! 🕵️‍♂️ Uh, your bot. Never mind, you know. Let us crack your PDFs! 💼
            Don't forget to upload a PDF or select an example PDF! 📎
            """)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Ask a question about the uploaded or selected PDFs", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Thinking..."):  # Display spinner while processing
            llm_instance = initialize_llm_model()
            if not llm_instance:
                st.write("Error initializing the LLM model.")
            else:
                chunks = load_and_split_pdf(uploaded_file, selected_example_pdfs)
                embedding_instance = initialize_embeddings()
                if not embedding_instance:
                    st.write("Error initializing embeddings.")
                else:
                    answer = qa_with_sources(llm_instance, chunks, embedding_instance, prompt)
                    with st.chat_message("assistant"):
                        st.write(answer["answer"])
                        st.session_state.messages.append({"role": "assistant", "content": answer["answer"]})
    

if selected_example_pdfs == "Overcoming huge challenges in cancer":
    st.sidebar.write("The document can be viewed under the following link")
    st.sidebar.info("Source documents: [United States National Library of Medicine](https://wires.onlinelibrary.wiley.com/doi/10.1002/wsbm.1254)")
    
    with st.chat_message("assistant"):
        st.markdown(""" 
        🌟 Welcome to Prime! 🤖 I'm your personal document detective! Feel free to ask me anything about the groundbreaking work of Dr. Roukos, who explored innovative approaches to curing cancer. 
        Together, we'll unravel the complexities and gain a deeper understanding of this fascinating subject.
        
        📚 I've already prepared a few questions to get us started:

        - How has the post-ENCODE era influenced biomedical research in understanding gene expression and cellular processes?
        
        - What challenges do traditional diagnostics and therapeutics face in treating advanced cancer, and how can network-based approaches offer new solutions?
        
        - How do next-generation sequencing technologies revolutionize personalized cancer treatment by mapping a patient's unique mutational landscape?

        🕵️‍♂️ I'm here to make complex scientific articles accessible and understandable for everyone! 💼 Don't forget to upload a PDF or select an example PDF to get started! 📎

        ---

        #### Reference:
        Roukos, D. H. (2014). Genome network medicine: innovation to overcome huge challenges in cancer therapy. *Wiley Interdisciplinary Reviews: Systems Biology and Medicine*, 6(2), 201-208.

        """)


    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input("Ask a question about the uploaded or selected PDFs", key="chat_input_1"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Thinking..."):  # Display spinner while processing
            llm_instance = initialize_llm_model()
            if not llm_instance:
                st.write("Error initializing the LLM model.")
            else:
                chunks = load_and_split_pdf(uploaded_file, selected_example_pdfs)
                embedding_instance = initialize_embeddings()
                if not embedding_instance:
                    st.write("Error initializing embeddings.")
                else:
                    with st.chat_message("assistant"):
                        answer = qa_with_sources(llm_instance, chunks, embedding_instance, prompt)
                        st.write(answer["answer"])
                        st.session_state.messages.append({"role": "assistant", "content": answer["answer"]})
                    
