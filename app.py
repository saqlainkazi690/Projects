import streamlit as st
import tempfile
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def process_pdf(self, pdf_file):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load and process the PDF
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            
            # Split into chunks
            docs = self.text_splitter.split_documents(pages)
            
            # Create vector store
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            
            # Cleanup
            os.unlink(tmp_path)
            
            return vectorstore
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None

class ChatBot:
    def __init__(self):
        # Initialize Llama 2 with streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = Ollama(
            model="llama2",
            callback_manager=callback_manager,
            temperature=0.7
        )
        
    def setup_chain(self, vectorstore):
        # Setup memory with output key specified
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'  # Specify the output key
        )
        
        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True,
            verbose=True  # Add verbose mode for debugging
        )
        
        return chain

class StreamlitApp:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.chatbot = ChatBot()
        
    def setup_session_state(self):
        if 'chain' not in st.session_state:
            st.session_state.chain = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'processed_file' not in st.session_state:
            st.session_state.processed_file = None
    
    def run(self):
        st.title("PDF Chat with Llama 2")
        self.setup_session_state()
        
        # File upload
        uploaded_file = st.file_uploader("Upload a PDF", type=['pdf'])
        
        if uploaded_file and (st.session_state.processed_file != uploaded_file.name):
            with st.spinner("Processing PDF..."):
                vectorstore = self.processor.process_pdf(uploaded_file)
                if vectorstore is not None:
                    st.session_state.chain = self.chatbot.setup_chain(vectorstore)
                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.chat_history = []  # Reset chat history for new document
                    st.success("PDF processed successfully!")
                else:
                    st.error("Failed to process PDF. Please try again.")
        
        if st.session_state.chain:
            # Chat interface
            user_question = st.text_input("Ask a question about your PDF:")
            
            if user_question:
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.chain({"question": user_question})
                        st.session_state.chat_history.append(("You", user_question))
                        st.session_state.chat_history.append(("Bot", response['answer']))
                        
                        # Display source documents if needed
                        if st.checkbox("Show source documents"):
                            st.write("Sources:")
                            for doc in response['source_documents']:
                                st.write(doc.page_content)
                                
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
            
            # Display chat history
            st.subheader("Chat History")
            for role, message in st.session_state.chat_history:
                if role == "You":
                    st.write(f"👤 **You:** {message}")
                else:
                    st.write(f"🤖 **Assistant:** {message}")

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()