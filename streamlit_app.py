import os
import time
import tempfile
import streamlit as st
from openai import AzureOpenAI
from typing import List, Dict, Optional, Union

class AzureDocumentAssistant:
    def __init__(self):
        # Get Azure OpenAI configuration from environment variables
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.assistant_id = os.getenv("AZURE_ASSISTANT_ID")
        self.vector_store_id = os.getenv("AZURE_VECTOR_STORE_ID")

        # Initialize Azure OpenAI client
        self._initialize_client()

        # Initialize session state
        self._initialize_session_state()

    def _initialize_client(self):
        """Initialize and test Azure OpenAI client connection"""
        try:
            self.client = AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version="2025-01-01-preview"
            )
            self.client.models.list()
        except Exception as e:
            st.error(f"❌ Failed to connect to Azure OpenAI: {str(e)}")
            st.info("Please check if your API key and endpoint are correct.")
            st.stop()

    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        state_vars = {
            "chat_history": [],
            "thread_id": None,
            "uploaded_documents": [],
            "vector_store_documents": [],
            "temperature": 0.7,
            "top_p": 0.9,
            "last_upload_time": None
        }
        
        for var, default in state_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default

        # Initialize vector store documents
        self._fetch_vector_store_documents()

    def _fetch_vector_store_documents(self):
        """Fetch existing documents from vector store"""
        try:
            vector_files = self.client.beta.vector_stores.files.list(
                vector_store_id=self.vector_store_id
            )
            # Get file details for each file ID
            file_details = []
            for vector_file in vector_files.data:
                try:
                    file_info = self.client.files.retrieve(file_id=vector_file.id)
                    file_details.append({
                        "id": file_info.id,
                        "filename": file_info.filename,
                        "created_at": vector_file.created_at
                    })
                except Exception as e:
                    st.warning(f"Failed to fetch details for file {vector_file.id}: {str(e)}")
                    continue
                    
            st.session_state.vector_store_documents = file_details
        except Exception as e:
            st.warning(f"Failed to fetch vector store documents: {str(e)}")
            st.session_state.vector_store_documents = []

    def _is_file_in_vector_store(self, filename: str) -> bool:
        """Check if file already exists in vector store"""
        return any(
            doc["filename"].lower() == filename.lower()
            for doc in st.session_state.vector_store_documents
        )

    def _validate_file(self, uploaded_file) -> bool:
        """Validate file type and size"""
        supported_extensions = {'.txt', '.pdf', '.docx'}
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        max_size = 10 * 1024 * 1024  # 10MB in bytes
        
        if file_ext not in supported_extensions:
            st.error(f"❌ Unsupported file type: {uploaded_file.name}")
            return False
        
        if uploaded_file.size > max_size:
            st.error(f"❌ File too large: {uploaded_file.name}. Maximum size is 10MB.")
            return False
            
        return True

    def upload_file(self, uploaded_file) -> Optional[Dict]:
        """Upload file to Azure with the original filename."""
        if not self._validate_file(uploaded_file):
            return None

        # Check if file already exists in vector store
        if self._is_file_in_vector_store(uploaded_file.name):
            st.info(f"ℹ️ File {uploaded_file.name} already exists in vector store. Skipping upload.")
            return None

        save_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)

        try:
            # Write the uploaded file's content to the specified path
            with open(save_path, 'wb') as out_file:
                out_file.write(uploaded_file.getvalue())

            # Upload the file to Azure OpenAI
            with open(save_path, 'rb') as file:
                file_response = self.client.files.create(file=file, purpose='assistants')

            # Remove the temporary file
            os.remove(save_path)

            file_details = {
                "id": file_response.id,
                "filename": uploaded_file.name,
                "index": len(st.session_state.uploaded_documents),
                "upload_time": time.time()
            }

            # Only append to uploaded_documents if it's not already there
            if not any(doc['filename'] == uploaded_file.name for doc in st.session_state.uploaded_documents):
                st.session_state.uploaded_documents.append(file_details)
                st.session_state.last_upload_time = time.time()
            
            return file_details

        except Exception as e:
            st.error(f"File upload error: {str(e)}")
            if os.path.exists(save_path):
                os.remove(save_path)
            return None

    def list_uploaded_documents(self) -> List[Dict]:
        """Return list of currently uploaded documents"""
        return st.session_state.uploaded_documents

    def list_vector_store_documents(self) -> List[Dict]:
        """Return list of documents in vector store"""
        return st.session_state.vector_store_documents

    def create_thread(self) -> str:
        """Create new conversation thread"""
        thread = self.client.beta.threads.create()
        return thread.id

    def add_files_to_vector_store(self, file_ids: List[str]):
        """Add uploaded files to vector store"""
        try:
            # Filter out file IDs that are already in vector store
            existing_ids = {doc["id"] for doc in st.session_state.vector_store_documents}
            new_file_ids = [fid for fid in file_ids if fid not in existing_ids]
            
            if not new_file_ids:
                return None
                
            batch = self.client.beta.vector_stores.file_batches.create_and_poll(
                vector_store_id=self.vector_store_id,
                file_ids=new_file_ids
            )
            # Refresh vector store documents after successful upload
            self._fetch_vector_store_documents()
            return batch
        except Exception as e:
            st.error(f"Vector store update error: {e}")
            return None

    def generate_response(self, query: str) -> Optional[str]:
        """Generate AI response with improved error handling and retry logic"""
        if not st.session_state.thread_id:
            st.session_state.thread_id = self.create_thread()

        try:
            documents = st.session_state.uploaded_documents
            instructions = self._process_query(query, documents)

            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    run = self.client.beta.threads.runs.create(
                        thread_id=st.session_state.thread_id,
                        assistant_id=self.assistant_id,
                        instructions=instructions,
                        temperature=st.session_state.temperature,
                        top_p=st.session_state.top_p
                    )

                    timeout = time.time() + 60  # 60 second timeout
                    while run.status not in ['completed', 'failed']:
                        if time.time() > timeout:
                            raise TimeoutError("Response generation timed out")
                        time.sleep(1)
                        run = self.client.beta.threads.runs.retrieve(
                            thread_id=st.session_state.thread_id, 
                            run_id=run.id
                        )

                    if run.status == 'completed':
                        messages = self.client.beta.threads.messages.list(
                            thread_id=st.session_state.thread_id
                        )
                        return messages.data[0].content[0].text.value
                    break

                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        raise e
                    time.sleep(2 ** retry_count)  # Exponential backoff

            st.error("AI response generation failed after multiple attempts.")
            return None

        except Exception as e:
            st.error(f"Response generation error: {str(e)}")
            return None

    def _process_query(self, query: str, documents: List[Dict]) -> str:
        """Process and enhance the user query with improved document reference handling"""
        query_lower = query.lower()
        summary_keywords = {"summarize", "summary", "summarise"}
        
        if any(keyword in query_lower for keyword in summary_keywords):
            # Check for exact document references first
            doc_reference = query_lower.replace("summarize", "").replace("summary", "").replace("summarise", "").strip()
            
            # Look for the document by filename or reference number
            for doc in documents:
                # Check if the query matches the filename (without extension)
                filename_without_ext = os.path.splitext(doc['filename'])[0].lower()
                if (doc_reference == filename_without_ext or 
                    doc_reference == doc['filename'].lower() or
                    doc_reference == f"document {doc['index'] + 1}" or 
                    doc_reference == str(doc['index'] + 1)):
                    return f"Provide a detailed summary of the document named '{doc['filename']}'. Focus on its key points and main ideas."
            
            # If no specific document is found, provide feedback
            if doc_reference:
                return f"I notice you want to summarize '{doc_reference}', but I couldn't find that specific document. Please make sure the document name is correct or specify the document number. Available documents: " + ", ".join(f"Document {d['index'] + 1}: {d['filename']}" for d in documents)
            
            # If no specific document is mentioned, summarize all
            return "Provide a comprehensive summary of all uploaded documents, covering key points from each document."
        
        return query

def main():
    """Main application function with improved error handling and UI"""
    st.set_page_config(
        page_title="📚 Advanced Document AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    try:
        # Check if required environment variables are set
        required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_ASSISTANT_ID", "AZURE_VECTOR_STORE_ID"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            st.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
            st.info("""
            Please set the following environment variables:
            - AZURE_OPENAI_API_KEY
            - AZURE_OPENAI_ENDPOINT
            - AZURE_ASSISTANT_ID
            - AZURE_VECTOR_STORE_ID
            
            You can set these in your terminal before running the script:
            ```
            export AZURE_OPENAI_API_KEY=your-api-key
            export AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
            export AZURE_ASSISTANT_ID=your-assistant-id
            export AZURE_VECTOR_STORE_ID=your-vector-store-id
            ```
            """)
            st.stop()

        assistant = AzureDocumentAssistant()
        
        with st.sidebar:
            st.header("🔧 Settings")
            
            st.session_state.temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.temperature, 
                step=0.1,
                help="Controls response creativity. Lower values make output more deterministic."
            )
            
            st.session_state.top_p = st.slider(
                "Top-p (Nucleus Sampling)", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.top_p, 
                step=0.1,
                help="Controls response diversity. Lower values make output more focused."
            )

            st.header("📄 Documents")
            
            # Display vector store documents
            st.subheader("Vector Store Documents")
            vector_docs = assistant.list_vector_store_documents()
            if vector_docs:
                for i, doc in enumerate(vector_docs, 1):
                    st.markdown(f"📚 {i}. {doc['filename']}")
            else:
                st.info("No documents in vector store.")
            
            # Display currently uploaded documents
            st.subheader("Recently Uploaded")
            documents = assistant.list_uploaded_documents()
            if documents:
                for doc in documents:
                    st.markdown(f"📝 Document {doc['index'] + 1}: {doc['filename']}")
            else:
                st.info("No documents uploaded in this session.")

        st.title("📚 Advanced Document AI Assistant")

        uploaded_files = st.file_uploader(
            "Upload Documents", 
            accept_multiple_files=True, 
            type=["txt", "pdf", "docx"],
            help="Supported formats: TXT, PDF, DOCX (Max size: 10MB per file)"
        )

        if uploaded_files:
            with st.spinner("Processing documents..."):
                file_details = [fid for fid in map(assistant.upload_file, uploaded_files) if fid]
                
                if file_details:
                    file_ids = [detail['id'] for detail in file_details]
                    if file_ids:  # Only try to add if there are new files
                        assistant.add_files_to_vector_store(file_ids)
                        st.success(f"✅ {len(file_ids)} new document(s) added to vector store!")
                else:
                    st.info("No new documents to add to vector store.")

        st.subheader("💬 Chat with AI")
        
        # Add system status indicator
        if st.session_state.thread_id:
            st.success("🟢 System ready")
        else:
            st.warning("🟡 Waiting for first message")

        user_query = st.chat_input(
            "Ask about your documents...",
            disabled=not (st.session_state.uploaded_documents or st.session_state.vector_store_documents)
        )

        if user_query:
            with st.spinner("Generating response..."):
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                ai_response = assistant.generate_response(user_query)
                
                if ai_response:
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                    st.markdown(msg["content"])

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again. If the error persists, check your configuration.")

if __name__ == "__main__":
    main()