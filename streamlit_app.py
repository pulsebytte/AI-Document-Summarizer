import openai
import streamlit as st
import time
import os
import tempfile

# Load OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found. Set OPENAI_API_KEY as an environment variable.")
    st.stop()

client = openai.OpenAI(api_key=api_key)

st.title("📚 AI Document Assistant 🤖")
st.write("Upload your documents, then chat with AI!")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "assistant_id" not in st.session_state:
    st.session_state.assistant_id = None

# File Upload UI
uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["txt", "pdf", "docx"])

def upload_file(uploaded_file):
    supported_extensions = [".txt", ".pdf", ".docx"]
    file_ext = os.path.splitext(uploaded_file.name)[1]
    
    if file_ext.lower() not in supported_extensions:
        st.error(f"❌ Unsupported file type: {file_ext}. Please upload .txt, .pdf, or .docx files.")
        return None
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    with open(temp_file_path, 'rb') as file:
        response = client.files.create(file=file, purpose='assistants')
    
    os.remove(temp_file_path)
    return response.id

def add_files_to_vector_store(vector_store_id, file_ids):
    client.beta.vector_stores.file_batches.create_and_poll(
        vector_store_id=vector_store_id,
        file_ids=file_ids
    )

if uploaded_files:
    st.success(f"{len(uploaded_files)} document(s) uploaded successfully! AI is processing...")
    
    vector_store = client.beta.vector_stores.create(name="User Uploaded Documents")
    file_ids = [upload_file(file) for file in uploaded_files if upload_file(file)]
    
    if file_ids:
        add_files_to_vector_store(vector_store.id, file_ids)
        assistant = client.beta.assistants.create(
            name="DocSum",
            instructions="You are an AI assistant that summarizes documents and answers questions based only on them.",
            model="gpt-4o-mini",
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
            temperature=0.7,
            top_p=0.7
        )
        st.session_state.assistant_id = assistant.id
        st.session_state.thread_id = client.beta.threads.create().id
        st.success("✅ AI is ready! Start chatting below.")

st.subheader("💬 Chat with AI")
user_query = st.chat_input("Ask anything about the documents:")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    if "summarize" in user_query.lower() or "summary" in user_query.lower() or "summarise" in user_query.lower():
        instructions = "Provide a concise summary of all uploaded documents, covering key points and main ideas."
    else:
        instructions = user_query
    
    response = client.beta.threads.runs.create(
        thread_id=st.session_state.thread_id,
        assistant_id=st.session_state.assistant_id,
        instructions=instructions
    )
    
    def wait_for_run_completion(run_id, thread_id):
        while True:
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            if run.status == 'completed':
                return run
            elif run.status == 'failed':
                st.error("⚠️ AI failed to generate a response.")
                return None
            time.sleep(2)

    completed_run = wait_for_run_completion(response.id, st.session_state.thread_id)
    
    if completed_run:
        messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
        output = messages.data[0].content[0].text.value if messages.data else "No response generated."
        st.session_state.chat_history.append({"role": "assistant", "content": output})

for msg in st.session_state.chat_history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.write(msg["content"])