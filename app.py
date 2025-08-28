import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain 

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chatHistory" not in st.session_state:
    st.session_state.chatHistory = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

def user_input(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question':user_question})
        st.session_state.chatHistory = response['chat_history']
        for i,message in enumerate(st.session_state.chatHistory):
            if i%2 ==0:
                st.write("User: ",message.content)
            else:
                st.write("Reply:", message.content)
    else:
        st.warning("Please upload PDFs and click 'Submit & Process' first.")

def main():
    st.set_page_config("Information Retrieval")
    st.header("Information-Retrieval-System 💻")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit and Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text =  get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)

                # Initialize vector_store and conversation ONLY ONCE
                # Check if vector_store is already in session state
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = get_vector_store(text_chunks)
                
                # Check if conversation is already in session state
                if st.session_state.conversation is None:
                    # Use the vector_store from session state
                    st.session_state.conversation = get_conversational_chain(st.session_state.vector_store)
                
                st.success("Done")

if __name__ == "__main__":
    main()