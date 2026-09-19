import os
import streamlit as st

from rag_utility import document_process_to_Chroma_db, question_answer


# Set the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))


st.title("🏆 PDF RAG Question Answering")


# File uploader widget
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"]
)


if uploaded_file is not None:

    # Define a save path
    save_path = os.path.join(
        working_dir,
        uploaded_file.name
    )

    # Save the uploaded file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the document
    process_document = document_process_to_Chroma_db(
        uploaded_file.name
    )

    st.success("Document processed successfully!")


    # Text widget to get user input
    user_question = st.text_area(
        "Ask your question about the document"
    )


    if st.button("Answer"):

        answer = question_answer(user_question)

        st.markdown("### Llama-3.3-70B Response")

        st.markdown(answer)
