import streamlit as st
import os
import shutil
from processor import split_video_ffmpeg, transcribe_chunks, index_transcripts, ask_question
from dotenv import load_dotenv

load_dotenv()

st.title("🎓 Lecture Video Q&A Assistant")


def delete_old_chunks(folder="chunks"):
    if os.path.exists(folder):
        shutil.rmtree(folder)


video_file = st.file_uploader("Upload a lecture video", type=["mp4"])

if video_file:
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.read())
    delete_old_chunks("chunks")

    with st.spinner("Splitting video into chunks..."):
        split_video_ffmpeg("uploaded_video.mp4", output_folder="chunks")

    with st.spinner("Transcribing audio from chunks..."):
        transcripts = transcribe_chunks("chunks")

    with st.spinner("Indexing transcript chunks into Pinecone..."):
        index_transcripts(transcripts)

    st.success("✅ Video processed and indexed!")

    st.markdown("### 💬 Ask a question about the lecture:")
    question = st.text_input("Enter your question here")

    if question:
        with st.spinner("Thinking..."):
            answer = ask_question(question)
        st.subheader("📌 Answer:")
        st.write(answer)
