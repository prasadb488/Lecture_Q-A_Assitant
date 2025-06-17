
import os
import subprocess
import whisper
from pydub import AudioSegment
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import google.generativeai as genai

def split_video_ffmpeg(video_path, output_folder="chunks", chunk_duration=30):
    os.makedirs(output_folder, exist_ok=True)
    output_template = os.path.join(output_folder, "chunk_%03d.mp4")
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-c", "copy",
        "-map", "0",
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-reset_timestamps", "1",
        output_template
    ]
    subprocess.run(cmd, check=True)
    return sorted(os.listdir(output_folder))
split_video_ffmpeg("/Users/saiprasadbanda/Desktop/untitled folder/14 DevTools Tricks Thatll Make You a Better Developer - camelCase (1080p, h264).mp4", chunk_duration=30)


model = whisper.load_model("base")

def convert_to_wav(mp4_path):
    audio = AudioSegment.from_file(mp4_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    wav_path = mp4_path.replace(".mp4", ".wav")
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_chunks(chunk_folder="chunks"):
    transcripts = []
    for file in sorted(os.listdir(chunk_folder)):
        if file.endswith(".mp4"):
            mp4_path = os.path.join(chunk_folder, file)
            wav_path = convert_to_wav(mp4_path)
            result = model.transcribe(wav_path)
            transcripts.append({"chunk": file, "text": result["text"]})
    return transcripts
transcripts = transcribe_chunks()

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)


index_name = "lecture-qa"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled"
    )

index = pc.Index(index_name)


def index_transcripts(transcripts):
    vectors = []
    for i, item in enumerate(transcripts):
        embedding = genai.embed_content(
            model="models/embedding-001",
            content=item["text"],
        )["embedding"]
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {
                "text": item["text"],
                "chunk": item["chunk"]
            }
        })
    index.upsert(vectors)
index_transcripts(transcripts)

def ask_question(question):
    query_embed = genai.embed_content(
        model="models/embedding-001",
        content=question,
        task_type="query"
    )["embedding"]
    result = index.query(
        vector=query_embed,
        top_k=3,
        include_metadata=True
    )
    context = "\n\n".join([match["metadata"]["text"] for match in result["matches"]])
    chat_model = genai.GenerativeModel("gemini-2.0-flash")
    response = chat_model.generate_content(
        f"Based on the following lecture excerpts, answer the question:\n\n{question}\n\nLecture Context:\n{context}"
    )
    return response.text
answer = ask_question("What are the most commonly used DevTools features mentioned in the video")
print(answer)