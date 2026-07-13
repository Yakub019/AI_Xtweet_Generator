from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(
 repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text generation"
    ,temperature=0.7)
model = ChatHuggingFace(llm=llm)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# YT

video_id = "JMdpszLl2FI"
api = YouTubeTranscriptApi()
try:
 transcript_list = api.fetch(video_id,languages=["en"])
 transcript = " ".join(chunk.text for chunk in transcript_list)
 
except TranscriptsDisabled:
 print("No captions available for this video")

 # Text splitting

splitter = RecursiveCharacterTextSplitter(
 chunk_size =700,
 chunk_overlap=200
)
chunks = splitter.create_documents([transcript])


# chunks to embeddings
vector_store = FAISS.from_documents(chunks,embeddings)


# Retriver

retriver = vector_store.as_retriever(search_type = "similarity",search_kwargs={"k":4})
result = retriver.invoke("what is agentic ai")
print(result[0])