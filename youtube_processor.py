import re
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings



load_dotenv()

GROQ_API_KEY = os.getenv("groq_key") 
MODEL_NAME = "llama-3.3-70b-versatile"  

def extract_video_id(url):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'embed\/([0-9A-Za-z_-]{11})',
        r'shorts\/([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join([t["text"] for t in transcript_list])
    except TranscriptsDisabled:
        raise Exception("No English captions available for this video.")
    except Exception as e:
        raise Exception(f"Error getting transcript: {str(e)}")

def setup_qa_chain(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
 
    vector_store = FAISS.from_documents(chunks, embeddings)
    

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    
    llm = ChatGroq(
        temperature=0.2,
        groq_api_key=GROQ_API_KEY, 
        model_name=MODEL_NAME      
    )
    
  
    prompt = PromptTemplate(
        template="""You are an expert at answering questions about YouTube videos.
        
        Video Transcript Context:
        {context}
        
        Question: {question}
        
        Provide a detailed answer using only the transcript. 
        If the answer isn't in the transcript, say "I couldn't find this information in the video".
        Include relevant timestamps when possible in [HH:MM:SS] format.
        Answer:""",
        input_variables=["context", "question"]
    )
    

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    
    return (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )