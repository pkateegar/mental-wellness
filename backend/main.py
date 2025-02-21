from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import datetime
import json
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load BERT-based GoEmotions Model from Hugging Face
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", top_k=3)

# Load Llama 2 model from Ollama
llm = OllamaLLM(model="llama2")

# Initialize FAISS vector store
vector_store = None

# ===================== KNOWLEDGE BASE (Psychologist-Style) ===================== #
def create_knowledge_base():
    """Creates a FAISS vector store with predefined psychology-based responses"""
    global vector_store
    knowledge_text = """
    - Feeling sad is completely normal. It's okay to take time for yourself.
    - If you're overwhelmed, take a deep breath and allow yourself to relax.
    - You're stronger than you think, and your feelings are valid.
    - Sometimes, expressing emotions can be the best way to process them.
    - If you're feeling anxious, try grounding techniques such as focusing on your breath or listing things you see.
    - You're not alone. Even when things feel dark, there is light ahead.
    - Self-care is important. It's okay to rest when you need to.
    - If you're happy, embrace the moment and share your joy with others.
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_text(knowledge_text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

# Create knowledge base on startup
create_knowledge_base()

# ===================== EMOTION & DEPRESSION DETECTION ===================== #
def analyze_emotion(text):
    """Detects emotions using a fine-tuned BERT model on GoEmotions"""
    results = emotion_pipeline(text)
    detected_emotions = [result["label"] for result in results[0]]

    # Depression-related emotions
    depression_indicators = {"sadness", "grief", "loneliness", "despair", "hopelessness"}

    is_depressed = any(emotion in depression_indicators for emotion in detected_emotions)
    return detected_emotions, is_depressed
class UserMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat(data: UserMessage):
    """Processes user input, detects emotions, identifies depression, and provides responses"""
    global vector_store

    user_message = data.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Perform emotion and depression analysis
    detected_emotions, is_depressed = analyze_emotion(user_message)

    # Retrieve relevant psychology-based context using FAISS
    retrieved_docs = vector_store.similarity_search(user_message, k=1)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Adjust response tone based on detected emotions
    if "joy" in detected_emotions or "excitement" in detected_emotions:
        response_tone = "I'm so happy to hear that! Here's something to celebrate your joy:"
    elif "sadness" in detected_emotions or "grief" in detected_emotions:
        response_tone = "I'm really sorry you're feeling this way. You're not alone, and I'm here to listen:"
    elif "fear" in detected_emotions or "nervousness" in detected_emotions:
        response_tone = "It's okay to feel this way. Take a deep breath, and know that you are strong."
    elif "anger" in detected_emotions or "annoyance" in detected_emotions:
        response_tone = "I hear your frustration. Expressing emotions is healthy, and I'm here to listen."
    else:
        response_tone = "I hear you. Let's explore this together with an open heart:"

    # If depression is detected, modify response to provide support
    if is_depressed:
        response_tone += "\n\n💙 I sense that you might be feeling really low. Remember, you are not alone. It's always okay to seek support from close ones or mental health professionals. Would you like some self-care suggestions or resources to help? 💙"

    # Create prompt for AI (Psychologist-style chat)
    prompt_text = f"""
    {response_tone}

    Empathetic Advice:
    {context}

    User: {user_message}
    AI:
    """

    try:
        # Generate response using Llama 2 (via Ollama)
        answer = llm.invoke(prompt_text)
        return JSONResponse(content={"emotions": detected_emotions, "is_depressed": is_depressed, "response": answer + '<br/><br/> <p style="color:red">Your emotions are : '+', '.join(detected_emotions)+'</p>', "response_tone":response_tone})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
