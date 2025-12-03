# sarvam_pinecone_backend.py

import os
import asyncio
import base64
import logging
import random
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import bcrypt
import httpx
import jwt
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pinecone import Pinecone
from pydantic import BaseModel
from pymongo import ReturnDocument
from pymongo.errors import ConfigurationError

# --- Load environment variables ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not (PINECONE_API_KEY and PINECONE_INDEX_NAME):
    raise RuntimeError("Missing pinecone environment variables.")

numbers = [1, 2, 3]
selected = random.choice(numbers)
api_key = "GROQ_API_KEY_" + str(selected)
GROQ_API_KEY = os.getenv(api_key)

EMBEDDING_API_URL = "https://rahulbro123-embedding-model.hf.space/get_embeddings"

SARVAM_API_KEY = "sk_f8fjoda1_s83hQcvwfwwmPIwImLdTaReh"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

# --- Authentication & Persistence Configuration ---
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iomp_backend")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_USERS_COLLECTION = os.getenv("MONGO_USERS_COLLECTION", "users")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
try:
    JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "1440"))
except ValueError:
    JWT_EXPIRATION_MINUTES = 1440
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
BREVO_SENDER_EMAIL = "rahulvalavoju123@gmail.com"
BREVO_EMAIL_ENDPOINT = "https://api.brevo.com/v3/smtp/email"

mongo_client: Optional[AsyncIOMotorClient] = None
users_collection: Optional[AsyncIOMotorCollection] = None
database: Optional[Any] = None

if MONGO_URI:
    try:
        mongo_client = AsyncIOMotorClient(MONGO_URI)
        try:
            database = mongo_client.get_default_database()
        except ConfigurationError:
            if MONGO_DB_NAME:
                database = mongo_client[MONGO_DB_NAME]
            else:
                database = None
                logger.warning(
                    "Mongo URI does not include a database name and MONGO_DB_NAME is not set."
                )
        if database is not None:
            users_collection = database[MONGO_USERS_COLLECTION]
            logger.info(
                "MongoDB connected for auth features (collection: %s)",
                MONGO_USERS_COLLECTION,
            )
    except Exception as exc:
        logger.error("MongoDB connection error: %s", exc)
else:
    logger.warning("MONGO_URI not provided. Auth endpoints will be unavailable.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pinecone Client ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# --- Request/Response Models ---
class QueryRequest(BaseModel):
    question: str


class RegisterRequest(BaseModel):
    username: str
    email: str


class VerifyOtpRequest(BaseModel):
    email: str
    otp: str
    password: str


class LoginRequest(BaseModel):
    identifier: str
    password: str


def _require_user_collection() -> AsyncIOMotorCollection:
    if users_collection is None:
        logger.error("User store not configured; check MongoDB settings.")
        raise HTTPException(status_code=500, detail="User store not configured.")
    return users_collection


def _create_access_token(email: str) -> str:
    if not JWT_SECRET:
        logger.error("JWT secret missing; cannot issue tokens.")
        raise HTTPException(status_code=500, detail="Authentication service unavailable.")
    expiry = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    payload = {"email": email, "exp": expiry}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _generate_otp() -> int:
    return secrets.randbelow(900000) + 100000


def _normalize_email(email: str) -> str:
    return email.strip().lower()


async def _send_otp_email(recipient: str, otp: int) -> None:
    headers = {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    payload = {
        "sender": {"email": BREVO_SENDER_EMAIL},
        "to": [{"email": recipient}],
        "subject": "Your Verification OTP",
        "htmlContent": f"<p>Your OTP is <strong>{otp}</strong></p>",
        "textContent": f"Your OTP is {otp}",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            BREVO_EMAIL_ENDPOINT,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()


def _chunk_text_for_tts(text: str, max_chars: int = 480) -> List[str]:
    """Split text into chunks that comply with the API character limit."""
    words = text.strip().split()
    if not words:
        return []

    chunks: List[str] = []
    current_words: List[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        if word_len > max_chars:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_len = 0
            for start in range(0, word_len, max_chars):
                chunks.append(word[start : start + max_chars])
            continue

        additional_len = word_len if not current_words else word_len + 1
        if current_len + additional_len > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_len = word_len
        else:
            if current_words:
                current_len += 1
            current_words.append(word)
            current_len += word_len

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks

# -------------------
# Sarvam TTS/STT Endpoints
# -------------------
@app.get("/tts")
async def tts(text: str, language_code: str = "en-IN", speaker: str = "anushka"):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text is required for TTS")

    chunks = _chunk_text_for_tts(text, max_chars=480)
    if not chunks:
        raise HTTPException(status_code=400, detail="Unable to prepare text for TTS")

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=120) as client:
        tasks = [
            client.post(
                SARVAM_TTS_URL,
                headers=headers,
                json={
                    "inputs": [chunk],
                    "target_language_code": language_code,
                    "speaker": speaker,
                    "audio_format": "mp3"
                },
            )
            for chunk in chunks
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    audio_segments: List[bytes] = []
    for idx, result in enumerate(responses):
        if isinstance(result, Exception):
            raise HTTPException(status_code=500, detail=f"TTS chunk {idx} failed: {result}")

        if result.status_code != 200:
            return JSONResponse(
                status_code=result.status_code,
                content={"error": result.text, "chunk": idx}
            )

        data = result.json()
        if "audios" not in data or not data["audios"]:
            return JSONResponse(
                status_code=502,
                content={"error": "No audio returned from Sarvam", "chunk": idx}
            )

        audio_segments.append(base64.b64decode(data["audios"][0]))

    if not audio_segments:
        return JSONResponse({"error": "No audio returned from Sarvam"})

    combined_audio = b"".join(audio_segments)
    return Response(content=combined_audio, media_type="audio/mpeg")

@app.post("/stt")
async def stt_sarvam(file: UploadFile = File(...), language_code: Optional[str] = Query(None)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty audio file")

    headers = {"api-subscription-key": SARVAM_API_KEY}
    data = {"model": "saarika:v2.5"}
    if language_code:
        data["language_code"] = language_code

    files = {"file": (file.filename, contents, file.content_type)}

    resp = requests.post(SARVAM_STT_URL, headers=headers, data=data, files=files)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"STT error: {resp.text}")

    resp_json = resp.json()
    transcript = resp_json.get("transcript") or resp_json.get("text") or ""
    return JSONResponse({"transcript": transcript})

# -------------------
# Embedding & Retrieval
# -------------------
async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts using the remote Hugging Face API."""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(EMBEDDING_API_URL, json={"texts": texts})
            response.raise_for_status()
            return response.json()["embeddings"]
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Embedding API error: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

async def retrieve_context(query_vector: List[float], top_k: int = 5) -> str:
    """Search Pinecone for relevant context."""
    try:
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        if not results.matches:
            return ""
        return "\n\n---\n\n".join([m["metadata"]["text"] for m in results.matches])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {e}")

async def get_answer(query: str, context: str) -> str:
    """Use Groq LLM to answer based on retrieved context."""
    if not context.strip():
        return "Sorry I can't find the details."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful Business Document Assistant Chatbot speaking directly to a user. "
                    "Answer the user's question based ONLY on the provided context. "
                    "If the answer is not in the context, say: "
                    "'I can't tell you, please contact the nearest expert.'"
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        "temperature": 0.2,
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code != 200:
            error_detail = r.json()
            raise HTTPException(
                status_code=r.status_code,
                detail=f"Groq answering failed: {error_detail}",
            )
        return r.json()["choices"][0]["message"]["content"].strip()

# -------------------
# Chat Endpoint
# -------------------
@app.post("/chat")
async def chat(request: QueryRequest) -> Dict[str, Any]:
    try:
        query_vector = (await embed_texts([request.question]))[0]
        context = await retrieve_context(query_vector, top_k=5)

        # Debug logs
        print("User query:", request.question)
        print("Retrieved context:", context)

        answer = await get_answer(request.question, context)

        return {
            "original_query": request.question,
            "answer": answer,
            "context_found": bool(context.strip())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register")
async def register_user(payload: RegisterRequest) -> Dict[str, str]:
    collection = _require_user_collection()
    if not BREVO_API_KEY:
        logger.error("BREVO_API_KEY missing; cannot send OTP emails.")
        raise HTTPException(status_code=500, detail="Email service unavailable.")

    email = _normalize_email(payload.email)
    username = payload.username.strip()

    existing_user = await collection.find_one({
        "email": email,
        "otp": {"$exists": False},
    })
    if existing_user:
        raise HTTPException(status_code=400, detail="User already registered")

    otp = _generate_otp()
    await collection.update_one(
        {"email": email},
        {"$set": {"username": username, "email": email, "otp": otp}},
        upsert=True,
    )

    try:
        await _send_otp_email(email, otp)
        logger.info("OTP sent to %s", email)
    except httpx.HTTPStatusError as exc:
        logger.error("Brevo API error while sending OTP to %s: %s", email, exc.response.text)
        raise HTTPException(status_code=500, detail="Error sending OTP") from exc
    except httpx.HTTPError as exc:
        logger.error("Failed to send OTP to %s: %s", email, exc)
        raise HTTPException(status_code=500, detail="Error sending OTP") from exc

    return {"message": "OTP sent to email"}


@app.post("/verify-otp")
async def verify_otp(payload: VerifyOtpRequest) -> Dict[str, Any]:
    collection = _require_user_collection()
    email = _normalize_email(payload.email)

    temp_user = await collection.find_one({"email": email})
    if not temp_user or "otp" not in temp_user:
        raise HTTPException(status_code=404, detail="No OTP record")

    try:
        submitted_otp = int(payload.otp)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid OTP") from exc

    if temp_user["otp"] != submitted_otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    hashed_password = bcrypt.hashpw(payload.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    updated_user = await collection.find_one_and_update(
        {"email": email},
        {"$set": {"password": hashed_password}, "$unset": {"otp": ""}},
        return_document=ReturnDocument.AFTER,
    )

    if not updated_user:
        raise HTTPException(status_code=500, detail="Server error")

    token = _create_access_token(updated_user["email"])

    return {
        "message": "Verified successfully",
        "token": token,
        "user": {
            "username": updated_user.get("username"),
            "email": updated_user.get("email"),
        },
    }


@app.post("/login")
async def login_user(payload: LoginRequest) -> Dict[str, Any]:
    collection = _require_user_collection()
    identifier_raw = payload.identifier.strip()
    email_candidate = _normalize_email(identifier_raw)

    query_conditions = [
        {"username": identifier_raw},
        {"email": identifier_raw},
    ]
    if email_candidate != identifier_raw:
        query_conditions.append({"email": email_candidate})

    user = await collection.find_one(
        {
            "$or": query_conditions,
            "otp": {"$exists": False},
        }
    )
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    stored_password = user.get("password")
    if not stored_password:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    if not bcrypt.checkpw(payload.password.encode("utf-8"), stored_password.encode("utf-8")):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = _create_access_token(user["email"])

    return {
        "message": "Login successful",
        "token": token,
        "user": {
            "username": user.get("username"),
            "email": user.get("email"),
        },
    }


@app.get("/")
def read_root():
    return {"status": "Customer support API is online"}