# sarvam_backend.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import requests
import base64

from typing import Optional

app = FastAPI()

# Allow your React origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SARVAM_API_KEY = "sk_f8fjoda1_s83hQcvwfwwmPIwImLdTaReh"
print(SARVAM_API_KEY)


SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
@app.get("/tts")
def tts(text: str, language_code: str = "en-IN", speaker: str = "anushka"):
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": [text],                # ✅ Sarvam expects array
        "target_language_code": language_code,
        "speaker": speaker,
        "audio_format": "mp3"
    }

    response = requests.post(SARVAM_TTS_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return JSONResponse(
            status_code=response.status_code,
            content={"error": response.text}
        )

    data = response.json()

    if "audios" in data and len(data["audios"]) > 0:
        audio_bytes = base64.b64decode(data["audios"][0])
        return Response(content=audio_bytes, media_type="audio/mpeg")

    return JSONResponse({"error": "No audio returned from Sarvam"})

@app.post("/stt")
async def stt_sarvam(file: UploadFile = File(...), language_code: Optional[str] = Query(None)):
    """
    Send audio file to Sarvam STT and return transcript.
    """
    # read file
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # send to Sarvam STT REST API
    url = "https://api.sarvam.ai/speech-to-text"  # Adjust if different endpoint
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
    }
    data = {}
    if language_code:
        data["language_code"] = language_code
    # model selection if needed, e.g. model="saarika:v2.5"
    data["model"] = "saarika:v2.5"

    files = {
        "file": (file.filename, contents, file.content_type)
    }

    resp = requests.post(url, headers=headers, data=data, files=files)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"STT error: {resp.text}")
    resp_json = resp.json()
    # According to Sarvam docs, output might include “transcript” key
    transcript = resp_json.get("transcript") or resp_json.get("text") or ""
    return JSONResponse({"transcript": transcript})