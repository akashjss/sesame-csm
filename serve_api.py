from typing import Optional
from litserve import Server, AsyncRouter
from fastapi import FastAPI
from pydantic import BaseModel
from generator import load_csm_1b
import torch
import torchaudio
import io
import base64

# Initialize model with appropriate backend
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = load_csm_1b(device=device)

class SpeechRequest(BaseModel):
    model: str = "csm-1b"
    input: str
    voice: str = "speaker_0"
    response_format: Optional[str] = "audio/wav"

class CloneVoiceRequest(BaseModel):
    audio_data: str  # base64 encoded audio file
    response_format: Optional[str] = "audio/wav"

router = AsyncRouter()

@router.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    # Convert speaker string to int
    speaker = int(request.voice.split("_")[1])
    
    # Generate audio
    audio = generator.generate(
        text=request.input,
        speaker=speaker,
        context=[],
        max_audio_length_ms=10_000,
    )
    
    # Convert to WAV bytes
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio.unsqueeze(0).cpu(), generator.sample_rate, format="wav")
    buffer.seek(0)
    
    return {
        "content_type": "audio/wav",
        "audio": base64.b64encode(buffer.read()).decode()
    }

@router.post("/v1/audio/speech/clone")
async def clone_voice(request: CloneVoiceRequest):
    # Decode audio data
    audio_bytes = base64.b64decode(request.audio_data)
    audio_buffer = io.BytesIO(audio_bytes)
    
    # Load audio
    audio_tensor, sample_rate = torchaudio.load(audio_buffer)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), 
        orig_freq=sample_rate, 
        new_freq=generator.sample_rate
    )
    
    # Clone the voice using the generator
    new_speaker_id = generator.clone_voice(audio_tensor)
    
    return {
        "voice_id": f"speaker_{new_speaker_id}",
        "status": "success"
    }

app = FastAPI(title="CSM API")
server = Server(app)
server.add_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 