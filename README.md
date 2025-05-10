# Sesame CSM UI

This repository contains Gradio app for locally running the Conversational Speech Model (CSM) with support for both CUDA, MLX (Apple Silicon) and CPU backends.

Check my blog post for [Setup Instructions](https://voipnuggets.com/2025/03/21/sesame-csm-gradio-ui-free-local-high-quality-text-to-speech-with-voice-cloning-cuda-apple-mlx-and-cpu/)

## Sample Audio
<div align="center">
  <a href="https://voipnuggets.wordpress.com/wp-content/uploads/2025/03/audio.wav">
    <img src="https://img.shields.io/badge/🔊_Listen_to_Sample-blue?style=for-the-badge" alt="Listen to Sample" />
  </a>
</div>

Generate your own samples using the UI.

## UI Screenshots:
![Gradio UI](assets/gradio-ui.jpg)
![Voice Clone](assets/speaker-voice.jpeg)

## Installation

VRAM needed to run the model is around 8.1 GB on MLX, 4.5 on CUDA GPU and 8.5GB on CPU.
### Setup

```bash
git clone git@github.com:SesameAILabs/csm.git
cd csm
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
You need to have access to these models on huggingface:

Llama-3.2-1B -- https://huggingface.co/meta-llama/Llama-3.2-1B

CSM-1B -- https://huggingface.co/sesame/csm-1b

Login to hugging face and request access, it should not take much time to get access
Once you get the access run the following command on the terminal to login into huggingface account
```bash
huggingface-cli login
```

## Usage

### Gradio Web Interface

Use `run_csm_gradio.py` to launch an interactive web interface:

```bash
python run_csm_gradio.py
```

Features:
- Interactive web UI for conversation generation
- Custom prompt selection for each speaker (Voice Cloning)
- Real-time audio preview
- Automatic backend selection (CUDA/MLX/CPU)

### Command Line Interface

Use `run_csm.py` to generate a conversation and save it to a WAV file:

```bash
python run_csm.py
```

The script will:
1. Automatically select the best available backend:
   - CUDA if NVIDIA GPU is available
   - MLX if running on Apple Silicon
   - CPU as fallback
2. Generate a sample conversation between two speakers
3. Save the output as `full_conversation.wav`

## Backends

The scripts support three backends:

1. **CUDA** (NVIDIA GPU)
   - Fastest on NVIDIA hardware
   - Uses PyTorch implementation

2. **MLX** (Apple Silicon)
   - Optimized for M1/M2/M3 Macs
   - Uses Apple's MLX framework
   - Automatically selected on Apple Silicon

3. **CPU**
   - Fallback option
   - Works on all platforms
   - Uses PyTorch implementation

CSM API Documentation

Getting Started
--------------
Run the API server using:
python serve_api.py

API Endpoints
------------

1. Text-to-Speech Generation
   Endpoint: POST /v1/audio/speech
   
   Generate speech from text using the specified voice.
   
   Request Body:
   {
       "model": "csm-1b",
       "input": "Text to convert to speech",
       "voice": "speaker_0",
       "response_format": "audio/wav"
   }
   
   Response:
   {
       "content_type": "audio/wav",
       "audio": "base64_encoded_audio_data"
   }

2. Voice Cloning
   Endpoint: POST /v1/audio/speech/clone
   
   Clone a new voice from an audio sample.
   
   Request Body:
   {
       "audio_data": "base64_encoded_audio_file",
       "response_format": "audio/wav"
   }
   
   Response:
   {
       "voice_id": "speaker_1",
       "status": "success"
   }

Python Client Example
-------------------
from openai import OpenAI
import base64

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# Generate speech
response = client.audio.speech.create(
    model="csm-1b",
    voice="speaker_0",
    input="Hello, this is a test!"
)
response.stream_to_file("output.wav")

# Clone a voice
with open("voice_sample.wav", "rb") as audio_file:
    audio_data = base64.b64encode(audio_file.read()).decode()
    
response = client.post(
    "audio/speech/clone",
    json={
        "audio_data": audio_data
    }
)
new_voice_id = response.json()["voice_id"]

# Use the cloned voice
response = client.audio.speech.create(
    model="csm-1b",
    voice=new_voice_id,
    input="Hello with the cloned voice!"
)
response.stream_to_file("cloned_output.wav")

Notes
-----
- Server automatically selects appropriate backend (CUDA/MPS/CPU)
- All audio is returned in WAV format
- Voice IDs are in the format speaker_N where N is an integer
- API is compatible with OpenAI's TTS API

## Requirements

- Python 3.10+
- PyTorch
- MLX (for Apple Silicon)
- Gradio
- torchaudio
- litserve
- fastapi
- uvicorn
- pydantic
- Other dependencies listed in requirements.txt

## Credits

- Original PyTorch implementation by [Sesame](https://github.com/SesameAILabs/csm)
- MLX port by [senstella](https://github.com/senstella/csm-mlx)

# CSM
**2025/03/25** - I am releasing support for API endpoints.

**2025/03/20** - I am releasing support for Apple MLX for Mac device. The UI will auto select the backend from CUDA, MPS or CPU. The MLX code is an adaptation from [Senstella/csm-mlx](https://github.com/senstella/csm-mlx)

**2025/03/15** - I am releasing support for CPU for non-CUDA device. I am relasing a Gradio UI as well.

**2025/03/13** - We are releasing the 1B CSM variant. The checkpoint is [hosted on HuggingFace](https://huggingface.co/sesame/csm_1b).

---

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. The model architecture employs a [Llama](https://www.llama.com/) backbone and a smaller audio decoder that produces [Mimi](https://huggingface.co/kyutai/mimi) audio codes.

A fine-tuned variant of CSM powers the [interactive voice demo](https://www.sesame.com/voicedemo) shown in our [blog post](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice).

A hosted [Hugging Face space](https://huggingface.co/spaces/sesame/csm-1b) is also available for testing audio generation.

### Python API

Generate a single utterance:

```python
from generator import load_csm_1b
import torchaudio
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = load_csm_1b(device=device)

audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

CSM sounds best when provided with context. You can prompt or provide context to the model using a `Segment` for each speaker's utterance.

```python
speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## FAQ

**Does this model come with any voices?**

The model open-sourced here is a base generation model. It is capable of producing a variety of voices, but it has not been fine-tuned on any specific voice.

**Can I converse with the model?**

CSM is trained to be an audio generation model and not a general-purpose multimodal LLM. It cannot generate text. We suggest using a separate LLM for text generation.

**Does it support other languages?**

The model has some capacity for non-English languages due to data contamination in the training data, but it likely won't do well.

## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

---

## Authors
Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team.
