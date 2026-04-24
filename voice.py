import sounddevice as sd
import numpy as np
import whisper
import requests
import tempfile
import scipy.io.wavfile as wav

# --- Config ---
PI5_URL = "http://pi5.local:5000/chat"
SAMPLE_RATE = 16000
DURATION = 5  # seconds to record, adjust as needed
MODEL_SIZE = "tiny"  # tiny is fastest on Pi 3, can use "base" for better accuracy

# --- Load Whisper model ---
print("Loading Whisper model...")
model = whisper.load_model(MODEL_SIZE)
print("Model loaded. Ready!")

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    print(f"\nRecording for {duration} seconds... speak now!")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )
    sd.wait()
    print("Recording done.")
    return audio, sample_rate

def transcribe(audio, sample_rate):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, sample_rate, audio)
        print("Transcribing...")
        result = model.transcribe(f.name)
        text = result["text"].strip()
        print(f"You said: {text}")
        return text

def send_to_pi5(text):
    try:
        print(f"Sending to Pi 5...")
        response = requests.post(PI5_URL, data={"message": text}, timeout=30)
        if response.ok:
            reply = response.text
            print(f"Pi 5 replied: {reply}")
            return reply
        else:
            print(f"Pi 5 returned error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("Could not reach Pi 5 — is it running?")

# --- Main loop ---
while True:
    input("\nPress Enter to start recording (Ctrl+C to quit)...")
    audio, rate = record_audio()
    text = transcribe(audio, rate)
    if text:
        send_to_pi5(text)