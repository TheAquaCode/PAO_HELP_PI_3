"""
voice.py — Atlas Voice Node (runs on Pi 3)

Listens continuously for the wake word "hey atlas",
then records speech, transcribes with Whisper,
and sends the text to Pi 5.

Usage:
    python3 voice.py
"""

import sounddevice as sd
import numpy as np
import whisper
import requests
import tempfile
import scipy.io.wavfile as wav
import time
import os

# --- Config ---
PI5_URL = os.environ.get("ATLAS_PI5_URL", "http://pi5.local:5000/chat")
SAMPLE_RATE = 16000
WAKE_WORD = "hey atlas"
WAKE_LISTEN_SECONDS = 2      # short clip to check for wake word
RECORD_SECONDS = 6           # how long to record after wake word
SILENCE_THRESHOLD = 300       # amplitude below this = silence
WHISPER_MODEL = os.environ.get("ATLAS_WHISPER_MODEL", "tiny")

# --- Load Whisper model ---
print("[*] Loading Whisper model...")
model = whisper.load_model(WHISPER_MODEL)
print(f"[✓] Whisper '{WHISPER_MODEL}' loaded. Listening for '{WAKE_WORD}'...\n")


def record_audio(duration, sample_rate=SAMPLE_RATE):
    """Record audio from the default microphone."""
    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16"
        )
        sd.wait()
        return audio, sample_rate
    except Exception as e:
        print(f"  [!] Microphone error: {e}")
        return None, sample_rate


def transcribe(audio, sample_rate):
    """Save audio to a temp file and run Whisper on it."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, sample_rate, audio)
            result = model.transcribe(f.name)
            text = result["text"].strip()
            os.unlink(f.name)
            return text
    except Exception as e:
        print(f"  [!] Transcription error: {e}")
        return ""


def has_speech(audio, threshold=SILENCE_THRESHOLD):
    """Check if the audio clip contains any meaningful sound."""
    return np.max(np.abs(audio)) > threshold


def send_to_pi5(text, persona="assistant"):
    """Send transcribed text to Pi 5 Flask endpoint."""
    try:
        print(f"  [>] Sending to Pi 5: '{text}'")
        response = requests.post(
            PI5_URL,
            json={"message": text, "persona": persona},
            timeout=60,
            stream=True
        )
        if response.ok:
            # Read the streamed response
            full = ""
            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        token = decoded[6:]
                        if token == "[DONE]":
                            break
                        full += token
            print(f"  [<] Atlas: {full}")
            return full
        else:
            print(f"  [!] Pi 5 returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("  [!] Cannot reach Pi 5 — is it running?")
    except requests.exceptions.Timeout:
        print("  [!] Pi 5 request timed out")
    except Exception as e:
        print(f"  [!] Network error: {e}")
    return None


# --- Main loop ---
print("=" * 40)
print(f"  Say '{WAKE_WORD}' to start talking")
print("=" * 40)

while True:
    try:
        # Step 1: Listen for wake word (short clips)
        audio, rate = record_audio(WAKE_LISTEN_SECONDS)
        if audio is None:
            time.sleep(1)
            continue

        # Skip if silence
        if not has_speech(audio):
            continue

        # Transcribe the short clip to check for wake word
        text = transcribe(audio, rate).lower()

        if WAKE_WORD not in text:
            continue

        # Step 2: Wake word detected — record the actual question
        print("\n[*] Wake word detected! Recording your question...")
        audio, rate = record_audio(RECORD_SECONDS)
        if audio is None:
            continue

        if not has_speech(audio):
            print("  [!] No speech detected, going back to listening.")
            continue

        # Step 3: Transcribe the question
        print("  [*] Transcribing...")
        question = transcribe(audio, rate)

        if not question:
            print("  [!] Could not transcribe, going back to listening.")
            continue

        print(f"  [✓] You said: '{question}'")

        # Step 4: Send to Pi 5
        send_to_pi5(question)

        print(f"\n[*] Listening for '{WAKE_WORD}'...\n")

    except KeyboardInterrupt:
        print("\n[*] Shutting down voice node.")
        break
    except Exception as e:
        print(f"[!] Unexpected error: {e}")
        time.sleep(2)