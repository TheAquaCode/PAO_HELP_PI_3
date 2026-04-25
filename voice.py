import sounddevice as sd
import numpy as np
import whisper
import requests
import tempfile
import scipy.io.wavfile as wav
import speech_recognition as sr
import threading
import time
from flask import Flask, jsonify

# ── Config ──
PI5_BASE   = "http://pi5.local:5000"
PI5_VOICE  = f"{PI5_BASE}/voice"
PI5_STATUS = f"{PI5_BASE}/pi3-status-feed"
PI5_NOTIFY = f"{PI5_BASE}/notify"
SAMPLE_RATE = 16000
DURATION    = 10
MODEL_SIZE  = "tiny"

WAKE_WORDS = [
    "atlas", "hey atlas", "hey at last", "hey at les",
    "hey atl", "atlas help", "ok atlas", "okay atlas"
]

# ── Ping server ──
ping_app = Flask(__name__)

@ping_app.route("/ping")
def ping():
    return "ok", 200

threading.Thread(
    target=lambda: ping_app.run(host="0.0.0.0", port=5001, use_reloader=False),
    daemon=True
).start()

# ── Send UI event to Pi 5 ──
def notify(event_type, data=""):
    try:
        requests.post(PI5_NOTIFY, json={"type": event_type, "data": data}, timeout=2)
    except:
        pass

# ── Pi 5 status polling for Windows SSH terminal ──
def poll_pi5_status():
    last = ""
    while True:
        try:
            r = requests.get(PI5_STATUS, timeout=2)
            if r.ok:
                msg = r.json().get("status", "")
                if msg and msg != last:
                    print(f"[Pi 5] {msg}")
                    last = msg
        except:
            pass
        time.sleep(1)

threading.Thread(target=poll_pi5_status, daemon=True).start()

# ── Load Whisper ──
print("Loading Whisper model...")
model = whisper.load_model(MODEL_SIZE)
print("Model loaded.")
print("Say 'Atlas' to wake up.\n")

def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )
    for i in range(duration, 0, -1):
        notify("countdown", i)
        print(f"  {i}s remaining...")
        time.sleep(1)
    sd.wait()
    print("Recording done.")
    return audio, sample_rate

def transcribe(audio, sample_rate):
    notify("transcribing", "")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, sample_rate, audio)
        print("Transcribing...")
        result = model.transcribe(f.name)
        text = result["text"].strip()
        print(f"You said: {text}")
        notify("heard", text)
        return text

def send_to_pi5(text):
    try:
        # Tell screen we are processing — Pi 5 will send done via ai_message
        notify("processing", text)
        print("Sending to Pi 5...")
        response = requests.post(
            PI5_VOICE,
            json={"message": text},
            timeout=180
        )
        if response.ok:
            reply = response.json().get("response", "")
            print(f"Atlas: {reply}\n")
            # Do NOT send done here — Pi 5 already pushed ai_message to the screen
            return reply
        else:
            print(f"Pi 5 error: {response.status_code}")
            notify("error", f"Pi 5 error {response.status_code}")
    except requests.exceptions.ReadTimeout:
        print("Pi 5 took too long.")
        notify("error", "Response timed out")
    except requests.exceptions.ConnectionError:
        print("Could not reach Pi 5.")
        notify("error", "Cannot reach Pi 5")

def check_wake_word(text):
    return any(w in text.lower() for w in WAKE_WORDS)

def listen_for_wake_word(recognizer, mic):
    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            text = recognizer.recognize_google(audio).lower()
            print(f"[Heard]: {text}")
            if check_wake_word(text):
                print("\n✓ Wake word detected!")
                notify("wake_detected", "")
                return True
        except sr.WaitTimeoutError:
            pass
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            print("Google speech unavailable — falling back to Enter key.")
            return False

# ── Main ──
recognizer = sr.Recognizer()
mic = sr.Microphone(sample_rate=SAMPLE_RATE)

print("Calibrating microphone...")
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=2)
print("Done. Listening for 'Atlas'...\n")

notify("idle", "")

while True:
    try:
        detected = listen_for_wake_word(recognizer, mic)

        if not detected:
            input("Press Enter to record (Ctrl+C to quit)...")
            notify("wake_detected", "")

        audio, rate = record_audio()
        text = transcribe(audio, rate)

        if text:
            send_to_pi5(text)

        notify("idle", "")
        print("Listening for 'Atlas'...")

    except KeyboardInterrupt:
        print("\nStopped.")
        notify("idle", "")
        break
