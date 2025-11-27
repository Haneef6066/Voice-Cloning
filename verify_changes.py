import requests
import json
import os

BASE_URL = "http://127.0.0.1:5000"

def test_advanced_tts():
    print("1. Getting voices list...")
    try:
        res = requests.get(f"{BASE_URL}/voices")
        voices = res.json().get('voices', [])
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    if not voices:
        print("No voices found. Please create a voice clone first via the UI.")
        return

    voice_id = voices[0]['id']
    print(f"Using voice: {voices[0]['name']} ({voice_id})")

    params = {
        "text": "This is a test of the advanced tuning parameters.",
        "voice_id": voice_id,
        "temperature": 0.1,  # Low temp
        "top_p": 0.5,
        "top_k": 10,
        "repetition_penalty": 2.0
    }

    print(f"2. Synthesizing with params: {json.dumps(params, indent=2)}")
    
    try:
        res = requests.post(f"{BASE_URL}/speak", json=params)
        if res.status_code == 200:
            print("Success! Audio content received.")
            with open("test_output.wav", "wb") as f:
                f.write(res.content)
            print("Saved to test_output.wav")
        else:
            print(f"Failed: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"Error during synthesis: {e}")

if __name__ == "__main__":
    test_advanced_tts()
