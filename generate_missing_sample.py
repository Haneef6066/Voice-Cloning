import sys
sys.path.insert(0, 'd:\\voice_clone_app1')

from app import get_tts
import os

# Generate the missing sample
voice_id = "600073da"
speaker_file = f"d:\\voice_clone_app1\\cloned\\ravi_ind_eng_{voice_id}.wav"
sample_path = f"d:\\voice_clone_app1\\cloned\\sample_{voice_id}.wav"

if os.path.exists(speaker_file):
    print(f"Generating sample for voice {voice_id}...")
    try:
        get_tts().tts_to_file(
            text="This is a cloned voice sample.",
            file_path=sample_path,
            speaker_wav=speaker_file,
            language="hi"  # Using Hindi for Indian accent
        )
        print(f"Sample generated successfully: {sample_path}")
    except Exception as e:
        print(f"Failed to generate sample: {e}")
else:
    print(f"Speaker file not found: {speaker_file}")
