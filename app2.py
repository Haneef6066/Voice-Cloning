from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from TTS.api import TTS
import os, uuid, json
from pydub import AudioSegment
import shutil

# Keep a small, focused server: upload/create clone, list clones, preview clone,
# and synthesize text by referencing a clone id/name. Old behavior removed.

# Basic FFmpeg/path handling (keep in case pydub needs it)
ffmpeg_exe = shutil.which("ffmpeg")
if ffmpeg_exe:
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

import torchaudio
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

try:
    import soundfile as _sf
    import numpy as _np
    import torch as _torch

    def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True):
        data, sr = _sf.read(path, dtype='float32')
        if data.ndim == 1:
            data = _np.expand_dims(data, 1)
        data = data.T
        tensor = _torch.from_numpy(data.copy())
        return tensor, sr

    torchaudio.load = _sf_load
except Exception:
    pass

# PyTorch safe globals needed for XTTS checkpoints
import torch
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# Flask app and folders
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
CLONED_FOLDER = "cloned"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLONED_FOLDER, exist_ok=True)

# voices metadata file (maps id -> {id,name,filename,created})
VOICES_FILE = os.path.join(CLONED_FOLDER, "voices.json")

def load_voices():
    if os.path.exists(VOICES_FILE):
        try:
            with open(VOICES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_voices(v):
    try:
        with open(VOICES_FILE, 'w', encoding='utf-8') as f:
            json.dump(v, f, indent=2)
    except Exception:
        app.logger.exception('Failed to save voices.json')

# load XTTS model (GPU disabled by default; enable if you have CUDA)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/voices', methods=['GET'])
def voices_list():
    """Return list of created voice clones."""
    voices = load_voices()
    # return as list newest-first
    arr = sorted(voices.values(), key=lambda x: x.get('created', ''), reverse=True)
    return jsonify({'voices': arr})


@app.route('/create_clone', methods=['POST'])
def create_clone():
    """Create a named voice clone from uploaded/recorded audio.

    Client must POST form-data with 'audio' file and optional 'name' and 'language' and 'sample_text'.
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    file = request.files['audio']
    name = request.form.get('name') or f"voice_{uuid.uuid4().hex[:6]}"
    # default sample text and language set to Hindi for Hindi-first workflow
    sample_text = request.form.get('sample_text') or 'नमस्ते, यह एक क्लोन नमूना आवाज़ है।'
    language = request.form.get('language') or 'hi'

    # save uploaded file (preserve extension)
    fname = secure_filename(file.filename or (name + '.wav'))
    base, ext = os.path.splitext(fname)
    if not ext:
        ext = '.wav'
    vid = uuid.uuid4().hex[:8]
    saved_name = f"{name}_{vid}{ext}"
    saved_path = os.path.join(CLONED_FOLDER, saved_name)
    file.save(saved_path)

    # if not wav, convert
    if ext.lower() != '.wav':
        try:
            wav_path = os.path.join(CLONED_FOLDER, f"{name}_{vid}.wav")
            sound = AudioSegment.from_file(saved_path)
            sound.export(wav_path, format='wav')
            try:
                os.remove(saved_path)
            except Exception:
                pass
            saved_name = os.path.basename(wav_path)
            saved_path = wav_path
        except Exception as e:
            app.logger.exception('Failed to convert uploaded audio')
            return jsonify({'error': 'Failed to convert to WAV'}), 500

    # add to voices.json
    voices = load_voices()
    vid_key = vid
    voices[vid_key] = {
        'id': vid_key,
        'name': name,
        'file': saved_name,
        'created': __import__('datetime').datetime.now().isoformat()
    }
    save_voices(voices)

    # synthesize a short sample and save it as sample_<id>.wav
    sample_name = f"sample_{vid_key}.wav"
    sample_path = os.path.join(CLONED_FOLDER, sample_name)
    try:
        tts.tts_to_file(text=sample_text, file_path=sample_path, speaker_wav=saved_path, language=language)
    except Exception:
        app.logger.exception('Failed to synthesize sample for new clone')

    return jsonify({'id': vid_key, 'name': name, 'file': saved_name, 'sample': sample_name})


@app.route('/cloned/<path:filename>')
def serve_cloned(filename):
    safe = os.path.basename(filename)
    full = os.path.join(CLONED_FOLDER, safe)
    if not os.path.exists(full):
        return jsonify({'error': 'not found'}), 404
    return send_from_directory(CLONED_FOLDER, safe)


@app.route('/speak', methods=['POST'])
def speak():
    """Synthesize text using a named clone. JSON body: { text, voice_id }"""
    data = request.get_json() or {}
    text = data.get('text')
    voice_id = data.get('voice_id')
    # default language to Hindi to match create_clone default
    language = data.get('language', 'hi')

    if not text or not voice_id:
        return jsonify({'error': 'text and voice_id required'}), 400

    voices = load_voices()
    v = voices.get(voice_id)
    if not v:
        return jsonify({'error': 'voice_id not found'}), 404

    speaker_file = os.path.join(CLONED_FOLDER, v['file'])
    if not os.path.exists(speaker_file):
        return jsonify({'error': 'speaker file missing'}), 500

    out_name = f"tts_{voice_id}_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(CLONED_FOLDER, out_name)
    try:
        tts.tts_to_file(text=text, file_path=out_path, speaker_wav=speaker_file, language=language)
    except Exception as e:
        app.logger.exception('TTS failed')
        return jsonify({'error': 'TTS failed', 'detail': str(e)}), 500

    return send_from_directory(CLONED_FOLDER, out_name, mimetype='audio/wav')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


# Serve a simple favicon to avoid browser /favicon.ico 404s
@app.route('/favicon.ico')
def favicon():
    fav_path = os.path.join('static', 'favicon.ico')
    if os.path.exists(fav_path):
        return send_from_directory('static', 'favicon.ico')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
           '<rect width="16" height="16" fill="#007bff"/></svg>')
    from flask import Response
    return Response(svg, mimetype='image/svg+xml')

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from TTS.api import TTS
import os, uuid, json
from pydub import AudioSegment
import shutil

# Keep a small, focused server: upload/create clone, list clones, preview clone,
# and synthesize text by referencing a clone id/name. Old behavior removed.

# Basic FFmpeg/path handling (keep in case pydub needs it)
ffmpeg_exe = shutil.which("ffmpeg")
if ffmpeg_exe:
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

import torchaudio
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

try:
    import soundfile as _sf
    import numpy as _np
    import torch as _torch

    def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True):
        data, sr = _sf.read(path, dtype='float32')
        if data.ndim == 1:
            data = _np.expand_dims(data, 1)
        data = data.T
        tensor = _torch.from_numpy(data.copy())
        return tensor, sr

    torchaudio.load = _sf_load
except Exception:
    pass

# PyTorch safe globals needed for XTTS checkpoints
import torch
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

# Flask app and folders
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
CLONED_FOLDER = "cloned"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLONED_FOLDER, exist_ok=True)

# voices metadata file (maps id -> {id,name,filename,created})
VOICES_FILE = os.path.join(CLONED_FOLDER, "voices.json")

def load_voices():
    if os.path.exists(VOICES_FILE):
        try:
            with open(VOICES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_voices(v):
    try:
        with open(VOICES_FILE, 'w', encoding='utf-8') as f:
            json.dump(v, f, indent=2)
    except Exception:
        app.logger.exception('Failed to save voices.json')

# load XTTS model (GPU disabled by default; enable if you have CUDA)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/voices', methods=['GET'])
def voices_list():
    """Return list of created voice clones."""
    voices = load_voices()
    # return as list newest-first
    arr = sorted(voices.values(), key=lambda x: x.get('created', ''), reverse=True)
    return jsonify({'voices': arr})


@app.route('/create_clone', methods=['POST'])
def create_clone():
    """Create a named voice clone from uploaded/recorded audio.

    Client must POST form-data with 'audio' file and optional 'name' and 'language' and 'sample_text'.
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    file = request.files['audio']
    name = request.form.get('name') or f"voice_{uuid.uuid4().hex[:6]}"
    # default sample text and language set to Hindi for Hindi-first workflow
    sample_text = request.form.get('sample_text') or 'नमस्ते, यह एक क्लोन नमूना आवाज़ है।'
    language = request.form.get('language') or 'hi'

    # save uploaded file (preserve extension)
    fname = secure_filename(file.filename or (name + '.wav'))
    base, ext = os.path.splitext(fname)
    if not ext:
        ext = '.wav'
    vid = uuid.uuid4().hex[:8]
    saved_name = f"{name}_{vid}{ext}"
    saved_path = os.path.join(CLONED_FOLDER, saved_name)
    file.save(saved_path)

    # if not wav, convert
    if ext.lower() != '.wav':
        try:
            wav_path = os.path.join(CLONED_FOLDER, f"{name}_{vid}.wav")
            sound = AudioSegment.from_file(saved_path)
            sound.export(wav_path, format='wav')
            try:
                os.remove(saved_path)
            except Exception:
                pass
            saved_name = os.path.basename(wav_path)
            saved_path = wav_path
        except Exception as e:
            app.logger.exception('Failed to convert uploaded audio')
            return jsonify({'error': 'Failed to convert to WAV'}), 500

    # add to voices.json
    voices = load_voices()
    vid_key = vid
    voices[vid_key] = {
        'id': vid_key,
        'name': name,
        'file': saved_name,
        'created': __import__('datetime').datetime.now().isoformat()
    }
    save_voices(voices)

    # synthesize a short sample and save it as sample_<id>.wav
    sample_name = f"sample_{vid_key}.wav"
    sample_path = os.path.join(CLONED_FOLDER, sample_name)
    try:
        tts.tts_to_file(text=sample_text, file_path=sample_path, speaker_wav=saved_path, language=language)
    except Exception:
        app.logger.exception('Failed to synthesize sample for new clone')

    return jsonify({'id': vid_key, 'name': name, 'file': saved_name, 'sample': sample_name})


@app.route('/cloned/<path:filename>')
def serve_cloned(filename):
    safe = os.path.basename(filename)
    full = os.path.join(CLONED_FOLDER, safe)
    if not os.path.exists(full):
        return jsonify({'error': 'not found'}), 404
    return send_from_directory(CLONED_FOLDER, safe)


@app.route('/speak', methods=['POST'])
def speak():
    """Synthesize text using a named clone. JSON body: { text, voice_id }"""
    data = request.get_json() or {}
    text = data.get('text')
    voice_id = data.get('voice_id')
    # default language to Hindi to match create_clone default
    language = data.get('language', 'hi')

    if not text or not voice_id:
        return jsonify({'error': 'text and voice_id required'}), 400

    voices = load_voices()
    v = voices.get(voice_id)
    if not v:
        return jsonify({'error': 'voice_id not found'}), 404

    speaker_file = os.path.join(CLONED_FOLDER, v['file'])
    if not os.path.exists(speaker_file):
        return jsonify({'error': 'speaker file missing'}), 500

    out_name = f"tts_{voice_id}_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(CLONED_FOLDER, out_name)
    try:
        tts.tts_to_file(text=text, file_path=out_path, speaker_wav=speaker_file, language=language)
    except Exception as e:
        app.logger.exception('TTS failed')
        return jsonify({'error': 'TTS failed', 'detail': str(e)}), 500

    return send_from_directory(CLONED_FOLDER, out_name, mimetype='audio/wav')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


# Serve a simple favicon to avoid browser /favicon.ico 404s
@app.route('/favicon.ico')
def favicon():
    fav_path = os.path.join('static', 'favicon.ico')
    if os.path.exists(fav_path):
        return send_from_directory('static', 'favicon.ico')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
           '<rect width="16" height="16" fill="#007bff"/></svg>')
    from flask import Response
    return Response(svg, mimetype='image/svg+xml')