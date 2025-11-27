import shutil, os
# import the app so it can apply the torchaudio.load monkey-patch
import app
print('Python executable:', os.sys.executable)
print('ffmpeg ->', shutil.which('ffmpeg'))
try:
    import torchaudio
    print('torchaudio version ->', getattr(torchaudio, '__version__', 'n/a'))
except Exception as e:
    print('Failed to import torchaudio:', e)

uploads_dir = 'uploads'
if not os.path.isdir(uploads_dir):
    print('uploads folder missing')
    raise SystemExit(1)

wavs = [f for f in os.listdir(uploads_dir) if f.lower().endswith('.wav')]
if not wavs:
    print('No WAV files found in uploads/. Please upload one via the web UI first')
    raise SystemExit(2)

p = os.path.join(uploads_dir, wavs[0])
print('Testing load of:', p)
try:
    wav, sr = torchaudio.load(p)
    print('Loaded OK: shape=', wav.shape, 'sr=', sr)
except Exception as e:
    print('torchaudio.load failed:', repr(e))
