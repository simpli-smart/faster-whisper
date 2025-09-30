## CLI usage: VAD and Diarization

### Prerequisites
- Pyannote models are gated on Hugging Face. Accept the terms for `pyannote/segmentation-3.0` and `pyannote/speaker-diarization-3.1`, then set:

```bash
export PYANNOTE_AUTH_TOKEN="hf_your_token_here"
```

### List available VAD backends
```bash
python faster_whisper_cli.py --list-vad-backends
```

### Transcription with VAD

- Silero VAD (default) and save output
```bash
python faster_whisper_cli.py /path/to/audio.mp3 --model base \
  --vad-backend silero --output transcript_silero.txt
```

- Silero VAD (tuned)
```bash
python faster_whisper_cli.py /path/to/audio.mp3 --vad-backend silero \
  --vad-threshold 0.35 --vad-min-speech-duration-ms 300 \
  --vad-speech-pad-ms 300 --vad-min-silence-duration-ms 1200
```

- Pyannote VAD
```bash
python faster_whisper_cli.py /path/to/audio.mp3 --model large-v3 \
  --vad-backend pyannote --pyannote-auth-token "$PYANNOTE_AUTH_TOKEN"
```

- Pyannote VAD (tuned)
```bash
python faster_whisper_cli.py /path/to/audio.mp3 --vad-backend pyannote \
  --pyannote-auth-token "$PYANNOTE_AUTH_TOKEN" \
  --pyannote-model pyannote/segmentation-3.0 \
  --vad-min-duration-on 0.1 --vad-min-duration-off 0.1
```

- Weighted VAD (confidence fusion)
```bash
python faster_whisper_cli.py /path/to/audio.mp3 --vad-backend weighted \
  --combination-method confidence --silero-weight 0.6 --pyannote-weight 0.4 \
  --combination-threshold 0.5 --pyannote-auth-token "$PYANNOTE_AUTH_TOKEN"
```

- Weighted VAD (majority fusion)
```bash
python faster_whisper_cli.py /path/to/audio.mp3 --vad-backend weighted \
  --combination-method majority --pyannote-auth-token "$PYANNOTE_AUTH_TOKEN"
```

- Word-level timestamps
```bash
python faster_whisper_cli.py /path/to/audio.mp3 --vad-backend pyannote \
  --word-timestamps --pyannote-auth-token "$PYANNOTE_AUTH_TOKEN"
```

- Device/compute examples
```bash
python faster_whisper_cli.py /path/to/audio.mp3 --vad-backend pyannote \
  --device cuda --compute-type float16 --pyannote-auth-token "$PYANNOTE_AUTH_TOKEN"
```

### Diarization (Pyannote)
```bash
python diarization_system.py /path/to/audio.mp3 --backend pyannote \
  --pyannote-auth-token "$PYANNOTE_AUTH_TOKEN" --output diarization.json
```

### Optional: suppress torchaudio deprecation warnings
```bash
PYTHONWARNINGS="ignore:.*torchaudio._backend.*:UserWarning" \
python faster_whisper_cli.py /path/to/audio.mp3 --vad-backend pyannote \
  --pyannote-auth-token "$PYANNOTE_AUTH_TOKEN"
```

### Notes
- Supported VAD backends: `silero`, `pyannote`, `weighted` (confidence | majority).
- Supported diarization backend: `pyannote`.
- Whisper model can be any supported name (e.g., `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3`).

