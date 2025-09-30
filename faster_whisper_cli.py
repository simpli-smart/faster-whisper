#!/usr/bin/env python3
"""
Command Line Interface for Faster Whisper with VAD Backend Selection
"""

import argparse
import sys
import os
from typing import Optional

# Add the faster_whisper directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, VadBackend


def create_vad_options(backend: str, **kwargs) -> VadOptions:
    """Create VAD options based on backend selection."""
    backend_enum = VadBackend(backend.lower())
    
    if backend_enum == VadBackend.SILERO:
        return VadOptions(
            backend=backend_enum,
            threshold=kwargs.get('threshold', 0.5),
            min_speech_duration_ms=kwargs.get('min_speech_duration_ms', 500),
            speech_pad_ms=kwargs.get('speech_pad_ms', 400),
            min_silence_duration_ms=kwargs.get('min_silence_duration_ms', 2000)
        )
    elif backend_enum == VadBackend.PYANNOTE:
        return VadOptions(
            backend=backend_enum,
            min_speech_duration_ms=kwargs.get('min_speech_duration_ms', 500),
            speech_pad_ms=kwargs.get('speech_pad_ms', 400),
            min_duration_on=kwargs.get('min_duration_on', 0.1),
            min_duration_off=kwargs.get('min_duration_off', 0.1),
            pyannote_model=kwargs.get('pyannote_model', 'pyannote/segmentation-3.0'),
            pyannote_auth_token=kwargs.get('pyannote_auth_token')
        )
    elif backend_enum == VadBackend.WEIGHTED_COMBINATION:
        return VadOptions(
            backend=backend_enum,
            silero_weight=kwargs.get('silero_weight', 0.5),
            pyannote_weight=kwargs.get('pyannote_weight', 0.5),
            combination_threshold=kwargs.get('combination_threshold', 0.5),
            combination_method=kwargs.get('combination_method', 'confidence'),
            min_overlap_duration_ms=kwargs.get('min_overlap_duration_ms', 200),
            max_gap_duration_ms=kwargs.get('max_gap_duration_ms', 1000),
            # also forward necessary VAD params for sub-backends
            threshold=kwargs.get('threshold', 0.5),
            min_speech_duration_ms=kwargs.get('min_speech_duration_ms', 500),
            speech_pad_ms=kwargs.get('speech_pad_ms', 400),
            min_silence_duration_ms=kwargs.get('min_silence_duration_ms', 2000),
            min_duration_on=kwargs.get('min_duration_on', 0.1),
            min_duration_off=kwargs.get('min_duration_off', 0.1),
            pyannote_model=kwargs.get('pyannote_model', 'pyannote/segmentation-3.0'),
            pyannote_auth_token=kwargs.get('pyannote_auth_token'),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def list_vad_backends():
    """List all available VAD backends."""
    print("Available VAD Backends:")
    print("=" * 50)
    for backend in VadBackend:
        print(f"- {backend.value}")


def main():
    parser = argparse.ArgumentParser(
        description="Faster Whisper with VAD Backend Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default Silero VAD
  python faster_whisper_cli.py audio.mp3

  # Use Pyannote VAD
  python faster_whisper_cli.py audio.mp3 --vad-backend pyannote

  # Use Weighted VAD (confidence | majority)
  python faster_whisper_cli.py audio.mp3 --vad-backend weighted --combination-method confidence --silero-weight 0.6 --pyannote-weight 0.4 --combination-threshold 0.5

  # List available VAD backends
  python faster_whisper_cli.py --list-vad-backends

  # Custom VAD parameters
  python faster_whisper_cli.py audio.mp3 --vad-backend silero --vad-threshold 0.3
        """
    )
    
    # Main arguments
    parser.add_argument("audio_file", nargs="?", help="Audio file to transcribe")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--language", help="Language code (e.g., 'en', 'es', 'fr')")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe", 
                       help="Task to perform")
    parser.add_argument("--output", help="Output file (default: stdout)")
    
    # VAD arguments
    parser.add_argument("--vad-backend", choices=[b.value for b in VadBackend], default="pyannote",
                       help="VAD backend to use (default: pyannote)")
    parser.add_argument("--list-vad-backends", action="store_true",
                       help="List all available VAD backends and exit")
    
    # Silero VAD parameters
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                       help="Speech detection threshold for Silero VAD")
    parser.add_argument("--vad-min-speech-duration-ms", type=int, default=500,
                       help="Minimum speech duration in milliseconds")
    parser.add_argument("--vad-speech-pad-ms", type=int, default=400,
                       help="Speech padding in milliseconds")
    parser.add_argument("--vad-min-silence-duration-ms", type=int, default=2000,
                       help="Minimum silence duration in milliseconds")
    
    # Pyannote VAD parameters
    parser.add_argument("--pyannote-model", default="pyannote/segmentation-3.0",
                       help="Pyannote model to use")
    parser.add_argument("--pyannote-auth-token", default=os.environ.get("PYANNOTE_AUTH_TOKEN"),
                       help="HuggingFace auth token for Pyannote")
    parser.add_argument("--vad-min-duration-on", type=float, default=0.1,
                       help="Minimum duration for speech segments (Pyannote)")
    parser.add_argument("--vad-min-duration-off", type=float, default=0.1,
                       help="Minimum duration for silence segments (Pyannote)")
    
    # Weighted Combination VAD parameters
    parser.add_argument("--combination-method", choices=["confidence", "majority"], default="confidence",
                       help="Combination strategy for weighted VAD")
    parser.add_argument("--silero-weight", type=float, default=0.5,
                       help="Weight for Silero in weighted combination")
    parser.add_argument("--pyannote-weight", type=float, default=0.5,
                       help="Weight for Pyannote in weighted combination")
    parser.add_argument("--combination-threshold", type=float, default=0.5,
                       help="Threshold for weighted combination score")
    parser.add_argument("--min-overlap-duration-ms", type=int, default=200,
                       help="Minimum duration to keep after combining intervals (ms)")
    parser.add_argument("--max-gap-duration-ms", type=int, default=1000,
                       help="Maximum gap to merge adjacent intervals (ms)")
    
    
    # Whisper parameters
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--best-of", type=int, default=5, help="Number of best candidates")
    parser.add_argument("--temperature", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                       help="Temperature for sampling")
    parser.add_argument("--word-timestamps", action="store_true", help="Include word timestamps")
    parser.add_argument("--device", default="auto", help="Device to use (cpu, cuda, auto)")
    parser.add_argument("--compute-type", default="default", help="Compute type for CTranslate2")
    
    args = parser.parse_args()
    
    # List VAD backends if requested
    if args.list_vad_backends:
        list_vad_backends()
        return
    
    # Check if audio file is provided
    if not args.audio_file:
        parser.error("Audio file is required (unless --list-vad-backends is used)")
    
    if not os.path.exists(args.audio_file):
        parser.error(f"Audio file not found: {args.audio_file}")
    
    # Create VAD options
    vad_kwargs = {
        'threshold': args.vad_threshold,
        'min_speech_duration_ms': args.vad_min_speech_duration_ms,
        'speech_pad_ms': args.vad_speech_pad_ms,
        'min_silence_duration_ms': args.vad_min_silence_duration_ms,
        'pyannote_model': args.pyannote_model,
        'pyannote_auth_token': args.pyannote_auth_token,
        'min_duration_on': args.vad_min_duration_on,
        'min_duration_off': args.vad_min_duration_off,
        'combination_method': args.combination_method,
        'silero_weight': args.silero_weight,
        'pyannote_weight': args.pyannote_weight,
        'combination_threshold': args.combination_threshold,
        'min_overlap_duration_ms': args.min_overlap_duration_ms,
        'max_gap_duration_ms': args.max_gap_duration_ms,
    }
    
    try:
        vad_options = create_vad_options(args.vad_backend, **vad_kwargs)
    except Exception as e:
        print(f"Error creating VAD options: {e}")
        return 1
    
    # Load Whisper model
    print(f"Loading Whisper model: {args.model}")
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type
    )
    
    # Transcribe audio
    print(f"Transcribing {args.audio_file} with {args.vad_backend} VAD...")
    
    segments, info = model.transcribe(
        args.audio_file,
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        best_of=args.best_of,
        temperature=args.temperature,
        word_timestamps=args.word_timestamps,
        vad_filter=True,
        vad_parameters=vad_options
    )
    
    # Output results
    output_text = ""
    for segment in segments:
        output_text += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Transcription saved to: {args.output}")
    else:
        print("\nTranscription:")
        print("=" * 50)
        print(output_text)
    
    # Print info
    print(f"\nTranscription Info:")
    print(f"Language: {info.language} (probability: {info.language_probability:.3f})")
    print(f"Duration: {info.duration:.2f}s")
    print(f"Duration after VAD: {info.duration_after_vad:.2f}s")
    print(f"VAD Backend: {info.vad_options.backend.value}")


if __name__ == "__main__":
    sys.exit(main())
