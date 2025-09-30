#!/usr/bin/env python3
"""
Comprehensive Diarization System
Uses the best available diarization models for speaker separation
"""

import numpy as np
import soundfile as sf
import tempfile
import os
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

# Try to import diarization libraries
try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: Pyannote not available. Install with: pip install pyannote.audio")

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

try:
    from speechbrain.pretrained import SpeakerRecognition
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    print("Warning: SpeechBrain not available. Install with: pip install speechbrain")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: Librosa not available")


class DiarizationBackend(Enum):
    """Available diarization backends."""
    PYANNOTE = "pyannote"


@dataclass
class SpeakerSegment:
    """Represents a speaker segment in the audio."""
    start: float
    end: float
    speaker: str
    confidence: float = 1.0
    
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start


@dataclass
class DiarizationResult:
    """Result of diarization process."""
    segments: List[SpeakerSegment]
    num_speakers: int
    total_duration: float
    processing_time: float
    backend: str
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "segments": [asdict(seg) for seg in self.segments],
            "num_speakers": self.num_speakers,
            "total_duration": self.total_duration,
            "processing_time": self.processing_time,
            "backend": self.backend,
            "metadata": self.metadata or {}
        }
    
    def get_speaker_stats(self) -> Dict:
        """Get statistics for each speaker."""
        speaker_stats = {}
        for segment in self.segments:
            if segment.speaker not in speaker_stats:
                speaker_stats[segment.speaker] = {
                    "total_duration": 0.0,
                    "num_segments": 0,
                    "avg_confidence": 0.0
                }
            
            speaker_stats[segment.speaker]["total_duration"] += segment.duration()
            speaker_stats[segment.speaker]["num_segments"] += 1
            speaker_stats[segment.speaker]["avg_confidence"] += segment.confidence
        
        # Calculate averages
        for speaker in speaker_stats:
            num_segments = speaker_stats[speaker]["num_segments"]
            if num_segments > 0:
                speaker_stats[speaker]["avg_confidence"] /= num_segments
        
        return speaker_stats


@dataclass
class DiarizationOptions:
    """Options for diarization."""
    backend: DiarizationBackend = DiarizationBackend.PYANNOTE
    
    # Pyannote options
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    pyannote_auth_token: Optional[str] = None
    pyannote_min_speakers: Optional[int] = None
    pyannote_max_speakers: Optional[int] = None
    
    # SpeechBrain options
    speechbrain_model: str = "speechbrain/speaker-diarization"
    
    # General options
    min_segment_duration: float = 0.5  # Minimum segment duration in seconds
    min_speaker_duration: float = 1.0  # Minimum speaker duration in seconds


class PyannoteDiarizer:
    """Pyannote-based diarization."""
    
    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1", auth_token: Optional[str] = None):
        """Initialize Pyannote diarizer."""
        if not PYANNOTE_AVAILABLE:
            raise ImportError("Pyannote not available. Install with: pip install pyannote.audio")
        
        self.model_name = model_name
        self.auth_token = auth_token
        self.pipeline = None
        
    def _load_pipeline(self):
        """Load the Pyannote pipeline."""
        if self.pipeline is None:
            if self.auth_token:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    use_auth_token=self.auth_token
                )
            else:
                self.pipeline = Pipeline.from_pretrained(self.model_name)
    
    def diarize(self, audio_file: str, options: DiarizationOptions) -> DiarizationResult:
        """Perform diarization using Pyannote."""
        start_time = time.time()
        
        # Allow overriding model name from options
        if getattr(options, "pyannote_model", None) and options.pyannote_model != self.model_name:
            self.model_name = options.pyannote_model
            self.pipeline = None

        self._load_pipeline()
        
        # Prepare parameters
        params = {}
        if options.pyannote_min_speakers is not None:
            params["min_speakers"] = options.pyannote_min_speakers
        if options.pyannote_max_speakers is not None:
            params["max_speakers"] = options.pyannote_max_speakers
        
        # Run diarization
        with ProgressHook() as hook:
            diarization = self.pipeline(audio_file, hook=hook, **params)
        
        # Extract segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.end - turn.start >= options.min_segment_duration:
                segments.append(SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                    confidence=1.0  # Pyannote doesn't provide confidence scores
                ))
        
        # Get unique speakers
        speakers = list(set(seg.speaker for seg in segments))
        num_speakers = len(speakers)
        
        # Get audio duration
        audio_info = sf.info(audio_file)
        total_duration = audio_info.duration
        
        processing_time = time.time() - start_time
        
        return DiarizationResult(
            segments=segments,
            num_speakers=num_speakers,
            total_duration=total_duration,
            processing_time=processing_time,
            backend="pyannote",
            metadata={
                "model": self.model_name,
                "parameters": params
            }
        )


class SimpleVADDiarizer:
    """Simple VAD-based diarization (single speaker)."""
    
    def diarize(self, audio_file: str, options: DiarizationOptions) -> DiarizationResult:
        """Perform simple VAD-based diarization."""
        start_time = time.time()
        
        # Load audio
        audio, sr = sf.read(audio_file)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        # Ensure audio is float32
        audio = audio.astype(np.float32)
        
        # Simple energy-based VAD
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Calculate energy
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))
        
        energy = np.array(energy)
        
        # Simple thresholding
        threshold = np.mean(energy) + 0.1 * np.std(energy)
        speech_frames = energy > threshold
        
        # Find speech segments
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                end_frame = i
                duration = (end_frame - start_frame) * hop_length / sr
                
                if duration >= options.min_segment_duration:
                    segments.append(SpeakerSegment(
                        start=start_frame * hop_length / sr,
                        end=end_frame * hop_length / sr,
                        speaker="SPEAKER_1",
                        confidence=0.7
                    ))
                
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech:
            duration = (len(speech_frames) - start_frame) * hop_length / sr
            if duration >= options.min_segment_duration:
                segments.append(SpeakerSegment(
                    start=start_frame * hop_length / sr,
                    end=len(speech_frames) * hop_length / sr,
                    speaker="SPEAKER_1",
                    confidence=0.7
                ))
        
        processing_time = time.time() - start_time
        
        return DiarizationResult(
            segments=segments,
            num_speakers=1,
            total_duration=len(audio) / sr,
            processing_time=processing_time,
            backend="simple_vad",
            metadata={
                "method": "energy-based VAD",
                "frame_length_ms": 25,
                "hop_length_ms": 10
            }
        )


class DiarizationSystem:
    """Main diarization system that can use different backends."""
    
    def __init__(self):
        """Initialize the diarization system."""
        self.diarizers = {}
        
        # Initialize available diarizers
        if PYANNOTE_AVAILABLE:
            self.diarizers[DiarizationBackend.PYANNOTE] = PyannoteDiarizer()
    
    def list_available_backends(self) -> List[str]:
        """List available diarization backends."""
        return [backend.value for backend in self.diarizers.keys()]
    
    def diarize(self, audio_file: str, options: DiarizationOptions) -> DiarizationResult:
        """Perform diarization using the specified backend."""
        if options.backend not in self.diarizers:
            raise ValueError(f"Backend {options.backend.value} not available. Available: {self.list_available_backends()}")
        
        diarizer = self.diarizers[options.backend]
        return diarizer.diarize(audio_file, options)
    
    def compare_backends(self, audio_file: str, options_list: List[DiarizationOptions] = None) -> Dict:
        """Compare different diarization backends."""
        if options_list is None:
            options_list = [
                DiarizationOptions(backend=DiarizationBackend.SIMPLE_VAD),
            ]
            
            if PYANNOTE_AVAILABLE:
                options_list.append(DiarizationOptions(backend=DiarizationBackend.PYANNOTE))
            
            if SPEECHBRAIN_AVAILABLE:
                options_list.append(DiarizationOptions(backend=DiarizationBackend.SPEECHBRAIN))
        
        results = {}
        
        for options in options_list:
            try:
                print(f"Running {options.backend.value} diarization...")
                result = self.diarize(audio_file, options)
                results[options.backend.value] = result
                print(f"✓ {options.backend.value}: {result.num_speakers} speakers, {len(result.segments)} segments")
            except Exception as e:
                print(f"✗ {options.backend.value} failed: {e}")
                results[options.backend.value] = {"error": str(e)}
        
        return results


def calculate_diarization_metrics(result: DiarizationResult) -> Dict:
    """Calculate diarization metrics."""
    if not result.segments:
        return {
            "total_speech_duration": 0.0,
            "speech_ratio": 0.0,
            "avg_segment_duration": 0.0,
            "min_segment_duration": 0.0,
            "max_segment_duration": 0.0,
            "speaker_durations": {},
            "speaker_segment_counts": {},
            "processing_speed": result.total_duration / result.processing_time if result.processing_time > 0 else 0.0
        }
    
    # Basic metrics
    total_speech_duration = sum(seg.duration() for seg in result.segments)
    speech_ratio = total_speech_duration / result.total_duration
    
    segment_durations = [seg.duration() for seg in result.segments]
    avg_segment_duration = np.mean(segment_durations)
    min_segment_duration = np.min(segment_durations)
    max_segment_duration = np.max(segment_durations)
    
    # Speaker statistics
    speaker_stats = result.get_speaker_stats()
    speaker_durations = {speaker: stats["total_duration"] for speaker, stats in speaker_stats.items()}
    speaker_segment_counts = {speaker: stats["num_segments"] for speaker, stats in speaker_stats.items()}
    
    return {
        "total_speech_duration": total_speech_duration,
        "speech_ratio": speech_ratio,
        "avg_segment_duration": avg_segment_duration,
        "min_segment_duration": min_segment_duration,
        "max_segment_duration": max_segment_duration,
        "speaker_durations": speaker_durations,
        "speaker_segment_counts": speaker_segment_counts,
        "processing_speed": result.total_duration / result.processing_time if result.processing_time > 0 else 0.0  # Real-time factor
    }


def main():
    """Main function for testing diarization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Diarization System")
    parser.add_argument("audio_file", help="Audio file to diarize")
    parser.add_argument("--backend", choices=["pyannote"], 
                       default="pyannote", help="Diarization backend")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--compare", action="store_true", help="Compare all available backends")
    parser.add_argument("--min-segment-duration", type=float, default=0.5, 
                       help="Minimum segment duration in seconds")
    parser.add_argument("--pyannote-auth-token", help="Pyannote auth token")
    parser.add_argument("--pyannote-model", default="pyannote/speaker-diarization-3.1", 
                       help="Pyannote diarization pipeline model (e.g., 'pyannote/speaker-diarization')")
    
    args = parser.parse_args()
    
    # Initialize system
    system = DiarizationSystem()
    
    print(f"Available backends: {system.list_available_backends()}")
    
    if args.compare:
        print("\nComparing all backends...")
        results = system.compare_backends(args.audio_file)
        
        print("\nResults Summary:")
        print("=" * 60)
        for backend, result in results.items():
            if isinstance(result, dict) and "error" in result:
                print(f"{backend}: ERROR - {result['error']}")
            else:
                metrics = calculate_diarization_metrics(result)
                print(f"{backend}:")
                print(f"  Speakers: {result.num_speakers}")
                print(f"  Segments: {len(result.segments)}")
                print(f"  Speech ratio: {metrics['speech_ratio']:.2%}")
                print(f"  Processing time: {result.processing_time:.2f}s")
                print(f"  Real-time factor: {metrics['processing_speed']:.2f}x")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({backend: result.to_dict() if hasattr(result, 'to_dict') else result 
                          for backend, result in results.items()}, f, indent=2)
    
    else:
        # Single backend
        options = DiarizationOptions(
            backend=DiarizationBackend(args.backend),
            min_segment_duration=args.min_segment_duration,
            pyannote_auth_token=args.pyannote_auth_token,
            pyannote_model=args.pyannote_model,
        )
        
        print(f"\nRunning {args.backend} diarization...")
        result = system.diarize(args.audio_file, options)
        
        # Calculate metrics
        metrics = calculate_diarization_metrics(result)
        
        print(f"\nDiarization Results:")
        print("=" * 40)
        print(f"Backend: {result.backend}")
        print(f"Speakers detected: {result.num_speakers}")
        print(f"Total segments: {len(result.segments)}")
        print(f"Total duration: {result.total_duration:.2f}s")
        print(f"Speech duration: {metrics['total_speech_duration']:.2f}s")
        print(f"Speech ratio: {metrics['speech_ratio']:.2%}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Real-time factor: {metrics['processing_speed']:.2f}x")
        
        if result.segments:
            print(f"\nSegment Details:")
            for i, segment in enumerate(result.segments[:10]):  # Show first 10
                print(f"  {i+1:2d}. {segment.start:6.2f}s - {segment.end:6.2f}s | {segment.speaker} | {segment.confidence:.2f}")
            if len(result.segments) > 10:
                print(f"  ... and {len(result.segments) - 10} more segments")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
