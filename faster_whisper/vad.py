import bisect
import functools
import os

from dataclasses import dataclass
from typing import Union , Dict, List, Optional, Tuple
from enum import Enum


import numpy as np

from faster_whisper.utils import get_assets_path

class VadBackend(Enum):  
    SILERO = "silero"  
    PYANNOTE = "pyannote"
    WEIGHTED_COMBINATION = "weighted"

# The code below is adapted from https://github.com/snakers4/silero-vad.
@dataclass
class VadOptions:
    """VAD options.

    Attributes:
      threshold: Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH. It is better to tune this
        parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
      neg_threshold: Silence threshold for determining the end of speech. If a probability is lower
        than neg_threshold, it is always considered silence. Values higher than neg_threshold
        are only considered speech if the previous sample was classified as speech; otherwise,
        they are treated as silence. This parameter helps refine the detection of speech
         transitions, ensuring smoother segment boundaries.
      min_speech_duration_ms: Final speech chunks shorter min_speech_duration_ms are thrown out.
      max_speech_duration_s: Maximum duration of speech chunks in seconds. Chunks longer
        than max_speech_duration_s will be split at the timestamp of the last silence that
        lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise, they will be
        split aggressively just before max_speech_duration_s.
      min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms
        before separating it
      speech_pad_ms: Final speech chunks are padded by speech_pad_ms each side
    """

    threshold: float = 0.5
    neg_threshold: float = None
    min_speech_duration_ms: int = 0
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400
 
    backend: VadBackend = VadBackend.SILERO  
    pyannote_model: str = "pyannote/segmentation-3.0"  
    pyannote_auth_token: Optional[str] = None 
    
    min_duration_on: float = 0.0  
    min_duration_off: float = 0.0

    # Weighted ensemble parameters
    silero_weight: float = 0.5
    pyannote_weight: float = 0.5
    combination_threshold: float = 0.5
    combination_method: str = "confidence"  # "confidence" | "majority"
    min_overlap_duration_ms: int = 200
    max_gap_duration_ms: int = 1000


def get_speech_timestamps(
    audio: np.ndarray,
    vad_options: Optional[VadOptions] = None,
    sampling_rate: int = 16000,
    **kwargs,
) -> List[dict]:
    """This method is used for splitting long audios into speech chunks using VAD.

    Args:
      audio: One dimensional float array.
      vad_options: Options for VAD processing.
      sampling rate: Sampling rate of the audio.
      kwargs: VAD options passed as keyword arguments for backward compatibility.

    Returns:
      List of dicts containing begin and end samples of each speech chunk.
    """
    if vad_options is None:
        vad_options = VadOptions(**kwargs)

    # Use Pyannote VAD if specified
    if vad_options.backend == VadBackend.PYANNOTE:
        try:
            pyannote_model = get_pyannote_vad_model(
                vad_options.pyannote_model, 
                vad_options.pyannote_auth_token
            )
            return pyannote_model.get_speech_timestamps(audio, vad_options, sampling_rate)
        except Exception as e:
            print(f"Warning: Failed to load pyannote VAD model: {e}")
            print("Falling back to Silero-only VAD")
            # Fall back to Silero
            vad_options.backend = VadBackend.SILERO

    # Weighted ensemble of Silero + Pyannote
    if vad_options.backend == VadBackend.WEIGHTED_COMBINATION:
        return get_weighted_combination_timestamps(audio, vad_options, sampling_rate)

    # Silero VAD (default)
    threshold = vad_options.threshold
    neg_threshold = vad_options.neg_threshold
    min_speech_duration_ms = vad_options.min_speech_duration_ms
    max_speech_duration_s = vad_options.max_speech_duration_s
    min_silence_duration_ms = vad_options.min_silence_duration_ms
    window_size_samples = 512
    speech_pad_ms = vad_options.speech_pad_ms
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    model = get_vad_model()

    padded_audio = np.pad(
        audio, (0, window_size_samples - audio.shape[0] % window_size_samples)
    )
    speech_probs = model(padded_audio.reshape(1, -1)).squeeze(0)

    triggered = False
    speeches = []
    current_speech = {}
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)

    # to save potential segment end (and tolerate some silence)
    temp_end = 0
    # to save potential segment limits in case of maximum segment size reached
    prev_end = next_start = 0

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = window_size_samples * i
            continue

        if (
            triggered
            and (window_size_samples * i) - current_speech["start"] > max_speech_samples
        ):
            if prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                # previously reached silence (< neg_thres) and is still not speech (< thres)
                if next_start < prev_end:
                    triggered = False
                else:
                    current_speech["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech["end"] = window_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            # condition to avoid cutting in very short silence
            if (window_size_samples * i) - temp_end > min_silence_samples_at_max_speech:
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (
                    current_speech["end"] - current_speech["start"]
                ) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    return speeches


def collect_chunks(
    audio: np.ndarray, chunks: List[dict], sampling_rate: int = 16000
) -> Tuple[List[np.ndarray], List[Dict[str, int]]]:
    """Collects audio chunks."""
    if not chunks:
        chunk_metadata = {
            "start_time": 0,
            "end_time": 0,
        }
        return [np.array([], dtype=np.float32)], [chunk_metadata]

    audio_chunks = []
    chunks_metadata = []
    for chunk in chunks:
        chunk_metadata = {
            "start_time": chunk["start"] / sampling_rate,
            "end_time": chunk["end"] / sampling_rate,
        }
        audio_chunks.append(audio[chunk["start"] : chunk["end"]])
        chunks_metadata.append(chunk_metadata)
    return audio_chunks, chunks_metadata


class SpeechTimestampsMap:
    """Helper class to restore original speech timestamps."""

    def __init__(self, chunks: List[dict], sampling_rate: int, time_precision: int = 2):
        self.sampling_rate = sampling_rate
        self.time_precision = time_precision
        self.chunk_end_sample = []
        self.total_silence_before = []

        previous_end = 0
        silent_samples = 0

        for chunk in chunks:
            silent_samples += chunk["start"] - previous_end
            previous_end = chunk["end"]

            self.chunk_end_sample.append(chunk["end"] - silent_samples)
            self.total_silence_before.append(silent_samples / sampling_rate)

    def get_original_time(
        self,
        time: float,
        chunk_index: Optional[int] = None,
    ) -> float:
        if chunk_index is None:
            chunk_index = self.get_chunk_index(time)

        total_silence_before = self.total_silence_before[chunk_index]
        return round(total_silence_before + time, self.time_precision)

    def get_chunk_index(self, time: float) -> int:
        sample = int(time * self.sampling_rate)
        return min(
            bisect.bisect(self.chunk_end_sample, sample),
            len(self.chunk_end_sample) - 1,
        )


@functools.lru_cache
def get_vad_model():
    """Returns the VAD model instance."""
    encoder_path = os.path.join(get_assets_path(), "silero_encoder_v5.onnx")
    decoder_path = os.path.join(get_assets_path(), "silero_decoder_v5.onnx")
    return SileroVADModel(encoder_path, decoder_path)


class SileroVADModel:
    def __init__(self, encoder_path, decoder_path):
        try:
            import onnxruntime
        except ImportError as e:
            raise RuntimeError(
                "Applying the VAD filter requires the onnxruntime package"
            ) from e

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4

        self.encoder_session = onnxruntime.InferenceSession(
            encoder_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

    def __call__(
        self, audio: np.ndarray, num_samples: int = 512, context_size_samples: int = 64
    ):
        assert (
            audio.ndim == 2
        ), "Input should be a 2D array with size (batch_size, num_samples)"
        assert (
            audio.shape[1] % num_samples == 0
        ), "Input size should be a multiple of num_samples"

        batch_size = audio.shape[0]

        state = np.zeros((2, batch_size, 128), dtype="float32")
        context = np.zeros(
            (batch_size, context_size_samples),
            dtype="float32",
        )

        batched_audio = audio.reshape(batch_size, -1, num_samples)
        context = batched_audio[..., -context_size_samples:]
        context[:, -1] = 0
        context = np.roll(context, 1, 1)
        batched_audio = np.concatenate([context, batched_audio], 2)

        batched_audio = batched_audio.reshape(-1, num_samples + context_size_samples)

        encoder_batch_size = 10000
        num_segments = batched_audio.shape[0]
        encoder_outputs = []
        for i in range(0, num_segments, encoder_batch_size):
            encoder_output = self.encoder_session.run(
                None, {"input": batched_audio[i : i + encoder_batch_size]}
            )[0]
            encoder_outputs.append(encoder_output)

        encoder_output = np.concatenate(encoder_outputs, axis=0)
        encoder_output = encoder_output.reshape(batch_size, -1, 128)

        decoder_outputs = []
        for window in np.split(encoder_output, encoder_output.shape[1], axis=1):
            out, state = self.decoder_session.run(
                None, {"input": window.squeeze(1), "state": state}
            )
            decoder_outputs.append(out)

        out = np.stack(decoder_outputs, axis=1).squeeze(-1)
        return out


def merge_segments(segments_list, vad_options: VadOptions, sampling_rate: int = 16000):
    if not segments_list:
        return []

    curr_end = 0
    seg_idxs = []
    merged_segments = []
    edge_padding = vad_options.speech_pad_ms * sampling_rate // 1000
    chunk_length = vad_options.max_speech_duration_s * sampling_rate

    curr_start = segments_list[0]["start"]

    for idx, seg in enumerate(segments_list):
        # if any segment start timing is less than previous segment end timing,
        # reset the edge padding. Similarly for end timing.
        if idx > 0:
            if seg["start"] < segments_list[idx - 1]["end"]:
                seg["start"] += edge_padding
        if idx < len(segments_list) - 1:
            if seg["end"] > segments_list[idx + 1]["start"]:
                seg["end"] -= edge_padding

        if seg["end"] - curr_start > chunk_length and curr_end - curr_start > 0:
            merged_segments.append(
                {
                    "start": curr_start,
                    "end": curr_end,
                    "segments": seg_idxs,
                }
            )
            curr_start = seg["start"]
            seg_idxs = []
        curr_end = seg["end"]
        seg_idxs.append((seg["start"], seg["end"]))
    # add final
    merged_segments.append(
        {
            "start": curr_start,
            "end": curr_end,
            "segments": seg_idxs,
        }
    )
    return merged_segments


def _pad_and_merge_like_silero(
    speeches: List[dict],
    audio_length_samples: int,
    sampling_rate: int,
    speech_pad_ms: int,
) -> List[dict]:
    if not speeches:
        return []
    speeches = sorted(speeches, key=lambda s: s["start"])  # ensure order
    speech_pad_samples = int(sampling_rate * speech_pad_ms / 1000)
    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )
    return speeches


class PyannoteVADModel:  
    def __init__(self, model_name: str = "pyannote/segmentation-3.0", auth_token: Optional[str] = None):  
        try:  
            from pyannote.audio import Model  
            from pyannote.audio.pipelines import VoiceActivityDetection  
        except ImportError as e:  
            raise RuntimeError(  
                "Pyannote VAD requires: pip install pyannote.audio"  
            ) from e  
          
        self.model = Model.from_pretrained(model_name, use_auth_token=auth_token)  
        self.pipeline = VoiceActivityDetection(segmentation=self.model)  
      
    def get_speech_timestamps(  
        self,   
        audio: np.ndarray,   
        vad_options: VadOptions,  
        sampling_rate: int = 16000  
    ) -> List[dict]:  
        # Prefer in-memory inference to avoid disk I/O; fallback to temp file if needed
        hyper_parameters = {
            "min_duration_on": vad_options.min_duration_on,
            "min_duration_off": vad_options.min_duration_off,
        }
        self.pipeline.instantiate(hyper_parameters)

        try:
            import torch
            waveform = torch.from_numpy(audio.astype(np.float32))
            # Ensure shape (num_channels, num_samples)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            file_obj = {"waveform": waveform, "sample_rate": sampling_rate}
            vad_result = self.pipeline(file_obj)
        except Exception:
            # Fallback to temporary WAV path
            import tempfile
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio, sampling_rate)
                try:
                    vad_result = self.pipeline(tmp_file.name)
                finally:
                    import os as _os
                    _os.unlink(tmp_file.name)

        # Convert pyannote.core.Annotation to faster-whisper format
        speeches = []
        for segment in vad_result.itersegments():
            start_sample = int(segment.start * sampling_rate)
            end_sample = int(segment.end * sampling_rate)
            duration_ms = (end_sample - start_sample) * 1000 / sampling_rate
            if duration_ms >= vad_options.min_speech_duration_ms:
                speeches.append({"start": start_sample, "end": end_sample})

        # Apply padding/merging similar to Silero for qualitative parity
        speeches = _pad_and_merge_like_silero(
            speeches,
            audio_length_samples=len(audio),
            sampling_rate=sampling_rate,
            speech_pad_ms=vad_options.speech_pad_ms,
        )

        return speeches  


def get_weighted_combination_timestamps(
    audio: np.ndarray,
    vad_options: VadOptions,
    sampling_rate: int = 16000,
) -> List[dict]:
    # Silero segments
    silero_options = VadOptions(
        backend=VadBackend.SILERO,
        threshold=vad_options.threshold,
        neg_threshold=vad_options.neg_threshold,
        min_speech_duration_ms=vad_options.min_speech_duration_ms,
        max_speech_duration_s=vad_options.max_speech_duration_s,
        min_silence_duration_ms=vad_options.min_silence_duration_ms,
        speech_pad_ms=vad_options.speech_pad_ms,
    )
    silero_segments = get_speech_timestamps(audio, silero_options, sampling_rate)

    # Pyannote segments
    pyannote_options = VadOptions(
        backend=VadBackend.PYANNOTE,
        min_speech_duration_ms=vad_options.min_speech_duration_ms,
        speech_pad_ms=vad_options.speech_pad_ms,
        min_duration_on=vad_options.min_duration_on,
        min_duration_off=vad_options.min_duration_off,
        pyannote_model=vad_options.pyannote_model,
        pyannote_auth_token=vad_options.pyannote_auth_token,
    )
    try:
        pyannote_segments = get_speech_timestamps(audio, pyannote_options, sampling_rate)
    except Exception:
        # If Pyannote fails, return Silero
        return silero_segments

    # Convert to time ranges (seconds)
    def to_ranges(segs):
        return [(s["start"] / sampling_rate, s["end"] / sampling_rate) for s in segs]

    s_ranges = to_ranges(silero_segments)
    p_ranges = to_ranges(pyannote_segments)

    # Build boundary points
    points = sorted(set([t for r in s_ranges for t in r] + [t for r in p_ranges for t in r]))
    if not points:
        return []

    intervals = []
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        # skip zero-length
        if b <= a:
            continue
        # compute overlap ratio as confidence in [0,1]
        def overlap_conf(a,b,ranges):
            total=0.0
            for rs,re in ranges:
                ov_start = a if a>rs else rs
                ov_end = b if b<re else re
                if ov_end>ov_start:
                    total += ov_end-ov_start
            return total / (b-a)
        s_conf = overlap_conf(a,b,s_ranges)
        p_conf = overlap_conf(a,b,p_ranges)

        method = (vad_options.combination_method or "confidence").lower()
        keep = False
        if method in ("confidence","weighted"):
            score = (vad_options.silero_weight * s_conf +
                     vad_options.pyannote_weight * p_conf)
            keep = score >= vad_options.combination_threshold
        elif method == "majority":
            votes = int(s_conf > 0.0) + int(p_conf > 0.0)
            keep = votes >= 1
        else:
            # fallback to confidence
            score = (vad_options.silero_weight * s_conf +
                     vad_options.pyannote_weight * p_conf)
            keep = score >= vad_options.combination_threshold

        if keep:
            intervals.append((a, b))

    # Merge intervals considering max_gap and min_overlap
    if not intervals:
        return []
    merged = []
    current_start, current_end = intervals[0]
    max_gap = vad_options.max_gap_duration_ms / 1000.0
    min_keep = vad_options.min_overlap_duration_ms / 1000.0
    for a, b in intervals[1:]:
        if a - current_end <= max_gap:
            current_end = max(current_end, b)
        else:
            if current_end - current_start >= min_keep:
                merged.append({
                    "start": int(current_start * sampling_rate),
                    "end": int(current_end * sampling_rate),
                })
            current_start, current_end = a, b
    if current_end - current_start >= min_keep:
        merged.append({
            "start": int(current_start * sampling_rate),
            "end": int(current_end * sampling_rate),
        })

    # Apply Silero-like padding/merge for qualitative parity
    merged = _pad_and_merge_like_silero(
        merged,
        audio_length_samples=len(audio),
        sampling_rate=sampling_rate,
        speech_pad_ms=vad_options.speech_pad_ms,
    )
    return merged
  
@functools.lru_cache  
def get_pyannote_vad_model(model_name: str = "pyannote/segmentation-3.0", auth_token: Optional[str] = None):  
    """Returns the Pyannote VAD model instance."""  
    return PyannoteVADModel(model_name, auth_token)

def get_default_vad_options(backend: VadBackend = VadBackend.SILERO) -> VadOptions:
    """Get default VAD options for a specific backend.
    
    Args:
        backend: The VAD backend to use
        
    Returns:
        VadOptions with default parameters for the specified backend
    """
    if backend == VadBackend.SILERO:
        return VadOptions(
            backend=backend,
            threshold=0.5,
            min_speech_duration_ms=500,
            speech_pad_ms=400,
            min_silence_duration_ms=2000
        )
    elif backend == VadBackend.PYANNOTE:
        return VadOptions(
            backend=backend,
            min_speech_duration_ms=500,
            speech_pad_ms=400,
            min_duration_on=0.1,
            min_duration_off=0.1,
            pyannote_model="pyannote/segmentation-3.0"
        )
    elif backend == VadBackend.WEIGHTED_COMBINATION:
        return VadOptions(
            backend=backend,
            silero_weight=0.5,
            pyannote_weight=0.5,
            combination_threshold=0.5,
            combination_method="weighted",
            min_overlap_duration_ms=200,
            max_gap_duration_ms=1000,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


