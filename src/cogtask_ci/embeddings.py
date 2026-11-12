"""Frozen multilingual text embeddings with reproducible pooling and windowing.

The optional ``torch`` and ``transformers`` dependencies are imported only when
an embedder is constructed. The implementation intentionally exposes no labels
or prediction interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import numpy as np


_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "intfloat/multilingual-e5-small": {
        "prefix": "query: ",
        "pooling": "masked_mean",
        "weight_patterns": ("model.safetensors",),
    },
    "BAAI/bge-m3": {
        "prefix": "",
        "pooling": "cls",
        "weight_patterns": ("pytorch_model.bin",),
    },
}
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PREPROCESSING_VERSION = "unicode-nfc-strip-segments-v1"
_CUBLAS_WORKSPACE_CONFIG = "CUBLAS_WORKSPACE_CONFIG"
_VALID_CUBLAS_WORKSPACE_CONFIGS = {":4096:8", ":16:8"}
_COMMON_SNAPSHOT_ALLOW_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "merges.txt",
]
_TOKENIZER_FILENAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
}


@dataclass(frozen=True)
class _Window:
    input_ids: tuple[int, ...]
    content_tokens: int


def _normalise_segments(segments: Sequence[str]) -> tuple[str, ...]:
    if isinstance(segments, (str, bytes)):
        raise TypeError("A transcript must be a sequence of text segments, not one string.")
    cleaned: list[str] = []
    for segment in segments:
        if not isinstance(segment, str):
            raise TypeError("Every transcript segment must be a string.")
        value = unicodedata.normalize("NFC", segment).strip()
        if value:
            cleaned.append(value)
    if not cleaned:
        raise ValueError("Transcript text is empty after preprocessing.")
    return tuple(cleaned)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("Tokenizer returned more than one sequence for a text segment.")
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def _special_token_count(tokenizer: Any) -> int:
    method = getattr(tokenizer, "num_special_tokens_to_add", None)
    if method is not None:
        return int(method(pair=False))
    return len(tokenizer.build_inputs_with_special_tokens([]))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_file_fingerprints(snapshot_path: Path) -> dict[str, dict[str, int | str]]:
    return {
        path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted(snapshot_path.iterdir(), key=lambda p: p.name)
        if path.is_file()
    }


def _weight_fingerprints(
    snapshot_path: Path,
    patterns: Sequence[str],
    artifact_hashes: Mapping[str, Mapping[str, int | str]],
) -> tuple[dict[str, str], str]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(snapshot_path.glob(pattern))
    unique = sorted(set(path for path in files if path.is_file()), key=lambda p: p.name)
    if not unique:
        raise RuntimeError(f"No supported model weight files found in {snapshot_path}.")
    fingerprints = {
        path.name: str(artifact_hashes[path.name]["sha256"])
        for path in unique
    }
    if len(fingerprints) == 1:
        # Preserve the Hub's exact file SHA-256 for the common single-file case.
        weight_sha256 = next(iter(fingerprints.values()))
    else:
        combined = hashlib.sha256()
        for name, digest in sorted(fingerprints.items()):
            combined.update(name.encode("utf-8"))
            combined.update(b"\0")
            combined.update(digest.encode("ascii"))
            combined.update(b"\n")
        weight_sha256 = combined.hexdigest()
    return fingerprints, weight_sha256


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "torch", "transformers", "tokenizers", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    cls = type(tokenizer)
    identity: dict[str, Any] = {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "is_fast": getattr(tokenizer, "is_fast", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    special_map = getattr(tokenizer, "special_tokens_map", None)
    if isinstance(special_map, Mapping):
        identity["special_tokens_map"] = {
            str(key): str(value) for key, value in sorted(special_map.items())
        }
    return identity


def _configure_torch(seed: int, threads: int) -> Any:
    """Enable strict deterministic FP32 inference or fail with an actionable error."""
    configured = os.environ.get(_CUBLAS_WORKSPACE_CONFIG)
    torch_loaded = sys.modules.get("torch")
    cuda_initialized = False
    if torch_loaded is not None:
        cuda = getattr(torch_loaded, "cuda", None)
        is_initialized = getattr(cuda, "is_initialized", None)
        if is_initialized is not None:
            cuda_initialized = bool(is_initialized())
    if configured not in _VALID_CUBLAS_WORKSPACE_CONFIGS:
        if cuda_initialized:
            raise RuntimeError(
                "CUDA was initialized before deterministic setup and "
                "CUBLAS_WORKSPACE_CONFIG was not a supported value. Start a fresh "
                "process with CUBLAS_WORKSPACE_CONFIG=:4096:8."
            )
        os.environ[_CUBLAS_WORKSPACE_CONFIG] = ":4096:8"

    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "Frozen embeddings require the optional dependencies: install "
            "`cogtask-ci[embeddings]`."
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError as exc:
        # PyTorch only permits setting inter-op threads before parallel work.
        # Accept an already matching setting; never swallow a real mismatch.
        if torch.get_num_interop_threads() != threads:
            raise RuntimeError(
                "Could not set the requested PyTorch inter-op thread count. "
                "Create the embedder before running parallel PyTorch work, or use "
                "the already-configured thread count."
            ) from exc

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    matmul_backend = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul_backend is not None and hasattr(matmul_backend, "fp32_precision"):
        # PyTorch 2.9+ recommends the precision API and deprecates allow_tf32.
        if hasattr(torch.backends, "fp32_precision"):
            torch.backends.fp32_precision = "ieee"
        matmul_backend.fp32_precision = "ieee"
        cudnn_backend = torch.backends.cudnn
        if hasattr(cudnn_backend, "fp32_precision"):
            cudnn_backend.fp32_precision = "ieee"
        for operator_backend in (
            getattr(cudnn_backend, "conv", None),
            getattr(cudnn_backend, "rnn", None),
        ):
            if operator_backend is not None and hasattr(operator_backend, "fp32_precision"):
                operator_backend.fp32_precision = "ieee"
    else:
        if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
            matmul_backend.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        set_precision = getattr(torch, "set_float32_matmul_precision", None)
        if set_precision is not None:
            set_precision("highest")
    return torch


class FrozenTextEmbedder:
    """Encode transcripts with a supported model pinned to an exact Hub revision.

    ``encode`` accepts a batch of transcripts, where each transcript is a
    sequence of text segments. Segments are packed at their boundaries into
    model windows; oversized segments are split without overlap. The returned
    vectors are content-token-weighted window means followed by L2 normalization.

    Pass both a tokenizer and model only for controlled injection (for example,
    unit tests). Production loading downloads the selected revision into the
    Hugging Face cache, fingerprints the weight file(s), and loads FP32 weights.
    """

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
        weight_sha256: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        device: str = "auto",
        max_tokens: int = 512,
        batch_size: int = 8,
        seed: int = 0,
        threads: int = 1,
    ) -> None:
        if model_id not in _MODEL_PROFILES:
            raise ValueError(f"Unsupported frozen embedding model: {model_id!r}.")
        if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
            raise ValueError("revision must be a frozen 40-character lowercase Hub commit SHA.")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 512:
            raise ValueError("max_tokens must be an integer between 1 and 512.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer.")
        if (tokenizer is None) != (model is None):
            raise ValueError("Provide both tokenizer and model for injection, or neither.")
        if weight_sha256 is not None and not _SHA256_RE.fullmatch(weight_sha256):
            raise ValueError("weight_sha256 must be a 64-character lowercase SHA-256 digest.")

        self.model_id = model_id
        self.revision = revision
        self.profile = dict(_MODEL_PROFILES[model_id])
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.seed = seed
        self.threads = threads
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.local_files_only = bool(local_files_only)
        self._torch = _configure_torch(seed, threads)

        self.weight_file_hashes: dict[str, str] = {}
        self.artifact_file_hashes: dict[str, dict[str, int | str]] = {}
        self.weight_sha256 = weight_sha256
        if tokenizer is None:
            tokenizer, model, loaded_hashes, loaded_sha256, artifact_hashes = self._load_frozen_model()
            self.weight_file_hashes = loaded_hashes
            self.artifact_file_hashes = artifact_hashes
            if self.weight_sha256 is not None and self.weight_sha256 != loaded_sha256:
                raise ValueError("Provided weight_sha256 does not match the frozen checkpoint files.")
            self.weight_sha256 = loaded_sha256
        self.tokenizer = tokenizer
        self.model = model
        self.tokenizer_file_hashes = {
            name: str(record["sha256"])
            for name, record in self.artifact_file_hashes.items()
            if name in _TOKENIZER_FILENAMES
        }
        self.model_config_sha256 = (
            str(self.artifact_file_hashes["config.json"]["sha256"])
            if "config.json" in self.artifact_file_hashes
            else None
        )
        self.tokenizer_identity = _tokenizer_identity(self.tokenizer)
        self.package_versions = _package_versions()

        self.device = self._resolve_device(device)
        self.model.to(self.device, dtype=self._torch.float32)
        self.model.eval()
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            raise ValueError("Tokenizer must define pad_token_id for deterministic batching.")
        self._prefix_ids = _token_ids(self.tokenizer, self.profile["prefix"])
        self._separator_ids = _token_ids(self.tokenizer, "\n")
        self._special_tokens = _special_token_count(self.tokenizer)
        self._payload_capacity = self.max_tokens - self._special_tokens - len(self._prefix_ids)
        if self._payload_capacity < 1:
            raise ValueError("max_tokens leaves no room for content after prefix and special tokens.")
        self.dimension = self._model_dimension()

    def _load_frozen_model(
        self,
    ) -> tuple[Any, Any, dict[str, str], str, dict[str, dict[str, int | str]]]:
        try:
            from huggingface_hub import snapshot_download
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Frozen embeddings require `transformers` and `huggingface_hub`; "
                "install `cogtask-ci[embeddings]`."
            ) from exc

        weight_patterns = tuple(self.profile["weight_patterns"])
        snapshot_path = Path(
            snapshot_download(
                repo_id=self.model_id,
                revision=self.revision,
                cache_dir=self.cache_dir,
                allow_patterns=[*_COMMON_SNAPSHOT_ALLOW_PATTERNS, *weight_patterns],
                local_files_only=self.local_files_only,
            )
        )
        artifact_hashes = _snapshot_file_fingerprints(snapshot_path)
        hashes, checkpoint_sha256 = _weight_fingerprints(
            snapshot_path, weight_patterns, artifact_hashes
        )
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot_path), use_fast=True, local_files_only=True)
        transformers_version = importlib.metadata.version("transformers")
        version_match = re.match(r"^(\d+)\.(\d+)", transformers_version)
        dtype_argument = "torch_dtype"
        if version_match and tuple(map(int, version_match.groups())) >= (4, 57):
            dtype_argument = "dtype"
        model = AutoModel.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
            **{dtype_argument: self._torch.float32},
        )
        return tokenizer, model, hashes, checkpoint_sha256, artifact_hashes

    def _resolve_device(self, device: str) -> Any:
        torch = self._torch
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return resolved

    def _model_dimension(self) -> int:
        config = getattr(self.model, "config", None)
        for name in ("hidden_size", "d_model", "dim"):
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                return value
        raise ValueError("Could not determine the model hidden dimension from model.config.")

    def _window_sequence(self, payload_ids: Sequence[int]) -> tuple[int, ...]:
        prepared = self.tokenizer.build_inputs_with_special_tokens(
            [*self._prefix_ids, *payload_ids]
        )
        sequence = tuple(int(token_id) for token_id in prepared)
        if len(sequence) > self.max_tokens:
            raise RuntimeError("Tokenizer special-token construction exceeded max_tokens.")
        return sequence

    def _prepare_windows(self, segments: Sequence[str]) -> list[_Window]:
        cleaned = _normalise_segments(segments)
        windows: list[_Window] = []
        current_ids: list[int] = []
        current_content = 0

        def emit_current() -> None:
            nonlocal current_ids, current_content
            if current_content:
                windows.append(
                    _Window(
                        input_ids=self._window_sequence(current_ids),
                        content_tokens=current_content,
                    )
                )
            current_ids = []
            current_content = 0

        for segment_index, segment in enumerate(cleaned):
            segment_ids = _token_ids(self.tokenizer, segment)
            if not segment_ids:
                continue
            delimiter = self._separator_ids if segment_index > 0 else []

            if len(segment_ids) > self._payload_capacity:
                # Keep an oversized segment contiguous in source order. Each
                # token appears in exactly one non-overlapping window.
                emit_current()
                for start in range(0, len(segment_ids), self._payload_capacity):
                    chunk = segment_ids[start : start + self._payload_capacity]
                    windows.append(
                        _Window(
                            input_ids=self._window_sequence(chunk),
                            content_tokens=len(chunk),
                        )
                    )
                continue

            required = len(segment_ids) + (len(delimiter) if current_ids else 0)
            if current_ids and len(current_ids) + required > self._payload_capacity:
                emit_current()
            if current_ids and delimiter:
                current_ids.extend(delimiter)
            current_ids.extend(segment_ids)
            current_content += len(segment_ids)

        emit_current()
        if not windows:
            raise ValueError("Transcript has no model content tokens after tokenization.")
        return windows

    def encode(self, transcripts: Sequence[Sequence[str]]) -> np.ndarray:
        """Encode a batch of segmented transcripts to finite unit-length vectors."""
        if isinstance(transcripts, (str, bytes)):
            raise TypeError("encode expects a batch of transcripts, not one string.")
        transcripts = list(transcripts)
        if not transcripts:
            return np.empty((0, self.dimension), dtype=np.float32)

        all_windows: list[tuple[int, _Window]] = []
        for transcript_index, segments in enumerate(transcripts):
            for window in self._prepare_windows(segments):
                all_windows.append((transcript_index, window))

        sums = np.zeros((len(transcripts), self.dimension), dtype=np.float64)
        weights = np.zeros(len(transcripts), dtype=np.int64)
        torch = self._torch
        pad_token_id = int(self.tokenizer.pad_token_id)

        for start in range(0, len(all_windows), self.batch_size):
            batch_items = all_windows[start : start + self.batch_size]
            max_length = max(len(window.input_ids) for _, window in batch_items)
            input_ids = torch.full(
                (len(batch_items), max_length),
                pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros(
                (len(batch_items), max_length), dtype=torch.long, device=self.device
            )
            for row, (_, window) in enumerate(batch_items):
                length = len(window.input_ids)
                input_ids[row, :length] = torch.as_tensor(
                    window.input_ids, dtype=torch.long, device=self.device
                )
                attention_mask[row, :length] = 1

            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
                if self.profile["pooling"] == "masked_mean":
                    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                else:
                    pooled = hidden[:, 0, :]
                pooled = pooled.to(dtype=torch.float32)
            vectors = pooled.detach().cpu().numpy()
            if vectors.shape != (len(batch_items), self.dimension):
                raise ValueError("Model returned embeddings with an unexpected shape.")
            if not np.isfinite(vectors).all():
                raise ValueError("Model produced non-finite embeddings.")
            for row, (transcript_index, window) in enumerate(batch_items):
                sums[transcript_index] += vectors[row].astype(np.float64) * window.content_tokens
                weights[transcript_index] += window.content_tokens

        if np.any(weights <= 0):
            raise ValueError("At least one transcript has no content tokens to pool.")
        means = sums / weights[:, None]
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        if not np.isfinite(means).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError("Cannot normalize an empty, zero, or non-finite embedding.")
        embeddings = (means / norms).astype(np.float32)
        if not np.isfinite(embeddings).all():
            raise ValueError("Final normalized embeddings contain non-finite values.")
        return embeddings

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of unsegmented texts, treating each text as one segment."""
        if isinstance(texts, (str, bytes)):
            raise TypeError("encode_texts expects a sequence of text strings.")
        return self.encode([[text] for text in texts])

    def cache_key(self, segments: Sequence[str]) -> str:
        """Return a label-free key containing text, preprocessing, weights, and config."""
        if self.weight_sha256 is None:
            raise ValueError("A checkpoint weight_sha256 is required to create a cache key.")
        if not _SHA256_RE.fullmatch(self.weight_sha256):
            raise ValueError("The loaded weight_sha256 is not a valid SHA-256 digest.")
        cleaned = _normalise_segments(segments)
        payload = {
            "text_segments": list(cleaned),
            "preprocessing": {
                "version": _PREPROCESSING_VERSION,
                "segment_separator": "\\n",
            },
            "model": {
                "id": self.model_id,
                "revision": self.revision,
                "weight_sha256": self.weight_sha256,
                "weight_file_hashes": self.weight_file_hashes,
                "artifact_file_hashes": self.artifact_file_hashes,
                "tokenizer_file_hashes": self.tokenizer_file_hashes,
                "model_config_sha256": self.model_config_sha256,
                "tokenizer_identity": self.tokenizer_identity,
            },
            "configuration": {
                "pooling": self.profile["pooling"],
                "prefix": self.profile["prefix"],
                "max_tokens": self.max_tokens,
                "batch_size": self.batch_size,
                "device": str(self.device),
                "seed": self.seed,
                "threads": self.threads,
                "precision": "float32",
                "deterministic_algorithms": True,
                "local_files_only": self.local_files_only,
                "package_versions": self.package_versions,
            },
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def runtime_metadata(self) -> dict[str, Any]:
        """Return versions, hardware, checkpoint, and determinism settings for a run record."""
        torch = self._torch
        cuda_available = bool(torch.cuda.is_available())
        cuda_name = torch.cuda.get_device_name(0) if cuda_available else None
        matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
        if matmul is not None and hasattr(matmul, "fp32_precision"):
            matmul_tf32 = matmul.fp32_precision != "ieee"
            cudnn_tf32 = torch.backends.cudnn.fp32_precision != "ieee"
        else:
            matmul_tf32 = getattr(matmul, "allow_tf32", None)
            cudnn_tf32 = getattr(torch.backends.cudnn, "allow_tf32", None)
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "weight_sha256": self.weight_sha256,
            "weight_file_hashes": dict(self.weight_file_hashes),
            "artifact_file_hashes": {
                name: dict(fingerprint)
                for name, fingerprint in self.artifact_file_hashes.items()
            },
            "tokenizer_file_hashes": dict(self.tokenizer_file_hashes),
            "model_config_sha256": self.model_config_sha256,
            "tokenizer_identity": dict(self.tokenizer_identity),
            "pooling": self.profile["pooling"],
            "prefix": self.profile["prefix"],
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "device": str(self.device),
            "precision": "float32",
            "local_files_only": self.local_files_only,
            "threads": {
                "intra_op": torch.get_num_threads(),
                "inter_op": torch.get_num_interop_threads(),
            },
            "determinism": {
                "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "matmul_allow_tf32": matmul_tf32,
                "cudnn_allow_tf32": cudnn_tf32,
                "cublas_workspace_config": os.environ.get(_CUBLAS_WORKSPACE_CONFIG),
            },
            "packages": dict(self.package_versions),
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor() or None,
                "cuda_available": cuda_available,
                "cuda_device": cuda_name,
                "cuda_runtime": getattr(torch.version, "cuda", None),
                "cudnn_version": torch.backends.cudnn.version(),
            },
        }


__all__ = ["FrozenTextEmbedder"]
