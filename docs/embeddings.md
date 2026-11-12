# Frozen text embeddings

The optional embedding module converts segmented text into fixed, normalized
vectors. It supports two multilingual encoder profiles:

| Model | Input prefix | Pooling |
|---|---|---|
| `intfloat/multilingual-e5-small` | `query: ` on every window | Attention-mask-weighted mean of final hidden states |
| `BAAI/bge-m3` | None | First-token (`CLS`) final hidden state |

The E5 model card recommends `query: ` when embeddings are used as features and
shows masked mean pooling with a 512-token model limit. The BGE-M3 card says its
dense retrieval path does not need an instruction prefix; the released pooling
configuration uses the CLS token. This module uses FP32 for both models.

## Install the optional dependencies

```console
python -m pip install ".[embeddings]"
```

The `embeddings` extra is kept separate from the core package because PyTorch
and Transformers are large optional dependencies. Importing `cogtask_ci` or the
embedding module does not download model weights.

## Pin a model revision

Pass a full 40-character Hub commit SHA. Branch names such as `main` are
rejected so that model updates cannot silently change an embedding run. The
currently reviewed upstream revisions are:

| Model | Hub revision | Checkpoint weight file SHA-256 |
|---|---|---|
| `intfloat/multilingual-e5-small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | `1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477` (`model.safetensors`) |
| `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38` (`pytorch_model.bin`) |

These hashes identify the Hub weight files. The encoder also computes the local
file hash when it loads a checkpoint and exposes it in `runtime_metadata()` and
the label-free cache key.

## Encode segmented text

Supply each transcript as a sequence of text segments. For records that store
tokens, join the tokens within each segment with spaces before encoding.

```python
from cogtask_ci.embeddings import FrozenTextEmbedder

encoder = FrozenTextEmbedder(
    "intfloat/multilingual-e5-small",
    revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    cache_dir="./model-cache",
    device="auto",
    max_tokens=512,
    batch_size=8,
    seed=0,
    threads=1,
)

texts = [
    ["A short synthetic example.", "A second segment stays separate."],
    ["Another synthetic example."],
]
vectors = encoder.encode(texts)
print(vectors.shape)  # (2, 384) for multilingual-e5-small
metadata = encoder.runtime_metadata()
```

`encode_texts(["..."])` is a convenience for unsegmented strings. `encode`
returns one FP32 vector per transcript. The vector dimension is 384 for E5-small
and 1024 for BGE-M3.

## Windowing and pooling

The model-token limit includes the E5 prefix, tokenizer-added special tokens,
and all text tokens. Text segments are packed in order without crossing their
boundaries. If one segment is too long for a window, it is split into adjacent
non-overlapping token slices. All source content tokens are covered exactly
once; window separators and special tokens do not count as content.

Each window produces the model-specific pooled vector. Window vectors are
averaged using the number of source content tokens in each window, then the
transcript vector is L2-normalized. Empty transcripts, non-finite outputs, and
zero vectors raise errors.

## Determinism and provenance

Construction seeds Python, NumPy, and PyTorch; selects FP32; turns on PyTorch
deterministic algorithms; disables cuDNN benchmarking and TF32; and records
package versions, thread counts, platform, and accelerator details. The
`CUBLAS_WORKSPACE_CONFIG` is set before CUDA initialization when possible. If
CUDA was initialized too early or PyTorch cannot provide the requested thread
configuration, construction raises an error rather than silently weakening the
determinism settings. Use a fresh process when changing runtime configuration.

Keep the returned `runtime_metadata()` with each run. It records the exact
model revision, local weight/config/tokenizer file hashes, pooling, prefix,
token limit, batch size, device, seed, thread counts, deterministic settings,
package versions, and hardware.

`encoder.cache_key(segments)` hashes the preprocessed text segments and their
order, the preprocessing version, model revision, local weight/config/tokenizer
file hashes, relevant package versions, and embedding configuration. It accepts
no labels and is not a cache of evaluation labels or predictions.

For inference without network access, download the pinned model snapshot first,
set `HF_HUB_OFFLINE=1`, and pass `local_files_only=True`. The model cache can be
large; choose a dedicated cache directory rather than clearing a shared cache.

## Model references

- [Multilingual E5-small model card](https://huggingface.co/intfloat/multilingual-e5-small)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
- [BGE-M3 CLS-pooling configuration at the pinned revision](https://huggingface.co/BAAI/bge-m3/blob/5617a9f61b028005a4858fdac845db406aefb181/1_Pooling/config.json)
