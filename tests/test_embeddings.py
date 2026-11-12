"""Offline unit tests for the frozen embedding adapter."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - embeddings are an optional extra
    torch = None

from cogtask_ci.embeddings import FrozenTextEmbedder


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        # A deterministic character tokenizer keeps all tests independent of Hub downloads.
        return [200 + ord(char) for char in text]

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, token_ids):
        return [101, *token_ids, 102]


if torch is not None:
    class FakeModel(torch.nn.Module):
        def __init__(self, *, nonfinite=False):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=3)
            self.nonfinite = nonfinite

        def forward(self, input_ids, attention_mask):
            values = input_ids.to(dtype=torch.float32)
            hidden = torch.stack((values, values * 0 + 1, values.square()), dim=-1)
            if self.nonfinite:
                hidden[0, 0, 0] = float("nan")
            return SimpleNamespace(last_hidden_state=hidden)


@unittest.skipIf(torch is None, "install the embeddings extra to exercise model pooling")
class FrozenEmbeddingContracts(unittest.TestCase):
    e5_revision = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    bge_revision = "5617a9f61b028005a4858fdac845db406aefb181"
    weight_hash = "a" * 64

    def make_embedder(self, model_id="intfloat/multilingual-e5-small", **kwargs):
        revision = self.e5_revision if model_id.startswith("intfloat/") else self.bge_revision
        return FrozenTextEmbedder(
            model_id,
            revision,
            tokenizer=FakeTokenizer(),
            model=FakeModel(),
            weight_sha256=self.weight_hash,
            device="cpu",
            **kwargs,
        )

    def test_requires_supported_model_and_full_revision(self):
        with self.assertRaises(ValueError):
            FrozenTextEmbedder("unknown/model", self.e5_revision)
        with self.assertRaises(ValueError):
            FrozenTextEmbedder("intfloat/multilingual-e5-small", "main")

    def test_oversized_segment_splits_without_overlap_and_respects_limit(self):
        embedder = self.make_embedder(max_tokens=12)
        segments = ["abcdefg", "hi"]
        windows = embedder._prepare_windows(segments)
        self.assertEqual(sum(window.content_tokens for window in windows), 9)
        self.assertTrue(all(len(window.input_ids) <= 12 for window in windows))

        prefix = [200 + ord(char) for char in "query: "]
        observed = []
        for window in windows:
            payload = list(window.input_ids[1:-1])
            self.assertEqual(payload[: len(prefix)], prefix)
            observed.extend(payload[len(prefix) :])
        expected = [200 + ord(char) for char in "abcdefghi"]
        self.assertEqual(observed, expected)

    def test_segments_pack_at_boundaries_and_count_only_source_tokens(self):
        embedder = self.make_embedder(max_tokens=16)
        windows = embedder._prepare_windows(["ab", "cd", "e"])
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].content_tokens, 5)
        prefix = [200 + ord(char) for char in "query: "]
        payload = list(windows[0].input_ids[1:-1])[len(prefix) :]
        self.assertEqual(payload, [200 + ord(char) for char in "ab\ncd\ne"])

    def test_e5_and_bge_profiles_produce_finite_unit_vectors(self):
        text = "synthetic example"
        e5 = self.make_embedder()
        bge = self.make_embedder("BAAI/bge-m3")
        e5_vector = e5.encode_texts([text])[0]
        bge_vector = bge.encode_texts([text])[0]
        self.assertEqual(e5_vector.shape, (3,))
        self.assertTrue(np.isfinite(e5_vector).all())
        self.assertAlmostEqual(float(np.linalg.norm(e5_vector)), 1.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(bge_vector)), 1.0, places=6)
        self.assertFalse(np.allclose(e5_vector, bge_vector))

    def test_window_mean_weights_by_content_tokens_before_l2(self):
        embedder = self.make_embedder(max_tokens=12)
        segments = ["abcdefghij"]
        windows = embedder._prepare_windows(segments)
        per_window = []
        weights = []
        for window in windows:
            ids = np.asarray(window.input_ids, dtype=np.float64)
            per_window.append(np.array([ids.mean(), 1.0, np.square(ids).mean()]))
            weights.append(window.content_tokens)
        expected = np.average(np.asarray(per_window), axis=0, weights=weights)
        expected /= np.linalg.norm(expected)

        actual = embedder.encode([segments])[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_cache_key_tracks_text_preprocessing_revision_weights_and_config(self):
        first = self.make_embedder(max_tokens=32)
        second = self.make_embedder(max_tokens=31)
        key = first.cache_key(["  e\u0301  ", "next segment"])
        self.assertEqual(key, first.cache_key(["é", "next segment"]))
        self.assertNotEqual(key, second.cache_key(["é", "next segment"]))
        self.assertNotEqual(key, first.cache_key(["different text"]))

    def test_empty_and_nonfinite_embeddings_fail_explicitly(self):
        embedder = self.make_embedder()
        with self.assertRaises(ValueError):
            embedder.encode([[]])
        with self.assertRaises(ValueError):
            embedder.encode_texts([" \t\n "])

        bad = FrozenTextEmbedder(
            "BAAI/bge-m3",
            self.bge_revision,
            tokenizer=FakeTokenizer(),
            model=FakeModel(nonfinite=True),
            weight_sha256=self.weight_hash,
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            bad.encode_texts(["synthetic"])

    def test_runtime_metadata_records_versions_device_and_determinism(self):
        metadata = self.make_embedder().runtime_metadata()
        self.assertEqual(metadata["precision"], "float32")
        self.assertEqual(metadata["revision"], self.e5_revision)
        self.assertEqual(metadata["determinism"]["algorithms_enabled"], True)
        self.assertFalse(metadata["determinism"]["cudnn_benchmark"])
        self.assertEqual(metadata["threads"]["intra_op"], 1)
        self.assertIn("torch", metadata["packages"])
        self.assertIn("platform", metadata["hardware"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
