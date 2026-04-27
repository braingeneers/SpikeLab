"""Tests for the HIPPIE cell-type classification adapter.

All tests mock the HuggingFace download and model forward pass so the
293 MB checkpoint is never fetched during CI.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# Skip every test in this file if hippie is not installed
hippie = pytest.importorskip("hippie", reason="spikelab[hippie] not installed")

from spikelab.spikedata.hippie_adapter import (
    _isi_histogram,
    _autocorrelogram,
    _preprocess_waveform,
    extract_features,
    classify_neurons,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_spike_train(n_spikes=200, duration_s=60.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0, duration_s, n_spikes))


def _make_waveform(n=82, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n)
    return np.sin(t) + rng.normal(0, 0.05, n)


def _make_spikedata(n_units=10, seed=0):
    """Return a minimal SpikeData with avg_waveform in neuron_attributes."""
    from spikelab.spikedata import SpikeData

    rng = np.random.default_rng(seed)
    trains = [_make_spike_train(200 + i * 10, seed=seed + i) for i in range(n_units)]
    waveforms = [_make_waveform(seed=seed + i) for i in range(n_units)]
    attrs = [{"avg_waveform": w} for w in waveforms]
    return SpikeData(trains, length=60.0, neuron_attributes=attrs)


# ---------------------------------------------------------------------------
# Unit tests for preprocessing helpers
# ---------------------------------------------------------------------------


class TestPreprocessWaveform:
    def test_output_shape(self):
        wave = _make_waveform(82)
        out = _preprocess_waveform(wave, target=50)
        assert out.shape == (50,)

    def test_range(self):
        wave = _make_waveform(82)
        out = _preprocess_waveform(wave)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_flat_waveform_does_not_crash(self):
        wave = np.zeros(50)
        out = _preprocess_waveform(wave)
        assert out.shape == (50,)
        assert np.isfinite(out).all()


class TestISIHistogram:
    def test_output_shape(self):
        st = _make_spike_train(200)
        hist = _isi_histogram(st, n_bins=100)
        assert hist.shape == (100,)

    def test_range(self):
        st = _make_spike_train(200)
        hist = _isi_histogram(st)
        assert hist.min() >= -1.0 - 1e-5
        assert hist.max() <= 1.0 + 1e-5

    def test_silent_neuron(self):
        hist = _isi_histogram(np.array([0.5]), n_bins=100)
        assert hist.shape == (100,)
        assert np.isfinite(hist).all()


class TestAutocorrelogram:
    def test_output_shape(self):
        st = _make_spike_train(200)
        acg = _autocorrelogram(st, n_bins=100)
        assert acg.shape == (100,)

    def test_range(self):
        st = _make_spike_train(200)
        acg = _autocorrelogram(st)
        assert acg.min() >= -1.0 - 1e-5
        assert acg.max() <= 1.0 + 1e-5

    def test_empty_train(self):
        acg = _autocorrelogram(np.array([]), n_bins=100)
        assert acg.shape == (100,)
        assert (acg == 0).all()


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_shapes(self):
        sd = _make_spikedata(n_units=8)
        feats = extract_features(sd)
        assert feats["wave"].shape == (8, 50)
        assert feats["isi"].shape == (8, 100)
        assert feats["acg"].shape == (8, 100)

    def test_dtype(self):
        sd = _make_spikedata(n_units=5)
        feats = extract_features(sd)
        for arr in feats.values():
            assert arr.dtype == np.float32

    def test_no_waveform_raises(self):
        from spikelab.spikedata import SpikeData

        trains = [_make_spike_train(100, seed=i) for i in range(3)]
        sd = SpikeData(trains, length=60.0)
        with pytest.raises(ValueError, match="avg_waveform"):
            extract_features(sd)


# ---------------------------------------------------------------------------
# classify_neurons — mocked end-to-end
# ---------------------------------------------------------------------------


class TestClassifyNeurons:
    """Full pipeline test with the HuggingFace download and HIPPIE model mocked out."""

    def _make_mock_classifier(self, n_neurons, z_dim=30):
        mock_clf = MagicMock()
        mock_clf.get_embeddings.return_value = np.random.randn(n_neurons, z_dim).astype(
            np.float32
        )
        mock_clf.umap_reduce.return_value = np.random.randn(n_neurons, 2).astype(
            np.float32
        )
        mock_clf.hdbscan_cluster.return_value = np.zeros(n_neurons, dtype=np.int32)
        return mock_clf

    @patch("hippie.inference.HIPPIEClassifier")
    def test_returns_all_keys_by_default(self, MockCls):
        n = 10
        MockCls.from_pretrained.return_value = self._make_mock_classifier(n)
        sd = _make_spikedata(n_units=n)
        result = classify_neurons(sd)
        assert "embeddings" in result
        assert "umap_coords" in result
        assert "cluster_labels" in result

    @patch("hippie.inference.HIPPIEClassifier")
    def test_no_umap_no_hdbscan(self, MockCls):
        n = 6
        MockCls.from_pretrained.return_value = self._make_mock_classifier(n)
        sd = _make_spikedata(n_units=n)
        result = classify_neurons(sd, run_umap=False, run_hdbscan=False)
        assert "embeddings" in result
        assert "umap_coords" not in result
        assert "cluster_labels" not in result

    @patch("hippie.inference.HIPPIEClassifier")
    def test_embedding_shape(self, MockCls):
        n = 12
        MockCls.from_pretrained.return_value = self._make_mock_classifier(n, z_dim=30)
        sd = _make_spikedata(n_units=n)
        result = classify_neurons(sd)
        assert result["embeddings"].shape == (n, 30)

    @patch("hippie.inference.HIPPIEClassifier")
    def test_tech_id_string(self, MockCls):
        n = 5
        mock_clf = self._make_mock_classifier(n)
        MockCls.from_pretrained.return_value = mock_clf
        sd = _make_spikedata(n_units=n)
        classify_neurons(sd, tech_id="silicon_probe")
        mock_clf.get_embeddings.assert_called_once()
        call_kwargs = mock_clf.get_embeddings.call_args
        assert call_kwargs.kwargs.get("tech_id") == "silicon_probe"
