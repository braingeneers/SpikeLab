"""Tests for spikedata/plot_utils.py — plot_heatmap and plot_recording."""

import pathlib
import sys

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import matplotlib.figure

# Ensure project root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpikeLab.spikedata import SpikeData
from SpikeLab.spikedata.plot_utils import plot_heatmap, plot_recording

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sd(n_units=3, length=400.0):
    """Create a small SpikeData for testing."""
    rng = np.random.default_rng(42)
    trains = [sorted(rng.uniform(0, length, size=10).tolist()) for _ in range(n_units)]
    return SpikeData(trains, N=n_units, length=length)


@pytest.fixture(autouse=True)
def close_figs():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_heatmap tests
# ---------------------------------------------------------------------------


class TestPlotHeatmap:
    """Tests for the plot_heatmap standalone function."""

    def test_standalone_returns_fig_and_ax(self):
        """
        Calling without an ax creates a standalone figure.

        Tests:
            (Test Case 1) Returns a (fig, ax) tuple when ax is None.
        """
        data = np.random.rand(5, 20)
        result = plot_heatmap(data)
        assert isinstance(result, tuple)
        assert len(result) == 2
        fig, ax = result
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_on_existing_ax_returns_ax(self):
        """
        Passing an existing Axes returns just the Axes.

        Tests:
            (Test Case 1) Returns a single Axes object (not a tuple).
        """
        fig, ax = plt.subplots()
        data = np.random.rand(4, 10)
        result = plot_heatmap(data, ax=ax)
        assert result is ax

    def test_row_normalisation(self):
        """
        Row normalisation scales each row to [0, 1].

        Tests:
            (Test Case 1) With norm='row', the imshow data has max 1.0
                per row (for rows with non-constant values).
        """
        data = np.array([[0, 5, 10], [2, 2, 2]], dtype=float)
        fig, ax = plot_heatmap(data, norm="row")
        im_data = ax.images[0].get_array()
        # First row: [0, 0.5, 1.0]; second row: constant → unchanged
        np.testing.assert_allclose(im_data[0], [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(im_data[1], [2, 2, 2])

    def test_custom_cmap(self):
        """
        The cmap parameter is applied to the imshow.

        Tests:
            (Test Case 1) Passing cmap='viridis' sets that colormap.
        """
        data = np.random.rand(3, 10)
        fig, ax = plot_heatmap(data, cmap="viridis")
        assert ax.images[0].cmap.name == "viridis"

    def test_default_cmap_is_hot(self):
        """
        Default colormap is 'hot'.

        Tests:
            (Test Case 1) Without specifying cmap, the image uses 'hot'.
        """
        data = np.random.rand(3, 10)
        fig, ax = plot_heatmap(data)
        assert ax.images[0].cmap.name == "hot"

    def test_extent_parameter(self):
        """
        The extent parameter is forwarded to imshow.

        Tests:
            (Test Case 1) Setting extent=(0, 100, 0, 5) maps pixel coords
                to those data coordinates.
        """
        data = np.random.rand(5, 20)
        ext = (0, 100, 0, 5)
        fig, ax = plot_heatmap(data, extent=ext)
        assert ax.images[0].get_extent() == list(ext)

    def test_vlines_and_hlines(self):
        """
        Vertical and horizontal lines are added to the axes.

        Tests:
            (Test Case 1) One vline and one hline added.
        """
        data = np.random.rand(4, 10)
        fig, ax = plot_heatmap(
            data,
            vlines=[{"x": 5, "color": "red"}],
            hlines=[{"y": 2, "color": "blue"}],
        )
        # axvline and axhline add Line2D objects to ax.lines
        assert len(ax.lines) >= 2

    def test_no_colorbar(self):
        """
        Setting show_colorbar=False suppresses the colorbar.

        Tests:
            (Test Case 1) Figure has only one Axes (no colorbar axes).
        """
        data = np.random.rand(3, 10)
        fig, ax = plot_heatmap(data, show_colorbar=False)
        assert len(fig.axes) == 1

    def test_save_path(self, tmp_path):
        """
        Providing save_path saves the figure and closes it.

        Tests:
            (Test Case 1) File is created at the given path.
        """
        data = np.random.rand(3, 10)
        out = tmp_path / "heatmap.png"
        plot_heatmap(data, save_path=str(out))
        assert out.exists()

    def test_custom_labels(self):
        """
        Custom xlabel and ylabel are applied.

        Tests:
            (Test Case 1) Axes labels match the provided strings.
        """
        data = np.random.rand(3, 10)
        fig, ax = plot_heatmap(data, xlabel="Bins", ylabel="Neuron")
        assert ax.get_xlabel() == "Bins"
        assert ax.get_ylabel() == "Neuron"

    def test_custom_ticks(self):
        """
        Custom xticks and yticks are applied.

        Tests:
            (Test Case 1) Tick positions and labels match provided values.
        """
        data = np.random.rand(4, 10)
        fig, ax = plot_heatmap(
            data,
            xticks=([0, 5, 9], ["0ms", "50ms", "90ms"]),
            yticks=([0, 3], ["U1", "U4"]),
        )
        xt = [t.get_text() for t in ax.get_xticklabels()]
        yt = [t.get_text() for t in ax.get_yticklabels()]
        assert xt == ["0ms", "50ms", "90ms"]
        assert yt == ["U1", "U4"]


# ---------------------------------------------------------------------------
# plot_recording tests
# ---------------------------------------------------------------------------


class TestPlotRecording:
    """Tests for plot_recording and SpikeData.plot."""

    def test_raster_only(self):
        """
        Raster-only figure has one panel.

        Tests:
            (Test Case 1) Returns a Figure with exactly 1 Axes.
        """
        sd = _make_sd()
        fig = plot_recording(sd, show_raster=True, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 1

    def test_raster_plus_pop_rate(self):
        """
        Raster + population rate produces 2 panels.

        Tests:
            (Test Case 1) Figure has 2 Axes.
        """
        sd = _make_sd()
        fig = plot_recording(sd, show_raster=True, show_pop_rate=True, show=False)
        assert len(fig.axes) == 2

    def test_all_four_panels(self):
        """
        Enabling all 4 panels produces 4+ Axes (extras from colorbars/twinx).

        Tests:
            (Test Case 1) At least 4 Axes in the figure.
        """
        sd = _make_sd()
        fig = plot_recording(
            sd,
            show_raster=True,
            show_pop_rate=True,
            show_fr_rates=True,
            model_states=np.random.rand(5, 10),
            cont_prob=np.random.rand(10),
            show=False,
        )
        assert len(fig.axes) >= 4

    def test_no_panels_raises(self):
        """
        Disabling all panels raises ValueError.

        Tests:
            (Test Case 1) ValueError with descriptive message.
        """
        sd = _make_sd()
        with pytest.raises(ValueError, match="No panels enabled"):
            plot_recording(sd, show_raster=False, show=False)

    def test_auto_enable_pop_rate_from_data(self):
        """
        Passing pop_rate auto-enables the population rate panel.

        Tests:
            (Test Case 1) Figure has 2 panels (raster + pop_rate).
        """
        sd = _make_sd()
        pop = sd.get_pop_rate()
        fig = plot_recording(sd, show_raster=True, pop_rate=pop, show=False)
        assert len(fig.axes) == 2

    def test_auto_enable_from_cont_prob(self):
        """
        Passing cont_prob auto-enables the population rate panel.

        Tests:
            (Test Case 1) Pop rate panel appears even without show_pop_rate=True.
        """
        sd = _make_sd()
        fig = plot_recording(
            sd,
            show_raster=True,
            cont_prob=np.random.rand(10),
            show=False,
        )
        # raster + pop_rate (auto-enabled) → at least 2 base axes
        assert len(fig.axes) >= 2

    def test_auto_enable_from_fr_rates(self):
        """
        Passing fr_rates auto-enables the FR heatmap panel.

        Tests:
            (Test Case 1) FR heatmap appears without show_fr_rates=True.
        """
        sd = _make_sd()
        raster_T = sd.sparse_raster(bin_size=1.0).shape[1]
        fr = np.random.rand(3, raster_T)
        fig = plot_recording(sd, show_raster=True, fr_rates=fr, show=False)
        # raster + fr_heatmap + colorbar
        assert len(fig.axes) >= 2

    def test_auto_enable_from_model_states(self):
        """
        Passing model_states auto-enables the model states panel.

        Tests:
            (Test Case 1) Model states panel appears without show_model_states=True.
        """
        sd = _make_sd()
        fig = plot_recording(
            sd,
            show_raster=True,
            model_states=np.random.rand(5, 20),
            show=False,
        )
        assert len(fig.axes) >= 2

    def test_imshow_raster_style(self):
        """
        Raster with imshow style shows an image, not eventplot.

        Tests:
            (Test Case 1) The raster axes contain an AxesImage.
        """
        sd = _make_sd()
        fig = plot_recording(sd, show_raster=True, raster_style="imshow", show=False)
        ax = fig.axes[0]
        assert len(ax.images) == 1

    def test_eventplot_raster_style(self):
        """
        Raster with eventplot style uses EventCollection, no images.

        Tests:
            (Test Case 1) The raster axes contain no AxesImage.
        """
        sd = _make_sd()
        fig = plot_recording(sd, show_raster=True, raster_style="eventplot", show=False)
        ax = fig.axes[0]
        assert len(ax.images) == 0

    def test_sort_indices(self):
        """
        sort_indices reorders units in the raster.

        Tests:
            (Test Case 1) Reversed sort_indices flips the raster rows.
        """
        sd = _make_sd(n_units=3)
        order = [2, 1, 0]

        fig_normal = plot_recording(
            sd, show_raster=True, raster_style="imshow", show=False
        )
        raster_normal = fig_normal.axes[0].images[0].get_array()

        fig_sorted = plot_recording(
            sd, show_raster=True, raster_style="imshow", sort_indices=order, show=False
        )
        raster_sorted = fig_sorted.axes[0].images[0].get_array()

        np.testing.assert_array_equal(raster_sorted, raster_normal[order, :])

    def test_time_range_crops(self):
        """
        time_range crops the raster to the specified window.

        Tests:
            (Test Case 1) Raster x-limits match the cropped range width.
        """
        sd = _make_sd(length=400.0)
        fig = plot_recording(sd, show_raster=True, time_range=(100, 300), show=False)
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        assert xlim[0] == 0
        assert xlim[1] == 200  # 300 - 100

    def test_gplvm_result_auto_extract(self):
        """
        gplvm_result dict auto-extracts model_states and cont_prob.

        Tests:
            (Test Case 1) Passing gplvm_result enables model_states and
                pop_rate panels automatically.
        """
        sd = _make_sd()
        gplvm_res = {
            "decode_res": {
                "posterior_latent_marg": np.random.rand(10, 5),
                "posterior_dynamics_marg": np.random.rand(10, 2),
            }
        }
        fig = plot_recording(sd, show_raster=True, gplvm_result=gplvm_res, show=False)
        # raster + pop_rate (from cont_prob) + model_states → ≥3 base axes
        assert len(fig.axes) >= 3

    def test_burst_overlays(self):
        """
        Burst times and edges are drawn on the population rate panel.

        Tests:
            (Test Case 1) Burst markers appear as scatter points.
            (Test Case 2) Burst edge spans appear as patches.
        """
        sd = _make_sd(length=400.0)
        bt = np.array([50.0, 150.0, 250.0])
        be = np.array([[40.0, 60.0], [140.0, 160.0], [240.0, 260.0]])
        fig = plot_recording(
            sd,
            show_raster=True,
            show_pop_rate=True,
            burst_times=bt,
            burst_edges=be,
            show=False,
        )
        # Pop rate is axes[1]; should have scatter (PathCollection)
        pop_ax = fig.axes[1]
        # At least one collection (scatter) and patches (axvspan)
        assert len(pop_ax.collections) >= 1
        assert len(pop_ax.patches) >= 3

    def test_different_time_resolution(self):
        """
        Data arrays with different time resolution than the raster are
        handled without error by linear x-axis scaling.

        Tests:
            (Test Case 1) model_states with 8 bins on a 400ms recording
                does not crash.
        """
        sd = _make_sd(length=400.0)
        fig = plot_recording(
            sd,
            show_raster=True,
            model_states=np.random.rand(5, 8),
            cont_prob=np.random.rand(8),
            show=False,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_save_path(self, tmp_path):
        """
        Providing save_path saves the figure to disk.

        Tests:
            (Test Case 1) File is created at the specified path.
        """
        sd = _make_sd()
        out = tmp_path / "recording.png"
        fig = plot_recording(sd, show_raster=True, save_path=str(out), show=False)
        assert out.exists()

    def test_pop_rate_only_no_raster(self):
        """
        Pop rate panel works without raster.

        Tests:
            (Test Case 1) Figure with only pop rate panel.
        """
        sd = _make_sd()
        fig = plot_recording(sd, show_raster=False, show_pop_rate=True, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 1

    def test_spikedata_plot_wrapper(self):
        """
        SpikeData.plot() delegates to plot_recording.

        Tests:
            (Test Case 1) Returns a Figure, same as calling plot_recording
                directly.
        """
        sd = _make_sd()
        fig = sd.plot(show_raster=True, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_figsize(self):
        """
        Custom figsize is applied to the figure.

        Tests:
            (Test Case 1) Figure dimensions match the provided figsize.
        """
        sd = _make_sd()
        fig = plot_recording(sd, show_raster=True, figsize=(8, 4), show=False)
        w, h = fig.get_size_inches()
        assert w == pytest.approx(8)
        assert h == pytest.approx(4)

    def test_custom_height_ratios(self):
        """
        Custom height_ratios are used for panel sizing.

        Tests:
            (Test Case 1) No error when providing matching height_ratios.
        """
        sd = _make_sd()
        fig = plot_recording(
            sd,
            show_raster=True,
            show_pop_rate=True,
            height_ratios=[3, 1],
            show=False,
        )
        assert len(fig.axes) == 2
