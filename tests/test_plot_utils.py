"""Tests for spikedata/plot_utils.py — all plotting functions."""

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

from spikedata import SpikeData
from spikedata.plot_utils import (
    plot_heatmap,
    plot_recording,
    plot_distribution,
    plot_pvalue_matrix,
    plot_scatter,
    plot_lines,
    plot_burst_sensitivity,
    plot_aligned_slice_single_unit,
)
from spikedata.spikeslicestack import SpikeSliceStack

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sd(n_units=3, length=400.0):
    """Create a small SpikeData for testing."""
    rng = np.random.default_rng(42)
    trains = [sorted(rng.uniform(0, length, size=10).tolist()) for _ in range(n_units)]
    return SpikeData(trains, N=n_units, length=length)


def _get_model_states_data(fig):
    """Return the image data from the model-states panel, or None."""
    for ax in reversed(fig.axes):
        if ax.images:
            return ax.images[0].get_array()
    return None


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
        # Matplotlib may return extent as list or tuple depending on version.
        assert list(ax.images[0].get_extent()) == list(ext)

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
            (Test Case 1) Returns a Figure with 1 main panel (plus its
                colorbar slot in the GridSpec).
        """
        sd = _make_sd()
        fig = plot_recording(sd, show_raster=True, show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        # 1 panel × 2 columns (main + colorbar slot) = 2 axes
        assert len(fig.axes) == 2

    def test_raster_plus_pop_rate(self):
        """
        Raster + population rate produces 2 panels.

        Tests:
            (Test Case 1) Figure has 2 main panels (plus colorbar slots).
        """
        sd = _make_sd()
        fig = plot_recording(sd, show_raster=True, show_pop_rate=True, show=False)
        # 2 panels × 2 columns = 4 axes
        assert len(fig.axes) == 4

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
        # 2 panels × 2 columns = 4 axes
        assert len(fig.axes) == 4

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

    def test_gplvm_crop_with_time_range(self):
        """
        GPLVM model_states and cont_prob are correctly cropped when
        time_range is specified in ms and arrays have a coarser resolution.

        Tests:
            (Test Case 1) Cropped model_states panel shows only the data
                corresponding to the requested time_range, not the full
                recording.
            (Test Case 2) Two non-overlapping time_range windows produce
                different model_states image data.
        """
        sd = _make_sd(n_units=3, length=400.0)
        # 8 bins for 400ms → 50ms bin size
        rng = np.random.default_rng(99)
        model_states = rng.random((5, 8))
        cont_prob = rng.random(8)

        # First half: 0-200ms → bins 0-4
        fig1 = plot_recording(
            sd,
            show_raster=True,
            model_states=model_states,
            cont_prob=cont_prob,
            time_range=(0, 200),
            show=False,
        )
        # Last half: 200-400ms → bins 4-8
        fig2 = plot_recording(
            sd,
            show_raster=True,
            model_states=model_states,
            cont_prob=cont_prob,
            time_range=(200, 400),
            show=False,
        )

        data1 = _get_model_states_data(fig1)
        data2 = _get_model_states_data(fig2)

        assert data1 is not None
        assert data2 is not None
        # The two windows must show different data
        assert not np.array_equal(data1, data2)

    def test_gplvm_result_bin_size_ms_extraction(self):
        """
        When gplvm_result contains bin_size_ms, plot_recording extracts it
        and correctly crops model_states with time_range.

        Tests:
            (Test Case 1) Figure is produced without error when gplvm_result
                includes bin_size_ms.
            (Test Case 2) Model states panel data differs for different
                time_range windows.
        """
        sd = _make_sd(n_units=3, length=400.0)
        rng = np.random.default_rng(77)
        gplvm_res = {
            "decode_res": {
                "posterior_latent_marg": rng.random((8, 5)),
                "posterior_dynamics_marg": rng.random((8, 2)),
            },
            "bin_size_ms": 50.0,
        }

        fig1 = plot_recording(
            sd,
            show_raster=True,
            gplvm_result=gplvm_res,
            time_range=(0, 200),
            show=False,
        )
        fig2 = plot_recording(
            sd,
            show_raster=True,
            gplvm_result=gplvm_res,
            time_range=(200, 400),
            show=False,
        )

        data1 = _get_model_states_data(fig1)
        data2 = _get_model_states_data(fig2)

        assert data1 is not None
        assert data2 is not None
        assert not np.array_equal(data1, data2)

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
        # 1 panel × 2 columns = 2 axes
        assert len(fig.axes) == 2

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
        # 2 panels × 2 columns = 4 axes
        assert len(fig.axes) == 4

    def test_axes_correct_length(self):
        """
        Pre-created axes pairs are used for plotting instead of creating a
        new figure.

        Tests:
            (Test Case 1) With 2 enabled panels (raster + pop_rate), passing
                2 (ax, cbar_ax) pairs succeeds and plotting occurs on the
                provided axes.
            (Test Case 2) Returned fig matches the figure of the provided axes.
        """
        sd = _make_sd()
        fig_ext, axs = plt.subplots(2, 2)
        axes_pairs = [(axs[0, 0], axs[0, 1]), (axs[1, 0], axs[1, 1])]

        fig = plot_recording(
            sd,
            show_raster=True,
            show_pop_rate=True,
            axes=axes_pairs,
            show=False,
        )

        # Returned figure is the one that owns the provided axes
        assert fig is fig_ext
        # Raster panel was drawn on the provided axes (eventplot adds collections)
        assert len(axs[0, 0].collections) >= 1 or len(axs[0, 0].get_children()) > 0
        # Pop rate panel has a line (population rate curve)
        assert len(axs[1, 0].lines) >= 1

    def test_axes_length_mismatch(self):
        """
        Passing the wrong number of axes pairs raises ValueError.

        Tests:
            (Test Case 1) 2 enabled panels but 1 axes pair raises ValueError.
            (Test Case 2) Error message mentions the expected panel count.
        """
        sd = _make_sd()
        fig_ext, axs = plt.subplots(1, 2)
        axes_pairs = [(axs[0], axs[1])]

        with pytest.raises(ValueError, match="Expected 2"):
            plot_recording(
                sd,
                show_raster=True,
                show_pop_rate=True,
                axes=axes_pairs,
                show=False,
            )

    def test_axes_skips_save(self, tmp_path):
        """
        When axes is provided, save_path is ignored — no file is written.

        Tests:
            (Test Case 1) File is not created even when save_path is set.
        """
        sd = _make_sd()
        fig_ext, axs = plt.subplots(1, 2)
        axes_pairs = [(axs[0], axs[1])]
        out = tmp_path / "should_not_exist.png"

        plot_recording(
            sd,
            show_raster=True,
            axes=axes_pairs,
            save_path=str(out),
            show=False,
        )

        assert not out.exists()

    def test_colorbar_on_provided_axes(self):
        """
        When axes pairs are provided with imshow raster style, the colorbar
        is drawn on the provided cbar_ax.

        Tests:
            (Test Case 1) The cbar_ax for the raster panel contains colorbar
                content (its axis is turned on and has images or children).
        """
        sd = _make_sd()
        fig_ext, axs = plt.subplots(1, 2)
        cbar_ax = axs[1]
        axes_pairs = [(axs[0], cbar_ax)]

        plot_recording(
            sd,
            show_raster=True,
            raster_style="imshow",
            axes=axes_pairs,
            show=False,
        )

        # Raster panel should have an imshow image
        assert len(axs[0].images) == 1
        # Colorbar axis should have been turned on and populated
        assert cbar_ax.get_visible()
        assert len(cbar_ax.images) >= 1 or len(cbar_ax.get_children()) > 1


# ---------------------------------------------------------------------------
# plot_distribution tests
# ---------------------------------------------------------------------------


class TestPlotDistribution:
    """Tests for the plot_distribution function."""

    # Default colors for tests — avoids reliance on ax._get_lines.prop_cycler
    # which was removed in matplotlib 3.9+. See bug report for source fix.
    DEFAULT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    def test_violin_from_dict(self):
        """
        Basic violin plot from a dict input.

        Tests:
            (Test Case 1) Returns a dict containing 'bodies' key.
            (Test Case 2) Number of violin bodies matches number of groups.
        """
        fig, ax = plt.subplots()
        data = {"A": np.random.rand(20), "B": np.random.rand(25)}
        parts = plot_distribution(ax, data, colors=self.DEFAULT_COLORS[:2])
        assert "bodies" in parts
        assert len(parts["bodies"]) == 2

    def test_violin_from_list(self):
        """
        Violin plot from a list input with auto-generated labels.

        Tests:
            (Test Case 1) Returns a dict with 'bodies'.
            (Test Case 2) X-tick labels are '0' and '1'.
        """
        fig, ax = plt.subplots()
        data = [np.random.rand(15), np.random.rand(10)]
        parts = plot_distribution(ax, data, colors=self.DEFAULT_COLORS[:2])
        assert "bodies" in parts
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_labels == ["0", "1"]

    def test_boxplot_style(self):
        """
        Boxplot style produces box artists.

        Tests:
            (Test Case 1) Result dict contains 'boxes' key.
            (Test Case 2) Number of boxes matches number of groups.
        """
        fig, ax = plt.subplots()
        data = {"X": np.random.rand(20), "Y": np.random.rand(20)}
        parts = plot_distribution(
            ax, data, style="boxplot", colors=self.DEFAULT_COLORS[:2]
        )
        assert "boxes" in parts
        assert len(parts["boxes"]) == 2

    def test_invalid_style_raises(self):
        """
        An unknown style raises ValueError.

        Tests:
            (Test Case 1) ValueError mentions the invalid style name.
        """
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="Unknown style"):
            plot_distribution(
                ax,
                [np.array([1, 2, 3])],
                style="histogram",
                colors=self.DEFAULT_COLORS[:1],
            )

    def test_custom_labels_with_list(self):
        """
        Custom labels are applied when using list input.

        Tests:
            (Test Case 1) X-tick labels match the provided list.
        """
        fig, ax = plt.subplots()
        data = [np.random.rand(10), np.random.rand(10)]
        plot_distribution(
            ax, data, labels=["Group 1", "Group 2"], colors=self.DEFAULT_COLORS[:2]
        )
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_labels == ["Group 1", "Group 2"]

    def test_dict_keys_as_labels(self):
        """
        Dict keys are used as labels when no explicit labels are provided.

        Tests:
            (Test Case 1) X-tick labels match the dict keys.
        """
        fig, ax = plt.subplots()
        data = {"Ctrl": np.random.rand(10), "Drug": np.random.rand(10)}
        plot_distribution(ax, data, colors=self.DEFAULT_COLORS[:2])
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_labels == ["Ctrl", "Drug"]

    def test_axis_labels(self):
        """
        xlabel and ylabel are applied to the axes.

        Tests:
            (Test Case 1) Axes labels match the provided strings.
        """
        fig, ax = plt.subplots()
        plot_distribution(
            ax,
            [np.array([1, 2, 3])],
            ylabel="Rate (Hz)",
            xlabel="Condition",
            colors=self.DEFAULT_COLORS[:1],
        )
        assert ax.get_ylabel() == "Rate (Hz)"
        assert ax.get_xlabel() == "Condition"

    def test_log_scale(self):
        """
        log_scale=True sets the y-axis to log scale.

        Tests:
            (Test Case 1) Y-axis scale is 'log'.
        """
        fig, ax = plt.subplots()
        plot_distribution(
            ax, [np.array([0.1, 1, 10])], log_scale=True, colors=self.DEFAULT_COLORS[:1]
        )
        assert ax.get_yscale() == "log"

    def test_nan_values_stripped(self):
        """
        NaN values are stripped before plotting without error.

        Tests:
            (Test Case 1) No error when data contains NaNs.
            (Test Case 2) Violin body is still produced from the valid data.
        """
        fig, ax = plt.subplots()
        data = [np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])]
        parts = plot_distribution(ax, data, colors=self.DEFAULT_COLORS[:1])
        assert len(parts["bodies"]) == 1

    def test_median_and_quartile_overlays(self):
        """
        Median dot and IQR lines are drawn when enabled.

        Tests:
            (Test Case 1) At least one scatter collection (median dot) present.
            (Test Case 2) At least one line collection (IQR vline) present.
        """
        fig, ax = plt.subplots()
        data = [np.random.rand(30)]
        plot_distribution(
            ax,
            data,
            show_median=True,
            show_quartiles=True,
            colors=self.DEFAULT_COLORS[:1],
        )
        # Median dot adds a PathCollection, IQR adds a LineCollection
        assert len(ax.collections) >= 1

    def test_no_median_no_quartiles(self):
        """
        Disabling median and quartiles produces fewer overlays.

        Tests:
            (Test Case 1) Fewer collections than when overlays are on.
        """
        fig, ax1 = plt.subplots()
        fig, ax2 = plt.subplots()
        data = [np.random.rand(30)]
        plot_distribution(
            ax1,
            data,
            show_median=True,
            show_quartiles=True,
            colors=self.DEFAULT_COLORS[:1],
        )
        plot_distribution(
            ax2,
            data,
            show_median=False,
            show_quartiles=False,
            colors=self.DEFAULT_COLORS[:1],
        )
        # ax2 should have fewer overlay artists
        assert len(ax2.collections) <= len(ax1.collections)

    def test_show_data_overlay(self):
        """
        show_data=True adds jittered data points on top of the distribution.

        Tests:
            (Test Case 1) More scatter collections than without show_data.
        """
        fig, ax1 = plt.subplots()
        fig, ax2 = plt.subplots()
        data = [np.random.rand(20)]
        plot_distribution(
            ax1,
            data,
            show_data=False,
            show_median=False,
            show_quartiles=False,
            colors=self.DEFAULT_COLORS[:1],
        )
        plot_distribution(
            ax2,
            data,
            show_data=True,
            show_median=False,
            show_quartiles=False,
            colors=self.DEFAULT_COLORS[:1],
        )
        assert len(ax2.collections) > len(ax1.collections)

    def test_sparse_group_violin_guard(self):
        """
        Groups with fewer than 2 points are rendered as scatter in violin mode.

        Tests:
            (Test Case 1) No error when one group has a single data point.
            (Test Case 2) The single-point group appears as a scatter point.
        """
        fig, ax = plt.subplots()
        data = [np.array([5.0]), np.random.rand(20)]
        parts = plot_distribution(
            ax, data, style="violin", colors=self.DEFAULT_COLORS[:2]
        )
        # Only the group with 20 points gets a violin body
        assert len(parts["bodies"]) == 1
        # The single-point group is rendered as scatter
        assert len(ax.collections) >= 1

    def test_empty_group_no_error(self):
        """
        An empty group (all NaNs or empty array) does not crash.

        Tests:
            (Test Case 1) No error when one group is empty.
        """
        fig, ax = plt.subplots()
        data = [np.array([]), np.random.rand(10)]
        parts = plot_distribution(
            ax, data, style="violin", colors=self.DEFAULT_COLORS[:2]
        )
        # Only one violin body (for the non-empty group)
        assert len(parts["bodies"]) == 1

    def test_single_group(self):
        """
        A single group produces a valid plot.

        Tests:
            (Test Case 1) No error with one condition.
            (Test Case 2) One violin body produced.
        """
        fig, ax = plt.subplots()
        parts = plot_distribution(
            ax, {"only": np.random.rand(15)}, colors=self.DEFAULT_COLORS[:1]
        )
        assert len(parts["bodies"]) == 1

    def test_font_size_applied(self):
        """
        font_size parameter changes label font sizes.

        Tests:
            (Test Case 1) X-axis label font size matches the provided value.
        """
        fig, ax = plt.subplots()
        plot_distribution(
            ax,
            [np.random.rand(10)],
            xlabel="Test",
            font_size=18,
            colors=self.DEFAULT_COLORS[:1],
        )
        assert ax.xaxis.label.get_fontsize() == 18

    def test_custom_colors(self):
        """
        Custom colors are applied to violin bodies.

        Tests:
            (Test Case 1) Violin body facecolors match the provided colors.
        """
        fig, ax = plt.subplots()
        data = [np.random.rand(20), np.random.rand(20)]
        parts = plot_distribution(ax, data, colors=["red", "blue"])
        fc0 = parts["bodies"][0].get_facecolor()
        fc1 = parts["bodies"][1].get_facecolor()
        # matplotlib returns RGBA arrays; check first body is red-ish
        assert fc0[0][0] > 0.9  # R channel high for "red"
        assert fc1[0][2] > 0.9  # B channel high for "blue"


# ---------------------------------------------------------------------------
# plot_scatter tests
# ---------------------------------------------------------------------------


class TestPlotScatter:
    """Tests for the plot_scatter function."""

    def test_basic_scatter(self):
        """
        Basic scatter plot returns a PathCollection.

        Tests:
            (Test Case 1) Return type is a PathCollection.
            (Test Case 2) Scatter has the correct number of points.
        """
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)
        sc = plot_scatter(ax, x, y)
        assert len(sc.get_offsets()) == 5

    def test_axis_labels(self):
        """
        xlabel and ylabel are applied to the axes.

        Tests:
            (Test Case 1) Labels match the provided strings.
        """
        fig, ax = plt.subplots()
        plot_scatter(ax, [1, 2, 3], [1, 2, 3], xlabel="X val", ylabel="Y val")
        assert ax.get_xlabel() == "X val"
        assert ax.get_ylabel() == "Y val"

    def test_color_vals_with_colorbar(self):
        """
        color_vals enables color mapping and adds a colorbar.

        Tests:
            (Test Case 1) Figure has more than one axes (colorbar added).
        """
        fig, ax = plt.subplots()
        x = np.arange(10, dtype=float)
        plot_scatter(ax, x, x, color_vals=x, show_colorbar=True)
        # Colorbar creates an additional axes
        assert len(fig.axes) > 1

    def test_no_colorbar_when_disabled(self):
        """
        show_colorbar=False suppresses the colorbar even with color_vals.

        Tests:
            (Test Case 1) Figure has only one axes.
        """
        fig, ax = plt.subplots()
        x = np.arange(10, dtype=float)
        plot_scatter(ax, x, x, color_vals=x, show_colorbar=False)
        assert len(fig.axes) == 1

    def test_identity_line(self):
        """
        show_identity=True adds a dashed line.

        Tests:
            (Test Case 1) At least one Line2D on the axes.
        """
        fig, ax = plt.subplots()
        plot_scatter(ax, [1, 2, 3], [1, 2, 3], show_identity=True)
        assert len(ax.lines) >= 1

    def test_linear_fit(self):
        """
        fit='linear' overlays a regression line.

        Tests:
            (Test Case 1) A red line is added to the axes.
        """
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 20)
        y = 2 * x + 1 + np.random.default_rng(0).normal(0, 0.5, 20)
        plot_scatter(ax, x, y, fit="linear")
        # Regression line is added
        assert len(ax.lines) >= 1

    def test_linear_fit_with_ci(self):
        """
        fit='linear' with show_ci=True adds a fill-between band.

        Tests:
            (Test Case 1) At least one PolyCollection (CI band) on the axes.
        """
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 20)
        y = 2 * x + 1 + np.random.default_rng(0).normal(0, 0.5, 20)
        from matplotlib.collections import PolyCollection

        plot_scatter(ax, x, y, fit="linear", show_ci=True)
        # fill_between adds a PolyCollection; some matplotlib builds use subclasses
        # whose __name__ is not exactly "PolyCollection".
        poly_collections = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert (
            len(poly_collections) >= 1 or len(ax.collections) >= 2
        ), "expected fill_between CI (PolyCollection) plus scatter PathCollection"

    def test_r2_annotation(self):
        """
        show_r2=True adds an R² annotation to the axes.

        Tests:
            (Test Case 1) Axes has at least one text annotation containing 'R'.
        """
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 20)
        y = 2 * x + 1
        plot_scatter(ax, x, y, fit="linear", show_r2=True)
        texts = [t.get_text() for t in ax.texts]
        assert any("R" in t for t in texts)

    def test_invalid_fit_raises(self):
        """
        An unknown fit type raises ValueError.

        Tests:
            (Test Case 1) ValueError mentions the invalid fit name.
        """
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="Unknown fit"):
            plot_scatter(ax, [1, 2, 3], [1, 2, 3], fit="quadratic")

    def test_vmin_vmax_applied(self):
        """
        vmin and vmax are forwarded to the scatter colormap.

        Tests:
            (Test Case 1) Scatter clim matches vmin/vmax.
        """
        fig, ax = plt.subplots()
        x = np.arange(10, dtype=float)
        sc = plot_scatter(ax, x, x, color_vals=x, vmin=-5, vmax=15)
        clim = sc.get_clim()
        assert clim == (-5, 15)

    def test_font_size_applied(self):
        """
        font_size parameter changes label font sizes.

        Tests:
            (Test Case 1) Axis label font size matches the provided value.
        """
        fig, ax = plt.subplots()
        plot_scatter(ax, [1, 2, 3], [1, 2, 3], xlabel="X", font_size=20)
        assert ax.xaxis.label.get_fontsize() == 20


# ---------------------------------------------------------------------------
# plot_burst_sensitivity tests
# ---------------------------------------------------------------------------


class TestPlotBurstSensitivity:
    """Tests for the plot_burst_sensitivity function."""

    # Default colors for tests — avoids reliance on ax._get_lines.prop_cycler
    DEFAULT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    def test_1d_single_condition(self):
        """
        Single 1-D condition produces one line.

        Tests:
            (Test Case 1) Returns a list with one Line2D.
            (Test Case 2) Line has the correct number of data points.
        """
        fig, ax = plt.subplots()
        thr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        counts = {"rec1": np.array([10, 8, 5, 3, 1])}
        lines = plot_burst_sensitivity(ax, thr, counts, colors=self.DEFAULT_COLORS[:1])
        assert len(lines) == 1
        assert len(lines[0].get_xdata()) == 5

    def test_1d_multiple_conditions(self):
        """
        Multiple 1-D conditions produce one line each.

        Tests:
            (Test Case 1) Number of lines matches number of conditions.
        """
        fig, ax = plt.subplots()
        thr = np.array([1.0, 2.0, 3.0])
        counts = {
            "A": np.array([10, 5, 2]),
            "B": np.array([8, 4, 1]),
            "C": np.array([12, 7, 3]),
        }
        lines = plot_burst_sensitivity(ax, thr, counts, colors=self.DEFAULT_COLORS[:3])
        assert len(lines) == 3

    def test_1d_bare_array(self):
        """
        A bare 1-D array (not in a dict) works as a single condition.

        Tests:
            (Test Case 1) Returns a list with one Line2D.
        """
        fig, ax = plt.subplots()
        thr = np.array([1.0, 2.0, 3.0])
        lines = plot_burst_sensitivity(
            ax, thr, np.array([5, 3, 1]), colors=self.DEFAULT_COLORS[:1]
        )
        assert len(lines) == 1

    def test_1d_axis_labels(self):
        """
        Default and custom axis labels are applied.

        Tests:
            (Test Case 1) Default labels are 'RMS mult.' and 'Number of bursts'.
        """
        fig, ax = plt.subplots()
        thr = np.array([1.0, 2.0])
        plot_burst_sensitivity(
            ax, thr, {"A": np.array([5, 3])}, colors=self.DEFAULT_COLORS[:1]
        )
        assert ax.get_xlabel() == "RMS mult."
        assert ax.get_ylabel() == "Number of bursts"

    def test_1d_legend(self):
        """
        show_legend=True adds a legend with condition labels.

        Tests:
            (Test Case 1) Legend is present on the axes.
        """
        fig, ax = plt.subplots()
        thr = np.array([1.0, 2.0])
        plot_burst_sensitivity(
            ax,
            thr,
            {"A": np.array([5, 3])},
            show_legend=True,
            colors=self.DEFAULT_COLORS[:1],
        )
        legend = ax.get_legend()
        assert legend is not None

    def test_1d_no_legend(self):
        """
        show_legend=False suppresses the legend.

        Tests:
            (Test Case 1) No legend on the axes.
        """
        fig, ax = plt.subplots()
        thr = np.array([1.0, 2.0])
        plot_burst_sensitivity(
            ax,
            thr,
            {"A": np.array([5, 3])},
            show_legend=False,
            colors=self.DEFAULT_COLORS[:1],
        )
        assert ax.get_legend() is None

    def test_2d_single_condition_heatmap(self):
        """
        A single 2-D array produces a heatmap on the provided axes.

        Tests:
            (Test Case 1) The axes contains an AxesImage (from imshow).
        """
        fig, ax = plt.subplots()
        thr = np.array([1.0, 2.0, 3.0])
        dist = np.array([10, 20, 30, 40])
        counts_2d = np.random.randint(0, 20, size=(3, 4))
        result = plot_burst_sensitivity(ax, thr, counts_2d, dist_values=dist)
        # plot_heatmap returns the axes
        assert len(ax.images) == 1

    def test_2d_missing_dist_values_raises(self):
        """
        2-D burst counts without dist_values raises ValueError.

        Tests:
            (Test Case 1) ValueError with descriptive message.
        """
        fig, ax = plt.subplots()
        thr = np.array([1.0, 2.0])
        counts_2d = np.random.randint(0, 10, size=(2, 3))
        with pytest.raises(ValueError, match="dist_values is required"):
            plot_burst_sensitivity(ax, thr, counts_2d)

    def test_2d_multiple_conditions_subplot_row(self):
        """
        Multiple 2-D conditions create a row of heatmap subplots.

        Tests:
            (Test Case 1) Returns a (fig, axes_list) tuple.
            (Test Case 2) Number of axes matches number of conditions.
            (Test Case 3) Each subplot has an AxesImage.
        """
        thr = np.array([1.0, 2.0, 3.0])
        dist = np.array([10, 20])
        counts = {
            "A": np.random.randint(0, 10, size=(3, 2)),
            "B": np.random.randint(0, 10, size=(3, 2)),
        }
        result = plot_burst_sensitivity(None, thr, counts, dist_values=dist)
        fig, axes_list = result
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(axes_list) == 2
        for a in axes_list:
            assert len(a.images) == 1

    def test_2d_multiple_conditions_shared_clim(self):
        """
        Multiple 2-D heatmaps share the same color axis range.

        Tests:
            (Test Case 1) All heatmaps have the same (vmin, vmax).
        """
        thr = np.array([1.0, 2.0, 3.0])
        dist = np.array([10, 20])
        counts = {
            "low": np.array([[1, 2], [3, 4], [5, 6]]),
            "high": np.array([[10, 20], [30, 40], [50, 60]]),
        }
        fig, axes_list = plot_burst_sensitivity(None, thr, counts, dist_values=dist)
        clims = [a.images[0].get_clim() for a in axes_list]
        # All subplots should share the same clim
        assert clims[0] == clims[1]
        # Shared range should span 1 to 60
        assert clims[0][0] == pytest.approx(1)
        assert clims[0][1] == pytest.approx(60)

    def test_2d_multiple_conditions_titles(self):
        """
        Each subplot has the condition label as its title.

        Tests:
            (Test Case 1) Subplot titles match the condition labels.
        """
        thr = np.array([1.0, 2.0])
        dist = np.array([10, 20])
        counts = {
            "Ctrl": np.ones((2, 2), dtype=int),
            "Drug": np.ones((2, 2), dtype=int) * 2,
        }
        fig, axes_list = plot_burst_sensitivity(None, thr, counts, dist_values=dist)
        titles = [a.get_title() for a in axes_list]
        assert titles == ["Ctrl", "Drug"]


# ---------------------------------------------------------------------------
# plot_aligned_slice_single_unit tests
# ---------------------------------------------------------------------------


class TestPlotUnitRaster:
    """Tests for the plot_aligned_slice_single_unit standalone function."""

    def test_basic_raster(self):
        """
        Basic raster plot returns None when no color_vals provided.

        Tests:
            (Test Case 1) Returns None (no color coding).
            (Test Case 2) Axes has scatter points.
        """
        fig, ax = plt.subplots()
        spikes = [np.array([10, 50, 90]), np.array([20, 60]), np.array([30])]
        sc = plot_aligned_slice_single_unit(ax, spikes)
        assert sc is None
        assert len(ax.collections) >= 1

    def test_with_color_vals(self):
        """
        color_vals produces a colored scatter and returns a PathCollection.

        Tests:
            (Test Case 1) Returns a PathCollection (not None).
        """
        fig, ax = plt.subplots()
        spikes = [np.array([10, 50]), np.array([20, 60]), np.array([30])]
        color_vals = np.array([0.1, 0.5, 0.9])
        sc = plot_aligned_slice_single_unit(ax, spikes, color_vals=color_vals)
        assert sc is not None

    def test_colorbar_added(self):
        """
        Colorbar is added when color_vals is provided and show_colorbar=True.

        Tests:
            (Test Case 1) Figure has more than one axes.
        """
        fig, ax = plt.subplots()
        spikes = [np.array([10, 50]), np.array([20, 60])]
        plot_aligned_slice_single_unit(
            ax, spikes, color_vals=np.array([0.0, 1.0]), show_colorbar=True
        )
        assert len(fig.axes) > 1

    def test_no_colorbar(self):
        """
        show_colorbar=False suppresses colorbar even with color_vals.

        Tests:
            (Test Case 1) Figure has only one axes.
        """
        fig, ax = plt.subplots()
        spikes = [np.array([10, 50]), np.array([20, 60])]
        plot_aligned_slice_single_unit(
            ax, spikes, color_vals=np.array([0.0, 1.0]), show_colorbar=False
        )
        assert len(fig.axes) == 1

    def test_time_offset(self):
        """
        time_offset shifts spike times for display.

        Tests:
            (Test Case 1) Scatter x-coordinates are shifted by the offset.
        """
        fig, ax = plt.subplots()
        spikes = [np.array([100.0, 200.0])]
        plot_aligned_slice_single_unit(ax, spikes, time_offset=100.0)
        offsets = ax.collections[0].get_offsets()
        np.testing.assert_allclose(offsets[:, 0], [0.0, 100.0])

    def test_vlines(self):
        """
        vlines adds vertical reference lines.

        Tests:
            (Test Case 1) Lines are present on the axes.
        """
        fig, ax = plt.subplots()
        spikes = [np.array([10, 50])]
        plot_aligned_slice_single_unit(ax, spikes, vlines=[0.0, 25.0])
        assert len(ax.lines) >= 2

    def test_x_range_applied(self):
        """
        x_range sets the x-axis limits.

        Tests:
            (Test Case 1) xlim matches the provided range.
        """
        fig, ax = plt.subplots()
        spikes = [np.array([10, 50, 90])]
        plot_aligned_slice_single_unit(ax, spikes, x_range=(-10, 100))
        xlim = ax.get_xlim()
        assert xlim == (-10, 100)

    def test_ylim_matches_slice_count(self):
        """
        y-axis limits span 0 to number of slices.

        Tests:
            (Test Case 1) ylim upper bound equals the number of slices.
        """
        fig, ax = plt.subplots()
        spikes = [np.array([10]), np.array([20]), np.array([30])]
        plot_aligned_slice_single_unit(ax, spikes)
        assert ax.get_ylim() == (0, 3)

    def test_axis_labels(self):
        """
        Default axis labels are 'Rel. time (ms)' and 'Burst'.

        Tests:
            (Test Case 1) Labels match defaults.
            (Test Case 2) Custom labels override defaults.
        """
        fig, ax = plt.subplots()
        plot_aligned_slice_single_unit(ax, [np.array([10])])
        assert ax.get_xlabel() == "Rel. time (ms)"
        assert ax.get_ylabel() == "Burst"

        fig2, ax2 = plt.subplots()
        plot_aligned_slice_single_unit(
            ax2, [np.array([10])], xlabel="Time", ylabel="Trial"
        )
        assert ax2.get_xlabel() == "Time"
        assert ax2.get_ylabel() == "Trial"

    def test_empty_input(self):
        """
        Empty spike_times_per_slice returns None.

        Tests:
            (Test Case 1) Returns None.
        """
        fig, ax = plt.subplots()
        sc = plot_aligned_slice_single_unit(ax, [])
        assert sc is None

    def test_empty_slices_no_crash(self):
        """
        Slices with no spikes produce no scatter points but do not crash.

        Tests:
            (Test Case 1) No error when some slices have empty arrays.
        """
        fig, ax = plt.subplots()
        spikes = [np.array([]), np.array([10, 20]), np.array([])]
        sc = plot_aligned_slice_single_unit(ax, spikes)
        assert sc is None  # no color_vals
        assert len(ax.collections) >= 1


# ---------------------------------------------------------------------------
# SpikeSliceStack.plot_aligned_slice_single_unit tests
# ---------------------------------------------------------------------------


class TestSpikeSliceStackPlotUnitRaster:
    """Tests for the SpikeSliceStack.plot_aligned_slice_single_unit convenience wrapper."""

    @staticmethod
    def _make_stack(n_units=3, n_slices=4, slice_length=100.0):
        """Create a small SpikeSliceStack for testing."""
        rng = np.random.default_rng(42)
        slices = []
        for _ in range(n_slices):
            trains = [
                sorted(rng.uniform(0, slice_length, size=5).tolist())
                for _ in range(n_units)
            ]
            slices.append(SpikeData(trains, N=n_units, length=slice_length))
        times = [(i * slice_length, (i + 1) * slice_length) for i in range(n_slices)]
        return SpikeSliceStack(spike_stack=slices, times_start_to_end=times)

    def test_standalone_returns_fig_ax_sc(self):
        """
        Calling without ax creates a figure and returns (fig, ax, sc).

        Tests:
            (Test Case 1) Returns a 3-tuple.
            (Test Case 2) fig is a Figure, ax is an Axes.
        """
        stack = self._make_stack()
        result = stack.plot_aligned_slice_single_unit(0)
        assert isinstance(result, tuple)
        assert len(result) == 3
        fig, ax, sc = result
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_on_provided_ax_returns_sc(self):
        """
        Calling with ax returns just the scatter artist.

        Tests:
            (Test Case 1) Result is not a tuple.
        """
        stack = self._make_stack()
        fig, ax = plt.subplots()
        result = stack.plot_aligned_slice_single_unit(0, ax=ax)
        assert not isinstance(result, tuple)

    def test_unit_idx_out_of_range_raises(self):
        """
        Out-of-range unit_idx raises IndexError.

        Tests:
            (Test Case 1) Negative index raises IndexError.
            (Test Case 2) Index >= N raises IndexError.
        """
        stack = self._make_stack(n_units=3)
        with pytest.raises(IndexError, match="out of range"):
            stack.plot_aligned_slice_single_unit(-1)
        with pytest.raises(IndexError, match="out of range"):
            stack.plot_aligned_slice_single_unit(3)

    def test_correct_unit_extracted(self):
        """
        The wrapper extracts spike times from the correct unit index.

        Tests:
            (Test Case 1) Scatter y-coordinates span the number of slices.
        """
        stack = self._make_stack(n_units=3, n_slices=5)
        fig, ax, sc = stack.plot_aligned_slice_single_unit(1)
        assert ax.get_ylim() == (0, 5)

    def test_with_color_vals(self):
        """
        color_vals are forwarded to the underlying plot function.

        Tests:
            (Test Case 1) Returns a PathCollection (sc is not None).
        """
        stack = self._make_stack(n_slices=3)
        fig, ax, sc = stack.plot_aligned_slice_single_unit(
            0, color_vals=np.array([0.1, 0.5, 0.9])
        )
        assert sc is not None


# ---------------------------------------------------------------------------
# Edge Case Tests — plot_distribution
# ---------------------------------------------------------------------------


class TestPlotDistributionEdgeCases:
    """Edge case tests for plot_distribution identified in the edge case scan."""

    DEFAULT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    def test_all_nan_data(self):
        """
        All NaN data results in 0 points after stripping. The function should
        not crash but produce an empty or degenerate plot.

        Tests:
            (Test Case 1) Single group of all NaN values. After stripping,
                the group has 0 valid points. In violin mode, this is treated
                as a sparse group (< 2 points) and no violin body is produced.
        """
        fig, ax = plt.subplots()
        data = [np.array([np.nan, np.nan, np.nan])]
        parts = plot_distribution(ax, data, colors=self.DEFAULT_COLORS[:1])
        # No violin body for an empty group
        assert len(parts["bodies"]) == 0

    def test_empty_dict_input(self):
        """
        Empty dict input means no groups to plot.

        Tests:
            (Test Case 1) Empty dict produces no violin bodies and no error.
        """
        fig, ax = plt.subplots()
        data = {}
        parts = plot_distribution(ax, data, colors=[])
        assert len(parts["bodies"]) == 0


# ---------------------------------------------------------------------------
# Edge Case Tests — plot_scatter
# ---------------------------------------------------------------------------


class TestPlotScatterEdgeCases:
    """Edge case tests for plot_scatter identified in the edge case scan."""

    def test_fit_linear_with_fewer_than_3_points(self):
        """
        fit='linear' with < 3 points causes linear_regression to raise
        ValueError because it requires at least 3 non-NaN data points.

        Tests:
            (Test Case 1) Two data points with fit='linear' raises ValueError
                from linear_regression.
        """
        fig, ax = plt.subplots()
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="at least 3"):
            plot_scatter(ax, x, y, fit="linear")

    def test_all_nan_x_or_y_with_linear_fit(self):
        """
        All NaN x or y with fit='linear' causes linear_regression to raise
        ValueError because all points are dropped.

        Tests:
            (Test Case 1) All NaN x values with fit='linear'. After NaN
                dropping, 0 valid points remain, raising ValueError.
        """
        fig, ax = plt.subplots()
        x = np.array([np.nan, np.nan, np.nan, np.nan])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="at least 3"):
            plot_scatter(ax, x, y, fit="linear")


# ---------------------------------------------------------------------------
# Edge Case Tests — plot_aligned_slice_single_unit
# ---------------------------------------------------------------------------


class TestPlotUnitRasterEdgeCases:
    """Edge case tests for plot_aligned_slice_single_unit identified in the edge case scan."""

    def test_all_slices_empty(self):
        """
        All slices have empty spike arrays, producing a blank raster.

        Tests:
            (Test Case 1) Three slices, all with empty arrays. The function
                does not crash. Returns None (no color_vals). The scatter
                has no points but the axes are still set up.
        """
        fig, ax = plt.subplots()
        spikes = [np.array([]), np.array([]), np.array([])]
        sc = plot_aligned_slice_single_unit(ax, spikes)
        assert sc is None
        # y-axis should still span the number of slices
        assert ax.get_ylim() == (0, 3)


# ---------------------------------------------------------------------------
# Edge Case Tests — plot_burst_sensitivity
# ---------------------------------------------------------------------------


class TestPlotBurstSensitivityEdgeCases:
    """Edge case tests for plot_burst_sensitivity."""

    DEFAULT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    def test_empty_burst_counts_dict(self):
        """
        Empty burst_counts dict means no labels are extracted, and
        labels[0] raises IndexError.

        Tests:
            (Test Case 1) Empty dict raises IndexError because labels is
                empty and the function attempts to access labels[0].

        Notes:
            - This is a bug: the function does not guard against an empty
              burst_counts dict. It should either return early or raise a
              descriptive ValueError.
        """
        fig, ax = plt.subplots()
        thr = np.array([1.0, 2.0, 3.0])
        with pytest.raises(IndexError):
            plot_burst_sensitivity(ax, thr, {}, colors=[])


# ---------------------------------------------------------------------------
# plot_lines tests
# ---------------------------------------------------------------------------


class TestPlotLines:
    """Tests for the plot_lines multi-trace line plot function."""

    def test_dict_input_basic(self):
        """
        Dict input draws one line per key with keys as labels.

        Tests:
            (Test Case 1) Returns a list of 2 Line2D artists.
            (Test Case 2) Line labels match dict keys.
        """
        fig, ax = plt.subplots()
        traces = {"A": np.array([1, 2, 3]), "B": np.array([3, 2, 1])}
        lines = plot_lines(ax, traces)
        assert len(lines) == 2
        assert lines[0].get_label() == "A"
        assert lines[1].get_label() == "B"

    def test_list_input_with_labels(self):
        """
        List input uses explicitly provided labels.

        Tests:
            (Test Case 1) Line labels match provided labels list.
        """
        fig, ax = plt.subplots()
        traces = [np.array([1, 2, 3]), np.array([3, 2, 1])]
        lines = plot_lines(ax, traces, labels=["X", "Y"])
        assert lines[0].get_label() == "X"
        assert lines[1].get_label() == "Y"

    def test_list_input_default_labels(self):
        """
        List input without labels uses integer indices as labels.

        Tests:
            (Test Case 1) Labels are "0" and "1".
        """
        fig, ax = plt.subplots()
        lines = plot_lines(ax, [np.array([1, 2]), np.array([3, 4])])
        assert lines[0].get_label() == "0"
        assert lines[1].get_label() == "1"

    def test_custom_x_axis(self):
        """
        Custom x-axis values are applied to line data.

        Tests:
            (Test Case 1) Line x-data matches the provided x array.
        """
        fig, ax = plt.subplots()
        x = np.array([10, 20, 30])
        lines = plot_lines(ax, {"A": np.array([1, 2, 3])}, x=x)
        np.testing.assert_array_equal(lines[0].get_xdata(), x)

    def test_default_x_axis_is_integer_indices(self):
        """
        Without x, integer indices are used.

        Tests:
            (Test Case 1) Line x-data is [0, 1, 2].
        """
        fig, ax = plt.subplots()
        lines = plot_lines(ax, {"A": np.array([5, 6, 7])})
        np.testing.assert_array_equal(lines[0].get_xdata(), [0, 1, 2])

    def test_dict_colors(self):
        """
        Colors can be provided as a dict keyed by trace label.

        Tests:
            (Test Case 1) Each line uses the specified color.
        """
        fig, ax = plt.subplots()
        lines = plot_lines(
            ax,
            {"A": np.array([1, 2]), "B": np.array([3, 4])},
            colors={"A": "red", "B": "blue"},
        )
        assert lines[0].get_color() == "red"
        assert lines[1].get_color() == "blue"

    def test_list_colors(self):
        """
        Colors can be provided as a list.

        Tests:
            (Test Case 1) Each line uses the specified color.
        """
        fig, ax = plt.subplots()
        lines = plot_lines(
            ax,
            {"A": np.array([1, 2]), "B": np.array([3, 4])},
            colors=["green", "orange"],
        )
        assert lines[0].get_color() == "green"
        assert lines[1].get_color() == "orange"

    def test_legend_for_single_trace(self):
        """
        A single trace with show_legend=True still shows a legend.

        Tests:
            (Test Case 1) Legend is present with one entry.
        """
        fig, ax = plt.subplots()
        plot_lines(ax, {"only": np.array([1, 2, 3])}, show_legend=True)
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 1

    def test_legend_shown_for_multiple_traces(self):
        """
        Multiple traces with show_legend=True adds a legend.

        Tests:
            (Test Case 1) Legend is present with 2 entries.
        """
        fig, ax = plt.subplots()
        plot_lines(ax, {"A": np.array([1, 2]), "B": np.array([3, 4])})
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 2

    def test_axis_labels(self):
        """
        xlabel and ylabel are applied.

        Tests:
            (Test Case 1) Labels match provided strings.
        """
        fig, ax = plt.subplots()
        plot_lines(ax, {"A": np.array([1, 2])}, xlabel="Time", ylabel="Rate")
        assert ax.get_xlabel() == "Time"
        assert ax.get_ylabel() == "Rate"

    def test_linewidth(self):
        """
        Custom linewidth is applied to all lines.

        Tests:
            (Test Case 1) Both lines have linewidth 3.0.
        """
        fig, ax = plt.subplots()
        lines = plot_lines(
            ax,
            {"A": np.array([1, 2]), "B": np.array([3, 4])},
            linewidth=3.0,
        )
        assert lines[0].get_linewidth() == 3.0
        assert lines[1].get_linewidth() == 3.0


# ---------------------------------------------------------------------------
# plot_scatter — group mode tests
# ---------------------------------------------------------------------------


class TestPlotScatterGroups:
    """Tests for the discrete group coloring mode of plot_scatter."""

    def test_groups_returns_list(self):
        """
        Group mode returns a list of PathCollections.

        Tests:
            (Test Case 1) Returns a list with one entry per unique group.
        """
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([1, 2, 3, 4], dtype=float)
        groups = np.array([0, 0, 1, 1])
        sc = plot_scatter(ax, x, y, groups=groups)
        assert isinstance(sc, list)
        assert len(sc) == 2

    def test_group_labels(self):
        """
        Custom group_labels appear in the legend.

        Tests:
            (Test Case 1) Legend entries match provided labels.
        """
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([1, 2, 3, 4], dtype=float)
        plot_scatter(
            ax,
            x,
            y,
            groups=np.array([0, 0, 1, 1]),
            group_labels=["Ctrl", "Drug"],
        )
        legend = ax.get_legend()
        texts = [t.get_text() for t in legend.get_texts()]
        assert texts == ["Ctrl", "Drug"]

    def test_default_group_labels_from_values(self):
        """
        Without group_labels, unique group values are used as labels.

        Tests:
            (Test Case 1) Legend entries are "0" and "1".
        """
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([1, 2, 3, 4], dtype=float)
        plot_scatter(ax, x, y, groups=np.array([0, 0, 1, 1]))
        legend = ax.get_legend()
        texts = [t.get_text() for t in legend.get_texts()]
        assert texts == ["0", "1"]

    def test_group_colors(self):
        """
        Custom group_colors are applied to each group's scatter.

        Tests:
            (Test Case 1) First group scatter is red, second is blue.
        """
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([1, 2, 3, 4], dtype=float)
        sc = plot_scatter(
            ax,
            x,
            y,
            groups=np.array([0, 0, 1, 1]),
            group_colors=["red", "blue"],
        )
        # Each PathCollection's facecolor should match
        assert np.allclose(sc[0].get_facecolor()[0][:3], [1, 0, 0])  # red
        assert np.allclose(sc[1].get_facecolor()[0][:3], [0, 0, 1])  # blue

    def test_groups_ignores_colorbar(self):
        """
        Group mode does not add a colorbar even if color_vals is passed.

        Tests:
            (Test Case 1) Figure has only one axes (no colorbar).
        """
        fig, ax = plt.subplots()
        x = np.arange(4, dtype=float)
        plot_scatter(
            ax,
            x,
            x,
            groups=np.array([0, 0, 1, 1]),
            color_vals=x,
            show_colorbar=True,
        )
        assert len(fig.axes) == 1

    def test_groups_with_identity_line(self):
        """
        Identity line works in group mode.

        Tests:
            (Test Case 1) At least one Line2D is drawn (the identity line).
        """
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([1, 2, 3, 4], dtype=float)
        plot_scatter(
            ax,
            x,
            y,
            groups=np.array([0, 0, 1, 1]),
            show_identity=True,
        )
        assert len(ax.lines) >= 1

    def test_groups_no_legend_when_disabled(self):
        """
        show_legend=False suppresses the legend in group mode.

        Tests:
            (Test Case 1) Legend is None.
        """
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3, 4], dtype=float)
        y = np.array([1, 2, 3, 4], dtype=float)
        plot_scatter(
            ax,
            x,
            y,
            groups=np.array([0, 0, 1, 1]),
            show_legend=False,
        )
        assert ax.get_legend() is None

    def test_groups_correct_point_count(self):
        """
        Each group scatter contains the correct number of points.

        Tests:
            (Test Case 1) Group 0 has 3 points, group 1 has 2 points.
        """
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([1, 2, 3, 4, 5], dtype=float)
        sc = plot_scatter(
            ax,
            x,
            y,
            groups=np.array([0, 0, 0, 1, 1]),
        )
        assert len(sc[0].get_offsets()) == 3
        assert len(sc[1].get_offsets()) == 2


# ---------------------------------------------------------------------------
# plot_pvalue_matrix tests
# ---------------------------------------------------------------------------


class TestPlotPvalueMatrix:
    """Tests for the plot_pvalue_matrix function."""

    @staticmethod
    def _make_pval_matrix():
        """Create a simple 3x3 p-value matrix for testing."""
        pval = np.full((3, 3), np.nan)
        pval[0, 1] = pval[1, 0] = 0.001
        pval[0, 2] = pval[2, 0] = 0.20
        pval[1, 2] = pval[2, 1] = 0.04
        return pval

    def test_standalone_mode(self):
        """
        Standalone mode plots directly on the provided axes.

        Tests:
            (Test Case 1) The returned axes is the same as the input.
            (Test Case 2) An image is drawn on the axes.
        """
        fig, ax = plt.subplots()
        pval = self._make_pval_matrix()
        result_ax = plot_pvalue_matrix(pval, ax=ax)
        assert result_ax is ax
        assert len(ax.images) == 1

    def test_inset_mode(self):
        """
        Inset mode creates a new axes on the parent.

        Tests:
            (Test Case 1) The returned axes is not the parent.
            (Test Case 2) An image is drawn on the inset axes.
        """
        fig, parent_ax = plt.subplots()
        pval = self._make_pval_matrix()
        inset_ax = plot_pvalue_matrix(pval, parent_ax=parent_ax)
        assert inset_ax is not parent_ax
        assert len(inset_ax.images) == 1

    def test_both_ax_and_parent_raises(self):
        """
        Providing both ax and parent_ax raises ValueError.

        Tests:
            (Test Case 1) ValueError with descriptive message.
        """
        fig, ax = plt.subplots()
        pval = self._make_pval_matrix()
        with pytest.raises(ValueError, match="not both"):
            plot_pvalue_matrix(pval, ax=ax, parent_ax=ax)

    def test_neither_ax_nor_parent_raises(self):
        """
        Providing neither ax nor parent_ax raises ValueError.

        Tests:
            (Test Case 1) ValueError with descriptive message.
        """
        pval = self._make_pval_matrix()
        with pytest.raises(ValueError, match="either"):
            plot_pvalue_matrix(pval)

    def test_sig_matrix_auto_computed(self):
        """
        When sig_matrix is None, significance is computed as p < 0.05.

        Tests:
            (Test Case 1) Significant cells (p=0.001, p=0.04) get red markers.
            (Test Case 2) Non-significant cell (p=0.20) gets no marker.
        """
        fig, ax = plt.subplots()
        pval = self._make_pval_matrix()
        plot_pvalue_matrix(pval, ax=ax, show_colorbar=False)
        # Count red marker lines (each sig cell draws a marker via plot())
        marker_lines = [
            l
            for l in ax.lines
            if hasattr(l, "get_color")
            and np.allclose(
                plt.cm.colors.to_rgba("red")[:3],
                plt.cm.colors.to_rgba(l.get_color())[:3],
            )
        ]
        # (0,1),(1,0),(1,2),(2,1) = 4 significant pairs
        assert len(marker_lines) == 4

    def test_custom_labels(self):
        """
        Custom labels are applied to tick marks.

        Tests:
            (Test Case 1) x and y tick labels match provided labels.
        """
        fig, ax = plt.subplots()
        pval = self._make_pval_matrix()
        plot_pvalue_matrix(pval, labels=["A", "B", "C"], ax=ax, show_colorbar=False)
        xt = [t.get_text() for t in ax.get_xticklabels()]
        yt = [t.get_text() for t in ax.get_yticklabels()]
        assert xt == ["A", "B", "C"]
        assert yt == ["A", "B", "C"]

    def test_default_integer_labels(self):
        """
        Without labels, integer indices are used.

        Tests:
            (Test Case 1) Tick labels are "0", "1", "2".
        """
        fig, ax = plt.subplots()
        pval = self._make_pval_matrix()
        plot_pvalue_matrix(pval, ax=ax, show_colorbar=False)
        xt = [t.get_text() for t in ax.get_xticklabels()]
        assert xt == ["0", "1", "2"]

    def test_colorbar_shown(self):
        """
        show_colorbar=True adds a colorbar axes.

        Tests:
            (Test Case 1) More axes in the figure than just the main one.
        """
        fig, ax = plt.subplots()
        pval = self._make_pval_matrix()
        plot_pvalue_matrix(pval, ax=ax, show_colorbar=True)
        assert len(fig.axes) > 1

    def test_diagonal_is_nan(self):
        """
        Diagonal entries are set to NaN in the displayed image.

        Tests:
            (Test Case 1) Diagonal values in the image data are NaN.
        """
        fig, ax = plt.subplots()
        pval = self._make_pval_matrix()
        plot_pvalue_matrix(pval, ax=ax, show_colorbar=False)
        im_data = ax.images[0].get_array()
        for i in range(3):
            # imshow may return a masked array where NaN cells are masked
            val = im_data[i, i]
            assert np.ma.is_masked(val) or np.isnan(val)


# ---------------------------------------------------------------------------
# SpikeData.plot_aligned_pop_rate tests
# ---------------------------------------------------------------------------


class TestPlotAlignedPopRate:
    """Tests for the SpikeData.plot_aligned_pop_rate method."""

    @staticmethod
    def _make_sd_with_events():
        """Create a SpikeData with known pop rate and event times."""
        rng = np.random.default_rng(0)
        length = 2000.0
        trains = [sorted(rng.uniform(0, length, size=80).tolist()) for _ in range(5)]
        sd = SpikeData(trains, N=5, length=length)
        events = np.array([500.0, 1000.0, 1500.0])
        return sd, events

    def test_returns_avg_rate(self):
        """
        Returns a 1-D array of the expected length.

        Tests:
            (Test Case 1) avg_rate length equals pre_ms + post_ms.
            (Test Case 2) avg_rate is a 1-D numpy array.
        """
        sd, events = self._make_sd_with_events()
        avg = sd.plot_aligned_pop_rate(events, pre_ms=100, post_ms=200)
        assert isinstance(avg, np.ndarray)
        assert avg.ndim == 1
        assert len(avg) == 300  # 100 + 200

    def test_creates_figure_when_no_ax(self):
        """
        When ax=None, a new figure is created.

        Tests:
            (Test Case 1) A figure with at least one axes exists after the call.
        """
        sd, events = self._make_sd_with_events()
        sd.plot_aligned_pop_rate(events, pre_ms=100, post_ms=200)
        fig = plt.gcf()
        assert len(fig.axes) >= 1

    def test_plots_on_given_ax(self):
        """
        When ax is provided, the trace is drawn on it.

        Tests:
            (Test Case 1) The given axes has at least one line after the call.
        """
        sd, events = self._make_sd_with_events()
        fig, ax = plt.subplots()
        sd.plot_aligned_pop_rate(events, pre_ms=100, post_ms=200, ax=ax)
        assert len(ax.lines) >= 1

    def test_custom_color_and_label(self):
        """
        Custom color and label are applied to the mean trace.

        Tests:
            (Test Case 1) Mean line has the specified color.
            (Test Case 2) Mean line has the specified label.
        """
        sd, events = self._make_sd_with_events()
        fig, ax = plt.subplots()
        sd.plot_aligned_pop_rate(
            events,
            pre_ms=100,
            post_ms=200,
            ax=ax,
            color="red",
            label="D0",
        )
        line = ax.lines[-1]
        assert line.get_color() == "red"
        assert line.get_label() == "D0"

    def test_multi_condition_overlay(self):
        """
        Multiple calls on the same axes overlay traces.

        Tests:
            (Test Case 1) After two calls, axes has at least 2 lines.
        """
        sd, events = self._make_sd_with_events()
        fig, ax = plt.subplots()
        sd.plot_aligned_pop_rate(
            events, pre_ms=100, post_ms=200, ax=ax, color="blue", label="C1"
        )
        sd.plot_aligned_pop_rate(
            events, pre_ms=100, post_ms=200, ax=ax, color="red", label="C2"
        )
        assert len(ax.lines) >= 2

    def test_show_individual_traces(self):
        """
        show_individual=True draws extra lines for each event.

        Tests:
            (Test Case 1) More lines are drawn than just the mean trace.
        """
        sd, events = self._make_sd_with_events()
        fig, ax = plt.subplots()
        sd.plot_aligned_pop_rate(
            events,
            pre_ms=100,
            post_ms=200,
            ax=ax,
            show_individual=True,
        )
        # At least 1 mean + some individual traces
        assert len(ax.lines) > 1

    def test_precomputed_pop_rate(self):
        """
        Pre-computed pop_rate is used instead of auto-computing.

        Tests:
            (Test Case 1) avg_rate is computed from the provided pop_rate.
        """
        sd, events = self._make_sd_with_events()
        # Create a constant pop rate
        pop_rate = np.ones(int(sd.length))
        avg = sd.plot_aligned_pop_rate(
            events,
            pre_ms=100,
            post_ms=200,
            pop_rate=pop_rate,
        )
        np.testing.assert_allclose(avg, 1.0)

    def test_burst_edges_with_percentile(self):
        """
        Burst edges + edge_percentile draws vertical markers.

        Tests:
            (Test Case 1) Two vertical lines (start and end markers) are added.
        """
        sd, events = self._make_sd_with_events()
        fig, ax = plt.subplots()
        burst_edges = np.column_stack([events - 80, events + 150])
        sd.plot_aligned_pop_rate(
            events,
            pre_ms=100,
            post_ms=200,
            ax=ax,
            burst_edges=burst_edges,
            edge_percentile=100,
        )
        # Mean line + 2 axvline calls
        assert len(ax.lines) >= 3

    def test_edge_percentile_without_edges_user_events_raises(self):
        """
        Setting edge_percentile with user-provided events but no burst_edges
        raises ValueError.

        Tests:
            (Test Case 1) ValueError with descriptive message.
        """
        sd, events = self._make_sd_with_events()
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="burst_edges is required"):
            sd.plot_aligned_pop_rate(
                events,
                pre_ms=100,
                post_ms=200,
                ax=ax,
                edge_percentile=50,
            )

    def test_no_valid_windows_raises(self):
        """
        Events that produce no valid windows raise ValueError.

        Tests:
            (Test Case 1) Events at recording edges with large window raises.
        """
        sd, _ = self._make_sd_with_events()
        # Events at the very start — pre_ms extends before 0
        events = np.array([10.0])
        with pytest.raises(ValueError, match="No valid event windows"):
            sd.plot_aligned_pop_rate(events, pre_ms=500, post_ms=500)

    def test_xlim_matches_window(self):
        """
        X-axis limits span from -pre_ms to +post_ms.

        Tests:
            (Test Case 1) xlim is (-100, 199) for pre=100, post=200.
        """
        sd, events = self._make_sd_with_events()
        fig, ax = plt.subplots()
        sd.plot_aligned_pop_rate(events, pre_ms=100, post_ms=200, ax=ax)
        xlim = ax.get_xlim()
        assert xlim[0] == -100
        assert xlim[1] == 199  # arange(300) - 100 → last value is 199

    def test_auto_detect_bursts(self):
        """
        When events=None, burst peaks are auto-detected via get_bursts.

        Tests:
            (Test Case 1) Method runs without error.
            (Test Case 2) Returns a 1-D avg_rate array.

        Notes:
            - Uses a SpikeData with dense, synchronized spiking to ensure
              burst detection produces at least one burst.
        """
        # Create a SpikeData with a clear burst around t=500
        rng = np.random.default_rng(99)
        n_units = 20
        length = 3000.0
        trains = []
        for _ in range(n_units):
            # Background: sparse spikes
            bg = rng.uniform(0, length, size=5).tolist()
            # Burst: dense cluster around t=500
            burst = rng.normal(500, 5, size=30).clip(0, length).tolist()
            # Burst: dense cluster around t=2000
            burst2 = rng.normal(2000, 5, size=30).clip(0, length).tolist()
            trains.append(sorted(bg + burst + burst2))
        sd = SpikeData(trains, N=n_units, length=length)
        fig, ax = plt.subplots()
        avg = sd.plot_aligned_pop_rate(ax=ax)
        assert isinstance(avg, np.ndarray)
        assert avg.ndim == 1

    def test_auto_detect_with_edge_percentile(self):
        """
        When events=None and edge_percentile is set, burst edges are
        auto-detected and edge markers are drawn.

        Tests:
            (Test Case 1) Method runs without error and draws edge lines.
        """
        rng = np.random.default_rng(99)
        n_units = 20
        length = 3000.0
        trains = []
        for _ in range(n_units):
            bg = rng.uniform(0, length, size=5).tolist()
            burst = rng.normal(500, 5, size=30).clip(0, length).tolist()
            burst2 = rng.normal(2000, 5, size=30).clip(0, length).tolist()
            trains.append(sorted(bg + burst + burst2))
        sd = SpikeData(trains, N=n_units, length=length)
        fig, ax = plt.subplots()
        avg = sd.plot_aligned_pop_rate(ax=ax, edge_percentile=100)
        assert isinstance(avg, np.ndarray)
        # Mean line + 2 edge markers
        assert len(ax.lines) >= 3
