import numpy as np

from napari_ooctyle_analysis._widget import OoctyleAnalysisWidget


def test_ooctyle_analysis_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = OoctyleAnalysisWidget(viewer)

    assert widget.viewer is viewer


def test_widget_with_image(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))

    widget = OoctyleAnalysisWidget(viewer)
    assert widget.viewer is viewer
    assert len(viewer.layers) == 1
