from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel


class OoctyleAnalysisWidget(QWidget):
    """Primary widget for oocyte analysis in napari."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("OoctyleAnalysis Plugin"))
