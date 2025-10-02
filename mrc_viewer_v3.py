import sys
import numpy as np
import mrcfile
import napari
from qtpy.QtWidgets import (
    QApplication, QFileDialog, QPushButton, QWidget, QVBoxLayout,
    QLabel, QRadioButton, QHBoxLayout
)
from qtpy.QtCore import Qt

def main():
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle('MRC Viewer')
    layout = QVBoxLayout()
    window.setLayout(layout)

    label = QLabel('No file selected')
    label.setAlignment(Qt.AlignCenter)
    layout.addWidget(label)

    button_layout = QHBoxLayout()
    open_button = QPushButton('Open MRC File')
    button_layout.addWidget(open_button)

    mode_label = QLabel('View Mode:')
    button_layout.addWidget(mode_label)

    slices_radio = QRadioButton('Slices')
    slices_radio.setChecked(True)
    button_layout.addWidget(slices_radio)

    projections_radio = QRadioButton('Projections')
    button_layout.addWidget(projections_radio)

    layout.addLayout(button_layout)

    fft_button = QPushButton('Perform FFT')
    fft_button.setEnabled(False)
    layout.addWidget(fft_button)

    viewer = None
    original_data = None
    fft_layer = None  # Store the FFT layer

    def on_open_button_clicked():
        nonlocal viewer, original_data, fft_layer
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            window, "Open MRC File", "", "MRC Files (*.mrc *.mrcs);;All Files (*)", options=options)
        if file_name:
            label.setText(f'Selected File: {file_name}')
            try:
                with mrcfile.open(file_name, permissive=True) as mrc:
                    data = mrc.data.astype(np.float32)
                original_data = data
                fft_button.setEnabled(True)
                update_viewer()
            except Exception as e:
                label.setText(f'Error loading MRC file: {e}')
                fft_button.setEnabled(False)

    def update_viewer():
        nonlocal viewer, original_data, fft_layer

        # Don't update if no data loaded yet
        if original_data is None:
            return

        if viewer is not None:
            try:
                viewer.close()  # Close existing viewer
            except RuntimeError:
                pass  # Already deleted
            viewer = None # Important: Reset the viewer to None
            fft_layer = None

        data = original_data
        if data.ndim == 2:
            viewer = napari.Viewer(title='MRC Viewer')
            viewer.add_image(data)
        elif data.ndim == 3:
            if projections_radio.isChecked():
                projections = generate_projections(data)
                viewer = napari.Viewer(title='MRC Viewer')
                viewer.add_image(projections, name='Projections')
            else:
                viewer = napari.Viewer(title='MRC Viewer')
                viewer.add_image(data, ndisplay=2, name='Slices')
        else:
            label.setText('Error: MRC file must be 2D or 3D.')
            return

    def generate_projections(volume):
        """
        Create a montage grid of all projections in the volume (n, m, m).
        Arranges n projections of size (m, m) into a 2D grid.
        """
        n, m, _ = volume.shape

        # Calculate grid dimensions (roughly square)
        grid_cols = int(np.ceil(np.sqrt(n)))
        grid_rows = int(np.ceil(n / grid_cols))

        # Create montage array
        montage = np.zeros((grid_rows * m, grid_cols * m), dtype=volume.dtype)

        # Fill montage with projections
        for idx in range(n):
            row = idx // grid_cols
            col = idx % grid_cols
            montage[row*m:(row+1)*m, col*m:(col+1)*m] = volume[idx]

        return montage


    def perform_fft():
        nonlocal viewer, fft_layer, original_data
        if viewer is None:
            return

        current_layer = viewer.layers[0]  # Get the current image layer
        image_data = current_layer.data

        if fft_layer is None:  # Create FFT layer if it doesn't exist
            if image_data.ndim == 3:
                fft_data = np.zeros_like(image_data, dtype=np.float32) # Initialize with zeros
            else:
                fft_data = np.zeros_like(image_data, dtype=np.float32)
            fft_layer = napari.layers.Image(fft_data, name='FFT', visible=False) # Start as hidden
            viewer.add_layer(fft_layer)

        # Calculate and update FFT data. Handle 2D and 3D
        if image_data.ndim == 3:
            for z in range(image_data.shape[0]):
                fft_result = np.fft.fftshift(np.fft.fft2(image_data[z]))
                magnitude_spectrum = np.abs(fft_result)
                fft_layer.data[z] = magnitude_spectrum
        else:
            fft_result = np.fft.fftshift(np.fft.fft2(image_data))
            magnitude_spectrum = np.abs(fft_result)
            fft_layer.data = magnitude_spectrum

        fft_layer.visible = not fft_layer.visible  # Toggle visibility
        current_layer.visible = not current_layer.visible # Toggle visibility of original layer

    open_button.clicked.connect(on_open_button_clicked)
    slices_radio.toggled.connect(update_viewer)
    projections_radio.toggled.connect(update_viewer)
    fft_button.clicked.connect(perform_fft)

    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()