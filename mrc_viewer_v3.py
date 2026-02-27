import sys
import numpy as np
import mrcfile
import napari
import concurrent.futures
from qtpy.QtWidgets import (
    QApplication, QFileDialog, QPushButton, QWidget, QVBoxLayout,
    QLabel, QRadioButton, QHBoxLayout, QMessageBox
)
from qtpy.QtCore import Qt, QTimer

def main():
    app = QApplication.instance() or QApplication(sys.argv)
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
    fft_future = None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def on_open_button_clicked():
        nonlocal viewer, original_data, fft_layer
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            window, "Open MRC File", "", "MRC Files (*.mrc *.mrcs);;All Files (*)", options=options)
        if file_name:
            label.setText(f'Selected File: {file_name}')
            try:
                # load data (keep as float32 for computations)
                with mrcfile.open(file_name, permissive=True) as mrc:
                    data = np.array(mrc.data, copy=True)
                if data is None or data.size == 0:
                    raise ValueError('MRC file contains no data')
                if data.ndim not in (2, 3):
                    raise ValueError(f'Unsupported data ndim: {data.ndim}')
                original_data = data.astype(np.float32)
                fft_button.setEnabled(True)
                update_viewer()
            except Exception as e:
                fft_button.setEnabled(False)
                msg = QMessageBox(window)
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle('Error loading MRC')
                msg.setText('Failed to load MRC file')
                msg.setInformativeText(str(e))
                msg.exec_()

    def update_viewer():
        nonlocal viewer, original_data, fft_layer

        # Don't update if no data loaded yet
        if original_data is None:
            return
        # Create viewer if needed, otherwise reuse existing one and clear image layers
        if viewer is None:
            viewer = napari.Viewer(title='MRC Viewer')
        else:
            # remove existing image layers (but keep other UI elements)
            for layer in list(viewer.layers):
                if isinstance(layer, napari.layers.Image):
                    viewer.layers.remove(layer)
            fft_layer = None

        data = original_data
        if data.ndim == 2:
            viewer.add_image(data, name='Original')
        elif data.ndim == 3:
            if projections_radio.isChecked():
                projections = generate_projections(data)
                viewer.add_image(projections, name='Projections')
            else:
                viewer.add_image(data, ndisplay=2, name='Original')
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
        nonlocal viewer, fft_layer, original_data, fft_future
        if viewer is None:
            return

        # find an image layer to operate on (prefer named 'Original')
        current_layer = None
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Image) and layer.name in ('Original', 'Slices', 'Projections'):
                current_layer = layer
                break
        if current_layer is None:
            # fallback to first image layer
            for layer in viewer.layers:
                if isinstance(layer, napari.layers.Image):
                    current_layer = layer
                    break
        if current_layer is None:
            return

        image_data = np.array(current_layer.data)

        def compute_fft(data):
            # compute magnitude spectrum, log-scale, and normalize to float32
            if data.ndim == 3:
                out = np.empty_like(data, dtype=np.float32)
                for z in range(data.shape[0]):
                    fft_result = np.fft.fftshift(np.fft.fft2(data[z]))
                    mag = np.abs(fft_result)
                    mag = np.log1p(mag)
                    # normalize per-slice
                    mag = mag.astype(np.float32)
                    if mag.max() > 0:
                        mag = (mag - mag.min()) / (mag.max() - mag.min())
                    out[z] = mag
                return out
            else:
                fft_result = np.fft.fftshift(np.fft.fft2(data))
                mag = np.abs(fft_result)
                mag = np.log1p(mag)
                mag = mag.astype(np.float32)
                if mag.max() > 0:
                    mag = (mag - mag.min()) / (mag.max() - mag.min())
                return mag

        def on_fft_done(result):
            nonlocal fft_layer
            # update viewer on main thread
            if fft_layer is None:
                fft_layer = viewer.add_image(result, name='FFT', visible=True)
                current_layer.visible = False
            else:
                fft_layer.data = result
                fft_layer.visible = True
                current_layer.visible = False

        # if a previous compute is running, ignore or wait
        if fft_future is not None and not fft_future.done():
            return

        fft_button.setEnabled(False)
        label.setText('Computing FFT...')
        fft_future = executor.submit(compute_fft, image_data)

        def _callback(fut):
            try:
                res = fut.result()
            except Exception as e:
                QTimer.singleShot(0, lambda: label.setText(f'FFT error: {e}'))
                QTimer.singleShot(0, lambda: fft_button.setEnabled(True))
                return
            QTimer.singleShot(0, lambda: on_fft_done(res))
            QTimer.singleShot(0, lambda: fft_button.setEnabled(True))
            QTimer.singleShot(0, lambda: label.setText(f'Selected File: {file_name}'))

        fft_future.add_done_callback(_callback)

    open_button.clicked.connect(on_open_button_clicked)
    slices_radio.toggled.connect(update_viewer)
    projections_radio.toggled.connect(update_viewer)
    fft_button.clicked.connect(perform_fft)

    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()