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
    # Create the Qt Application
    app = QApplication(sys.argv)

    # Create the main window widget
    window = QWidget()
    window.setWindowTitle('MRC Viewer')
    layout = QVBoxLayout()
    window.setLayout(layout)

    # Create a label to show the selected file
    label = QLabel('No file selected')
    label.setAlignment(Qt.AlignCenter)
    layout.addWidget(label)

    # Create a horizontal layout for buttons and options
    button_layout = QHBoxLayout()

    # Create a button to open the file dialog
    open_button = QPushButton('Open MRC File')
    button_layout.addWidget(open_button)

    # Create radio buttons for viewing mode
    mode_label = QLabel('View Mode:')
    button_layout.addWidget(mode_label)

    slices_radio = QRadioButton('Slices')
    slices_radio.setChecked(True)  # Default mode
    button_layout.addWidget(slices_radio)

    projections_radio = QRadioButton('Projections')
    button_layout.addWidget(projections_radio)

    layout.addLayout(button_layout)

    # Create a button to perform FFT
    fft_button = QPushButton('Perform FFT')
    fft_button.setEnabled(False)  # Disabled until an image is loaded
    layout.addWidget(fft_button)

    # Create a variable to hold the viewer
    viewer = None
    # Variable to store the original image data
    original_data = None
    # Variable to track if FFT is applied
    fft_applied = False

    # Define the open file button click event
    def on_open_button_clicked():
        nonlocal viewer, original_data, fft_applied
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            window, "Open MRC File", "", "MRC Files (*.mrc *.mrcs);;All Files (*)", options=options)
        if file_name:
            label.setText(f'Selected File: {file_name}')
            # Load the MRC file
            try:
                with mrcfile.open(file_name, permissive=True) as mrc:
                    data = mrc.data.astype(np.float32)
                original_data = data  # Store original data
                fft_applied = False
                fft_button.setEnabled(True)
                update_viewer()
            except Exception as e:
                label.setText(f'Error loading MRC file: {e}')
                fft_button.setEnabled(False)

    # Define the function to update the viewer based on the selected mode
    def update_viewer():
        nonlocal viewer, original_data, fft_applied
        if viewer is not None:
            # Close existing viewer
            viewer.window.close()
        data = original_data
        if data.ndim == 2:
            # 2D image
            viewer = napari.view_image(data, title='MRC Viewer')
        elif data.ndim == 3:
            if projections_radio.isChecked():
                # Generate projections
                projections = generate_projections(data)
                viewer = napari.view_image(projections, name='Projections', title='MRC Viewer')
            else:
                # View slices
                viewer = napari.view_image(data, ndisplay=2, name='Slices', title='MRC Viewer')
        else:
            label.setText('Error: MRC file must be 2D or 3D.')
            return

        # Connect viewer events
        viewer.dims.events.current_step.connect(on_slice_change)
        viewer.window.qt_viewer.setWindowTitle('MRC Viewer')
        fft_applied = False  # Reset FFT applied flag

    # Function to generate projections from the 3D volume
    def generate_projections(volume):
        # For simplicity, we'll create cumulative projections along the Z-axis
        projections = []
        for i in range(1, volume.shape[0] + 1):
            projection = np.sum(volume[:i, :, :], axis=0)
            projections.append(projection)
        projections = np.array(projections)
        return projections

    # Function to perform FFT on the current displayed image
    def perform_fft():
        nonlocal viewer, fft_applied
        if viewer is None:
            return
        # Get the current displayed image
        layer = viewer.layers[0]
        image_data = layer.data
        # Perform FFT on the current slice
        if image_data.ndim == 3:
            # Get current frame
            current_frame = viewer.dims.current_step[0]
            image = image_data[current_frame]
        else:
            image = image_data

        # Compute FFT
        fft_result = np.fft.fftshift(np.fft.fft2(image))
        magnitude_spectrum = np.abs(fft_result)
        # Update the layer data
        if image_data.ndim == 3:
            image_data[current_frame] = magnitude_spectrum
            layer.data = image_data
        else:
            layer.data = magnitude_spectrum
        fft_applied = True

    # Function to handle slice changes (to reset FFT if necessary)
    def on_slice_change(event):
        nonlocal fft_applied
        if fft_applied:
            # Reset to original data when changing slices
            update_viewer()

    # Connect the open button click event
    open_button.clicked.connect(on_open_button_clicked)

    # Connect the viewing mode radio buttons
    slices_radio.toggled.connect(update_viewer)
    projections_radio.toggled.connect(update_viewer)

    # Connect the FFT button click event
    fft_button.clicked.connect(perform_fft)

    # Show the window
    window.show()

    # Start the Qt event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
