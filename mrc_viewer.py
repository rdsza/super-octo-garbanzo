import sys
import numpy as np
import mrcfile
import napari
from qtpy.QtWidgets import QApplication, QFileDialog, QPushButton, QWidget, QVBoxLayout, QLabel
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

    # Create a button to open the file dialog
    button = QPushButton('Open MRC File')
    layout.addWidget(button)

    # Create a variable to hold the viewer
    viewer = None

    # Define the button click event
    def on_button_clicked():
        nonlocal viewer
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(window, "Open MRC File", "", "MRC Files (*.mrc *.mrcs);;All Files (*)", options=options)
        if file_name:
            label.setText(f'Selected File: {file_name}')
            # Load the MRC file
            try:
                with mrcfile.open(file_name, permissive=True) as mrc:
                    data = mrc.data.astype(np.float32)
                # Check if data is 2D or 3D
                if data.ndim == 2:
                    # 2D image
                    if viewer is None:
                        viewer = napari.view_image(data, title='MRC Viewer')
                    else:
                        viewer.add_image(data)
                elif data.ndim == 3:
                    # 3D volume
                    if viewer is None:
                        viewer = napari.view_image(data, ndisplay=3, title='MRC Viewer')
                    else:
                        viewer.add_image(data)
                else:
                    label.setText('Error: MRC file must be 2D or 3D.')
            except Exception as e:
                label.setText(f'Error loading MRC file: {e}')

    # Connect the button click event
    button.clicked.connect(on_button_clicked)

    # Show the window
    window.show()

    # Start the Qt event loop
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
