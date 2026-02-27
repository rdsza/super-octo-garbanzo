import sys
import numpy as np
import mrcfile
import napari
import concurrent.futures
import os
import time
# Optional pyFFTW acceleration (best-effort)
try:
    import pyfftw
    from pyfftw.interfaces import numpy_fft as pw_fft
    try:
        pw_fft.cache.enable()
    except Exception:
        pass
    PYFFTW_AVAILABLE = True
    PYFFTW_THREADS = min(4, (os.cpu_count() or 1))
except Exception:
    PYFFTW_AVAILABLE = False
    pw_fft = None
    PYFFTW_THREADS = 1
import threading
from qtpy.QtWidgets import (
    QApplication, QFileDialog, QPushButton, QWidget, QVBoxLayout,
    QLabel, QRadioButton, QHBoxLayout, QMessageBox, QCheckBox, QSpinBox,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QProgressBar
)
from qtpy.QtCore import Qt, QTimer


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle('MRC Viewer')
    layout = QHBoxLayout()
    window.setLayout(layout)

    # left controls panel
    controls_widget = QWidget()
    controls_widget.setMaximumWidth(360)
    controls_layout = QVBoxLayout()
    controls_widget.setLayout(controls_layout)

    # file group (open + filename label)
    file_group = QGroupBox('File')
    file_layout = QVBoxLayout()
    file_group.setLayout(file_layout)
    open_button = QPushButton('Open MRC File')
    file_layout.addWidget(open_button)
    label = QLabel('No file selected')
    label.setAlignment(Qt.AlignCenter)
    file_layout.addWidget(label)
    controls_layout.addWidget(file_group)

    # load options group (mmap, downsample)
    load_group = QGroupBox('Load Options')
    load_layout = QFormLayout()
    load_group.setLayout(load_layout)
    mmap_checkbox = QCheckBox('Keep file open (memory-map friendly)')
    downsample_spin = QSpinBox()
    downsample_spin.setRange(1, 8)
    downsample_spin.setValue(1)
    load_layout.addRow(mmap_checkbox)
    load_layout.addRow('Downsample:', downsample_spin)
    controls_layout.addWidget(load_group)

    # view mode group
    view_group = QGroupBox('View Mode')
    view_layout = QVBoxLayout()
    view_group.setLayout(view_layout)
    slices_radio = QRadioButton('Slices')
    slices_radio.setChecked(True)
    projections_radio = QRadioButton('Projections')
    view_layout.addWidget(slices_radio)
    view_layout.addWidget(projections_radio)
    mode_label = QLabel('Mode: N/A')
    mode_label.setAlignment(Qt.AlignCenter)
    view_layout.addWidget(mode_label)
    controls_layout.addWidget(view_group)

    # analysis group (FFT)
    analysis_group = QGroupBox('Analysis')
    analysis_layout = QVBoxLayout()
    analysis_group.setLayout(analysis_layout)
    fft_button = QPushButton('Perform FFT')
    fft_button.setEnabled(False)
    analysis_layout.addWidget(fft_button)
    # filter controls
    filter_type = QComboBox()
    filter_type.addItems(['None', 'Gaussian Low-pass', 'Gaussian High-pass', 'Band-pass', 'Wiener'])
    analysis_layout.addWidget(filter_type)

    # sigma controls in pixels (spatial domain)
    cutoff_low = QDoubleSpinBox()
    cutoff_low.setRange(0.1, 100.0)
    cutoff_low.setValue(1.0)
    cutoff_low.setSingleStep(0.5)
    cutoff_high = QDoubleSpinBox()
    cutoff_high.setRange(0.1, 100.0)
    cutoff_high.setValue(3.0)
    cutoff_high.setSingleStep(0.5)
    cutoff_layout = QHBoxLayout()
    cutoff_layout.addWidget(QLabel('Sigma low (px):'))
    cutoff_layout.addWidget(cutoff_low)
    cutoff_layout.addWidget(QLabel('Sigma high (px):'))
    cutoff_layout.addWidget(cutoff_high)
    analysis_layout.addLayout(cutoff_layout)

    apply_filter_btn = QPushButton('Apply Filter')
    analysis_layout.addWidget(apply_filter_btn)

    reset_filter_btn = QPushButton('Reset Filter')
    analysis_layout.addWidget(reset_filter_btn)

    filter_progress = QProgressBar()
    filter_progress.setRange(0, 100)
    filter_progress.setValue(0)
    analysis_layout.addWidget(filter_progress)
    # live preview
    live_preview_cb = QCheckBox('Live preview (current slice)')
    live_preview_cb.setChecked(False)
    analysis_layout.addWidget(live_preview_cb)
    # FFT progress and cancel
    fft_progress = QProgressBar()
    fft_progress.setRange(0, 100)
    fft_progress.setValue(0)
    analysis_layout.addWidget(fft_progress)

    fft_cancel_btn = QPushButton('Cancel FFT')
    fft_cancel_btn.setEnabled(False)
    analysis_layout.addWidget(fft_cancel_btn)
    # pyFFTW options
    use_pyfftw_cb = QCheckBox('Use pyFFTW if available')
    use_pyfftw_cb.setChecked(PYFFTW_AVAILABLE)
    analysis_layout.addWidget(use_pyfftw_cb)

    pyfftw_threads_spin = QSpinBox()
    max_cpu = max(1, (os.cpu_count() or 1))
    pyfftw_threads_spin.setRange(1, max_cpu)
    pyfftw_threads_spin.setValue(PYFFTW_THREADS)
    threads_layout = QHBoxLayout()
    threads_layout.addWidget(QLabel('pyFFTW threads:'))
    threads_layout.addWidget(pyfftw_threads_spin)
    analysis_layout.addLayout(threads_layout)
    # pyFFTW status and benchmark
    pyfftw_status_label = QLabel('pyFFTW: available' if PYFFTW_AVAILABLE else 'pyFFTW: not available')
    analysis_layout.addWidget(pyfftw_status_label)

    fft_bench_btn = QPushButton('Benchmark FFT')
    analysis_layout.addWidget(fft_bench_btn)
    controls_layout.addWidget(analysis_group)

    # annotations group (ROIs, points)
    annot_group = QGroupBox('Annotations')
    annot_layout = QVBoxLayout()
    annot_group.setLayout(annot_layout)
    add_rect_btn = QPushButton('Add Rectangle ROI')
    add_ellipse_btn = QPushButton('Add Ellipse ROI')
    add_points_btn = QPushButton('Add Points')
    export_roi_btn = QPushButton('Export ROI to MRC')
    clear_ann_btn = QPushButton('Clear Annotations')
    annot_layout.addWidget(add_rect_btn)
    annot_layout.addWidget(add_ellipse_btn)
    annot_layout.addWidget(add_points_btn)
    annot_layout.addWidget(export_roi_btn)
    annot_layout.addWidget(clear_ann_btn)
    controls_layout.addWidget(annot_group)

    # right viewer container
    viewer_container = QWidget()
    viewer_container.setMinimumSize(600, 400)
    viewer_container_layout = QVBoxLayout()
    viewer_container.setLayout(viewer_container_layout)

    # add panels to main layout (controls left, viewer right)
    layout.addWidget(controls_widget)
    layout.addWidget(viewer_container)

    # area where the napari viewer widget will be embedded
    viewer = None
    viewer_container = QWidget()
    viewer_container.setMinimumSize(600, 400)
    viewer_container_layout = QVBoxLayout()
    viewer_container.setLayout(viewer_container_layout)
    layout.addWidget(viewer_container)
    original_data = None
    fft_layer = None  # Store the FFT layer
    fft_future = None
    filter_future = None
    preview_future = None
    preview_timer = QTimer()
    preview_timer.setSingleShot(True)
    preview_timer.setInterval(300)  # ms debounce
    # Use multiple workers to allow parallel per-slice FFTs and responsive UI
    max_workers = min(8, (os.cpu_count() or 1))
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    fft_cancel_event = None
    fft_progress_state = {'value': 0}
    current_mrc = None
    current_file = None
    shapes_layer = None
    points_layer = None

    def on_open_button_clicked():
        nonlocal viewer, original_data, fft_layer, current_mrc, current_file
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            window, "Open MRC File", "", "MRC Files (*.mrc *.mrcs);;All Files (*)", options=options)
        if not file_name:
            return

        label.setText(f'Selected File: {file_name}')
        current_file = file_name

        try:
            # close any previously opened MRC if present
            if current_mrc is not None:
                try:
                    current_mrc.close()
                except Exception:
                    pass
                current_mrc = None

            downsample_factor = int(downsample_spin.value())
            use_mmap = mmap_checkbox.isChecked()

            if use_mmap:
                # keep file open so data may be backed by mmap and avoid a large copy
                mrc = mrcfile.open(file_name, permissive=True)
                data = mrc.data
                current_mrc = mrc
            else:
                with mrcfile.open(file_name, permissive=True) as mrc:
                    data = np.array(mrc.data, copy=True)

            if data is None or getattr(data, 'size', 0) == 0:
                raise ValueError('MRC file contains no data')
            if getattr(data, 'ndim', None) not in (2, 3):
                raise ValueError(f'Unsupported data ndim: {getattr(data, "ndim", None)}')

            # apply simple downsampling (spatial only)
            if downsample_factor > 1:
                f = downsample_factor
                if data.ndim == 3:
                    data = data[:, ::f, ::f]
                else:
                    data = data[::f, ::f]

            # avoid unnecessary copy when memory-mapping requested
            if use_mmap:
                original_data = data
            else:
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
            # create a napari Viewer (try to keep it from showing a separate window)
            try:
                viewer = napari.Viewer(show=False)
            except TypeError:
                # older/newer napari may not accept show kwarg
                viewer = napari.Viewer()
            try:
                qtwidget = viewer.window.qt_viewer
                viewer_container_layout.addWidget(qtwidget)
            except Exception:
                # fallback: still usable but not embedded
                pass
        else:
            # remove existing image layers (but keep other UI elements)
            for layer in list(viewer.layers):
                if isinstance(layer, napari.layers.Image):
                    viewer.layers.remove(layer)
            fft_layer = None

        data = original_data
        # Determine whether this is a true 3D volume (cubic NxNxN)
        is_volume = False
        if getattr(data, 'ndim', None) == 3:
            s0, s1, s2 = data.shape
            if s0 == s1 == s2:
                is_volume = True

        if data.ndim == 2:
            # simple 2D micrograph
            viewer.add_image(data, name='Original')
            try:
                mode_label.setText('Mode: 2D micrograph')
            except Exception:
                pass
        elif data.ndim == 3 and is_volume:
            # true 3D cubic volume
            if projections_radio.isChecked():
                projections = generate_projections(data)
                viewer.add_image(projections, name='Projections')
                try:
                    mode_label.setText('Mode: Volume (projections)')
                except Exception:
                    pass
            else:
                # show as 3D volume/stack
                viewer.add_image(data, name='Original')
                try:
                    mode_label.setText('Mode: 3D Volume')
                except Exception:
                    pass
        elif data.ndim == 3 and not is_volume:
            # non-cubic stack (treat as 2D collection) — create a montage for 2D inspection
            montage = generate_projections(data)
            viewer.add_image(montage, name='Original')
            try:
                mode_label.setText('Mode: 2D stack (montage)')
            except Exception:
                pass
        else:
            label.setText('Error: MRC file must be 2D or 3D.')
            return

        # ensure annotation layers exist
        nonlocal shapes_layer, points_layer
        if shapes_layer is None:
            try:
                shapes_layer = viewer.add_shapes(name='ROIs')
                shapes_layer.mode = 'pan_zoom'
            except Exception:
                shapes_layer = None
        if points_layer is None:
            try:
                points_layer = viewer.add_points(name='Points')
                points_layer.mode = 'pan_zoom'
            except Exception:
                points_layer = None
        # connect viewer dims change to live preview scheduler (only once)
        try:
            if not getattr(viewer, '_preview_connected', False):
                try:
                    viewer.dims.events.current_step.connect(lambda e: schedule_preview())
                except Exception:
                    # fallback: older napari API
                    try:
                        viewer.dims.events.connect(lambda e: schedule_preview())
                    except Exception:
                        pass
                viewer._preview_connected = True
        except Exception:
            pass

    def generate_projections(volume):
        """
        Create a montage grid of all projections in the volume (n, m, m).
        Arranges n projections of size (m, m) into a 2D grid.
        """
        # handle non-cubic slices
        if volume.ndim != 3:
            raise ValueError('Projections require a 3D volume')
        n, h, w = volume.shape

        # Calculate grid dimensions (roughly square)
        grid_cols = int(np.ceil(np.sqrt(n)))
        grid_rows = int(np.ceil(n / grid_cols))

        # Create montage array
        montage = np.zeros((grid_rows * h, grid_cols * w), dtype=volume.dtype)

        # Fill montage with projections
        for idx in range(n):
            row = idx // grid_cols
            col = idx % grid_cols
            montage[row * h:(row + 1) * h, col * w:(col + 1) * w] = volume[idx]

        return montage


    def perform_fft():
        nonlocal viewer, fft_layer, original_data, fft_future, fft_cancel_event, fft_progress_state
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

        def compute_fft(data, cancel_event, progress_state):
            # compute magnitude spectrum, log-scale, and normalize to float32
            if getattr(data, 'ndim', 2) == 3:
                zdim = data.shape[0]
                out = np.empty_like(data, dtype=np.float32)

                # define per-slice worker
                def _fft_slice(slice_idx):
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError('FFT cancelled')
                    arr = data[slice_idx]
                    # choose FFT implementation (respect UI toggle)
                    use_py = False
                    try:
                        use_py = use_pyfftw_cb.isChecked() and PYFFTW_AVAILABLE and pw_fft is not None
                    except Exception:
                        use_py = PYFFTW_AVAILABLE and pw_fft is not None
                    if use_py:
                        threads = PYFFTW_THREADS
                        try:
                            threads = int(pyfftw_threads_spin.value())
                        except Exception:
                            pass
                        try:
                            fft_raw = pw_fft.fft2(arr, threads=threads)
                        except TypeError:
                            fft_raw = pw_fft.fft2(arr)
                    else:
                        fft_raw = np.fft.fft2(arr)
                    fft_result = np.fft.fftshift(fft_raw)
                    mag = np.abs(fft_result)
                    mag = np.log1p(mag).astype(np.float32)
                    # normalize per-slice
                    if mag.max() > 0:
                        mag = (mag - mag.min()) / (mag.max() - mag.min())
                    return slice_idx, mag

                # use a local thread pool to parallelize slices (better than sequential loop)
                workers = min(zdim, max(1, (os.cpu_count() or 1)))
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(_fft_slice, z): z for z in range(zdim)}
                    completed = 0
                    for fut in concurrent.futures.as_completed(futures):
                        if cancel_event is not None and cancel_event.is_set():
                            progress_state['value'] = 0
                            raise RuntimeError('FFT cancelled')
                        idx, mag = fut.result()
                        out[idx] = mag
                        completed += 1
                        progress_state['value'] = int(completed / zdim * 100)
                return out
            else:
                if cancel_event is not None and cancel_event.is_set():
                    progress_state['value'] = 0
                    raise RuntimeError('FFT cancelled')
                # single 2D case (respect UI toggle for pyFFTW)
                use_py = False
                try:
                    use_py = use_pyfftw_cb.isChecked() and PYFFTW_AVAILABLE and pw_fft is not None
                except Exception:
                    use_py = PYFFTW_AVAILABLE and pw_fft is not None
                if use_py:
                    threads = PYFFTW_THREADS
                    try:
                        threads = int(pyfftw_threads_spin.value())
                    except Exception:
                        pass
                    try:
                        fft_raw = pw_fft.fft2(data, threads=threads)
                    except TypeError:
                        fft_raw = pw_fft.fft2(data)
                else:
                    fft_raw = np.fft.fft2(data)
                fft_result = np.fft.fftshift(fft_raw)
                mag = np.abs(fft_result)
                mag = np.log1p(mag).astype(np.float32)
                if mag.max() > 0:
                    mag = (mag - mag.min()) / (mag.max() - mag.min())
                progress_state['value'] = 100
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

        # prepare cancel event and progress state
        fft_cancel_event = threading.Event()
        fft_progress_state['value'] = 0

        fft_button.setEnabled(False)
        fft_cancel_btn.setEnabled(True)
        label.setText('Computing FFT...')
        fft_future = executor.submit(compute_fft, image_data, fft_cancel_event, fft_progress_state)

        # poll progress via QTimer
        progress_timer = QTimer()
        progress_timer.setInterval(100)

        def on_poll():
            try:
                val = int(fft_progress_state.get('value', 0))
            except Exception:
                val = 0
            fft_progress.setValue(val)
            if fft_future.done():
                progress_timer.stop()
        progress_timer.timeout.connect(on_poll)
        progress_timer.start()

        def _callback(fut):
            nonlocal fft_cancel_event
            try:
                res = fut.result()
            except Exception as e:
                # check for cancellation
                if isinstance(e, RuntimeError) and 'cancel' in str(e).lower():
                    QTimer.singleShot(0, lambda: label.setText('FFT cancelled'))
                else:
                    QTimer.singleShot(0, lambda: label.setText(f'FFT error: {e}'))
                QTimer.singleShot(0, lambda: fft_button.setEnabled(True))
                QTimer.singleShot(0, lambda: fft_cancel_btn.setEnabled(False))
                QTimer.singleShot(0, lambda: fft_progress.setValue(0))
                fft_cancel_event = None
                return
            QTimer.singleShot(0, lambda: on_fft_done(res))
            QTimer.singleShot(0, lambda: fft_button.setEnabled(True))
            QTimer.singleShot(0, lambda: fft_cancel_btn.setEnabled(False))
            QTimer.singleShot(0, lambda: label.setText(f'Selected File: {current_file}'))
            QTimer.singleShot(0, lambda: fft_progress.setValue(100))
            fft_cancel_event = None

        fft_future.add_done_callback(_callback)

        def cancel_fft():
            nonlocal fft_cancel_event
            if fft_cancel_event is not None:
                fft_cancel_event.set()
                fft_cancel_btn.setEnabled(False)

        fft_cancel_btn.clicked.connect(cancel_fft)

    def benchmark_fft():
        # run a short benchmark in background to pick best pyFFTW thread count
        nonlocal pyfftw_status_label
        if not PYFFTW_AVAILABLE:
            QMessageBox.information(window, 'pyFFTW not installed', 'pyFFTW is not available in this environment.')
            return

        fft_bench_btn.setEnabled(False)
        pyfftw_status_label.setText('Benchmarking...')

        def _bench():
            # choose test array: use a representative slice or synthetic
            if original_data is not None:
                if original_data.ndim == 3:
                    arr = original_data[original_data.shape[0] // 2]
                else:
                    arr = original_data
            else:
                arr = np.random.randn(512, 512).astype(np.float32)

            # downsample large arrays for faster benchmarking
            max_dim = 1024
            if arr.shape[0] > max_dim or arr.shape[1] > max_dim:
                arr = arr[::max(1, arr.shape[0] // max_dim), ::max(1, arr.shape[1] // max_dim)]

            max_threads = max(1, (os.cpu_count() or 1))
            trials = 3
            timings = {}
            for t in range(1, max_threads + 1):
                # warm-up
                try:
                    _ = pw_fft.fft2(arr, threads=t)
                except Exception:
                    try:
                        _ = pw_fft.fft2(arr)
                    except Exception:
                        return None
                # timed trials
                ts = []
                for _ in range(trials):
                    t0 = time.perf_counter()
                    try:
                        fft_raw = pw_fft.fft2(arr, threads=t)
                    except TypeError:
                        fft_raw = pw_fft.fft2(arr)
                    _ = np.fft.fftshift(fft_raw)
                    _ = np.abs(_)
                    _ = np.log1p(_)
                    ts.append(time.perf_counter() - t0)
                timings[t] = float(np.mean(ts))
            # pick best (lowest time)
            best_t = min(timings, key=timings.get)
            return best_t, timings

        future = executor.submit(_bench)

        def _done(fut):
            fft_bench_btn.setEnabled(True)
            try:
                res = fut.result()
            except Exception as e:
                pyfftw_status_label.setText('pyFFTW: benchmark failed')
                QMessageBox.warning(window, 'Benchmark error', str(e))
                return
            if res is None:
                pyfftw_status_label.setText('pyFFTW: not usable')
                QMessageBox.information(window, 'Benchmark', 'pyFFTW benchmark could not be completed.')
                return
            best_t, timings = res
            try:
                pyfftw_threads_spin.setValue(int(best_t))
                use_pyfftw_cb.setChecked(True)
            except Exception:
                pass
            pyfftw_status_label.setText(f'pyFFTW: available (best {best_t} threads)')
            # show summary
            msg = 'FFT benchmark results (avg seconds):\n' + '\n'.join([f'{k} threads: {v:.4f}s' for k, v in sorted(timings.items())])
            QMessageBox.information(window, 'FFT benchmark', msg)

        future.add_done_callback(lambda f: QTimer.singleShot(0, lambda: _done(f)))

    fft_bench_btn.clicked.connect(benchmark_fft)

    open_button.clicked.connect(on_open_button_clicked)
    slices_radio.toggled.connect(update_viewer)
    projections_radio.toggled.connect(update_viewer)
    fft_button.clicked.connect(perform_fft)
    
    def apply_filter():
        nonlocal viewer, filter_future
        if viewer is None:
            return

        # find image layer
        current_layer = None
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Image) and layer.name in ('Original', 'Projections'):
                current_layer = layer
                break
        if current_layer is None:
            for layer in viewer.layers:
                if isinstance(layer, napari.layers.Image):
                    current_layer = layer
                    break
        if current_layer is None:
            return

        image = np.array(current_layer.data)
        ftype = filter_type.currentText()
        low = float(cutoff_low.value())
        high = float(cutoff_high.value())

        def compute_filter(img, ftype, low, high):
            # operate per-slice if 3D
            def _filter2d(im):
                # use spatial-domain filters with sigma in pixels
                m, n = im.shape
                if ftype == 'Gaussian Low-pass':
                    from filters import lowpass_spatial

                    return lowpass_spatial(im, sigma_px=low)
                elif ftype == 'Gaussian High-pass':
                    from filters import highpass_spatial

                    return highpass_spatial(im, sigma_px=low)
                elif ftype == 'Band-pass' or ftype == 'DoG Band-pass':
                    from filters import dog_spatial

                    return dog_spatial(im, sigma_low=low, sigma_high=high)
                elif ftype == 'Wiener':
                    from filters import wiener_filter

                    return wiener_filter(im)
                else:
                    return im

            if img.ndim == 3:
                out = np.empty_like(img, dtype=np.float32)
                for z in range(img.shape[0]):
                    out[z] = _filter2d(img[z]).astype(np.float32)
                return out
            else:
                return _filter2d(img).astype(np.float32)

        # submit compute
        apply_filter_btn.setEnabled(False)
        filter_progress.setValue(0)
        filter_future = executor.submit(compute_filter, image, ftype, low, high)

        def _done(fut):
            try:
                res = fut.result()
            except Exception as e:
                QTimer.singleShot(0, lambda: QMessageBox.critical(window, 'Filter error', str(e)))
                QTimer.singleShot(0, lambda: apply_filter_btn.setEnabled(True))
                return
            def _update():
                nonlocal viewer
                try:
                    # find or create the Filtered layer
                    existing = None
                    for ly in viewer.layers:
                        if isinstance(ly, napari.layers.Image) and ly.name == 'Filtered':
                            existing = ly
                            break
                    if existing is None:
                        # set sensible display kwargs for 2D micrographs
                        try:
                            clim = (float(np.nanmin(res)), float(np.nanmax(res)))
                        except Exception:
                            clim = None
                        try:
                            if getattr(res, 'ndim', 2) == 2:
                                new_layer = viewer.add_image(res, name='Filtered', colormap='gray')
                            else:
                                new_layer = viewer.add_image(res, name='Filtered')
                            if clim is not None:
                                try:
                                    new_layer.contrast_limits = clim
                                except Exception:
                                    pass
                            try:
                                new_layer.opacity = 1.0
                            except Exception:
                                pass
                        except Exception:
                            # fallback simple add
                            new_layer = viewer.add_image(res, name='Filtered')
                    else:
                        existing.data = res
                        new_layer = existing
                        try:
                            if getattr(res, 'ndim', 2) == 2:
                                existing.colormap = 'gray'
                                existing.opacity = 1.0
                                existing.contrast_limits = (float(np.nanmin(res)), float(np.nanmax(res)))
                        except Exception:
                            pass

                    # Ensure visibility: show Filtered and hide other image layers
                    try:
                        for ly in viewer.layers:
                            if isinstance(ly, napari.layers.Image):
                                if ly is new_layer:
                                    ly.visible = True
                                else:
                                    try:
                                        ly.visible = False
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                    # mark progress and re-enable button
                    apply_filter_btn.setEnabled(True)
                    filter_progress.setValue(100)
                except Exception as e:
                    QMessageBox.critical(window, 'Update error', f'Failed to update Filtered layer: {e}')
                    apply_filter_btn.setEnabled(True)
                    filter_progress.setValue(0)
            QTimer.singleShot(0, _update)

        filter_future.add_done_callback(_done)

    apply_filter_btn.clicked.connect(apply_filter)

    def reset_filter():
        nonlocal viewer
        if viewer is None:
            return
        for ly in list(viewer.layers):
            if isinstance(ly, napari.layers.Image) and ly.name == 'Filtered':
                viewer.layers.remove(ly)
        # also remove preview overlay and reset controls
        for ly in list(viewer.layers):
            if isinstance(ly, napari.layers.Image) and ly.name == 'Preview':
                try:
                    viewer.layers.remove(ly)
                except Exception:
                    pass
        filter_progress.setValue(0)
        apply_filter_btn.setEnabled(True)

    reset_filter_btn.clicked.connect(reset_filter)

    # --- Live preview support ---
    def compute_preview_slice(im2d, ftype, low, high):
        # Direct 2D filtering using filters functions
        if ftype == 'Gaussian Low-pass':
            from filters import lowpass_spatial

            return lowpass_spatial(im2d, sigma_px=low)
        elif ftype == 'Gaussian High-pass':
            from filters import highpass_spatial

            return highpass_spatial(im2d, sigma_px=low)
        elif ftype == 'Band-pass' or ftype == 'DoG Band-pass':
            from filters import dog_spatial

            return dog_spatial(im2d, sigma_low=low, sigma_high=high)
        elif ftype == 'Wiener':
            from filters import wiener_filter

            return wiener_filter(im2d)
        else:
            return im2d

    def schedule_preview():
        # debounce and schedule
        if not live_preview_cb.isChecked():
            return
        preview_timer.start()

    def run_preview():
        nonlocal preview_future
        if viewer is None or not live_preview_cb.isChecked():
            return

        # find current image layer
        current_layer = None
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Image) and layer.name in ('Original', 'Filtered', 'Projections'):
                current_layer = layer
                break
        if current_layer is None:
            return

        image_data = np.array(current_layer.data)
        # pick current slice if 3D
        if getattr(image_data, 'ndim', 2) == 3:
            try:
                z = int(viewer.dims.current_step[0])
            except Exception:
                z = 0
            slice2d = image_data[z]
        else:
            slice2d = image_data

        ftype = filter_type.currentText()
        low = float(cutoff_low.value())
        high = float(cutoff_high.value())

        # cancel previous preview if running (we can't kill it, but we'll ignore its result)
        preview_future = executor.submit(compute_preview_slice, slice2d, ftype, low, high)

        def _done(fut):
            try:
                res = fut.result()
            except Exception:
                return
            def _update():
                # update or add a Preview layer for the current slice
                existing = None
                for ly in viewer.layers:
                    if isinstance(ly, napari.layers.Image) and ly.name == 'Preview':
                        existing = ly
                        break
                if existing is None:
                    viewer.add_image(res, name='Preview', opacity=0.7)
                else:
                    existing.data = res
                    existing.opacity = 0.7
            QTimer.singleShot(0, _update)

        preview_future.add_done_callback(_done)

    # wire live-preview controls
    def on_live_preview_toggled(enabled):
        if not enabled:
            # remove Preview layer when disabling
            if viewer is not None:
                for ly in list(viewer.layers):
                    if isinstance(ly, napari.layers.Image) and ly.name == 'Preview':
                        try:
                            viewer.layers.remove(ly)
                        except Exception:
                            pass
        else:
            # schedule an immediate preview when enabling
            schedule_preview()

    live_preview_cb.toggled.connect(on_live_preview_toggled)
    filter_type.currentIndexChanged.connect(lambda _: schedule_preview())
    cutoff_low.valueChanged.connect(lambda _: schedule_preview())
    cutoff_high.valueChanged.connect(lambda _: schedule_preview())
    preview_timer.timeout.connect(run_preview)

    # --- Annotation handlers ---
    def enable_add_rectangle():
        nonlocal shapes_layer
        if viewer is None:
            return
        if shapes_layer is None:
            shapes_layer = viewer.add_shapes(name='ROIs')
        try:
            shapes_layer.shape_type = 'rectangle'
        except Exception:
            pass
        try:
            shapes_layer.mode = 'add'
        except Exception:
            # napari mode API varies by version; instruct user to enable add mode via the layer toolbar
            QMessageBox.information(window, 'Draw ROI', 'To add shapes: select the ROIs layer and use the shapes toolbar to enter add mode.')

    def enable_add_ellipse():
        nonlocal shapes_layer
        if viewer is None:
            return
        if shapes_layer is None:
            shapes_layer = viewer.add_shapes(name='ROIs')
        try:
            shapes_layer.shape_type = 'ellipse'
        except Exception:
            pass
        try:
            shapes_layer.mode = 'add'
        except Exception:
            QMessageBox.information(window, 'Draw ROI', 'To add shapes: select the ROIs layer and use the shapes toolbar to enter add mode.')

    def enable_add_points():
        nonlocal points_layer
        if viewer is None:
            return
        if points_layer is None:
            points_layer = viewer.add_points(name='Points')
        try:
            points_layer.mode = 'add'
        except Exception:
            QMessageBox.information(window, 'Add Points', 'To add points: select the Points layer and use the points toolbar to enter add mode.')

    def clear_annotations():
        nonlocal shapes_layer, points_layer
        if shapes_layer is not None:
            try:
                viewer.layers.remove(shapes_layer)
            except Exception:
                pass
            shapes_layer = None
        if points_layer is not None:
            try:
                viewer.layers.remove(points_layer)
            except Exception:
                pass
            points_layer = None

    def export_roi_to_mrc():
        nonlocal shapes_layer, original_data, current_file, current_mrc
        if shapes_layer is None or len(shapes_layer.data) == 0:
            QMessageBox.warning(window, 'No ROI', 'No ROI found to export')
            return
        # use first ROI
        coords = np.asarray(shapes_layer.data[0])
        # coords are in (row, col) order
        ys = coords[:, 0]
        xs = coords[:, 1]
        y0, y1 = int(np.floor(ys.min())), int(np.ceil(ys.max()))
        x0, x1 = int(np.floor(xs.min())), int(np.ceil(xs.max()))

        if original_data is None:
            QMessageBox.warning(window, 'No data', 'No image data loaded')
            return

        # If 3D volume, crop all Z along spatial dims
        if original_data.ndim == 3:
            cropped = original_data[:, y0:y1, x0:x1]
        else:
            cropped = original_data[y0:y1, x0:x1]

        # determine original dtype to preserve
        orig_dtype = None
        if current_mrc is not None:
            try:
                orig_dtype = current_mrc.data.dtype
            except Exception:
                orig_dtype = None
        if orig_dtype is None:
            orig_dtype = original_data.dtype if original_data is not None else cropped.dtype

        # Ask user where to save
        save_path, _ = QFileDialog.getSaveFileName(window, 'Save cropped MRC', current_file or '', 'MRC Files (*.mrc);;All Files (*)')
        if not save_path:
            return
        try:
            # Use mrcfile.new to create a clean MRC and write data with original dtype
            with mrcfile.new(save_path, overwrite=True) as mrc:
                try:
                    mrc.set_data(cropped.astype(orig_dtype))
                except Exception:
                    # fallback to float32 if dtype not supported
                    mrc.set_data(cropped.astype(np.float32))

                # Try to copy header/metadata where possible
                if current_mrc is not None:
                    try:
                        # best-effort copy of header
                        hdr = None
                        try:
                            hdr = current_mrc.header.copy()
                        except Exception:
                            hdr = None
                        if hdr is not None:
                            # adjust dimensional fields if present
                            try:
                                if cropped.ndim == 3:
                                    hdr.nz, hdr.ny, hdr.nx = cropped.shape
                                else:
                                    hdr.nz, hdr.ny, hdr.nx = 1, cropped.shape[0], cropped.shape[1]
                            except Exception:
                                pass
                            try:
                                mrc.header = hdr
                            except Exception:
                                # header assignment may not be supported; ignore safely
                                pass
                    except Exception:
                        pass

            QMessageBox.information(window, 'Saved', f'Saved cropped ROI to {save_path}')
        except Exception as e:
            QMessageBox.critical(window, 'Save error', str(e))

    add_rect_btn.clicked.connect(enable_add_rectangle)
    add_ellipse_btn.clicked.connect(enable_add_ellipse)
    add_points_btn.clicked.connect(enable_add_points)
    clear_ann_btn.clicked.connect(clear_annotations)
    export_roi_btn.clicked.connect(export_roi_to_mrc)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
