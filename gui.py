import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from regressor_analysis import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "Regression Analysis"

        self.left   = 10
        self.top    = 10
        self.width  = 400
        self.height = 800

        self.checkboxes = []

        self.reset_variables()

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # create main widget & layout
        self.main_widget = QWidget(self)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(0)

        # set main widget to be the central widget
        self.setCentralWidget(self.main_widget)

        widget = QWidget()
        self.main_layout.addWidget(widget)
        layout = QHBoxLayout(widget)

        self.load_calcium_video_button = QPushButton("Load Calcium Video(s)...")
        self.load_calcium_video_button.setFixedWidth(180)
        self.load_calcium_video_button.clicked.connect(self.load_calcium_video)
        layout.addWidget(self.load_calcium_video_button)

        layout.addStretch()

        self.load_roi_data_button = QPushButton("Load ROI Data File(s)...")
        self.load_roi_data_button.setFixedWidth(180)
        self.load_roi_data_button.clicked.connect(self.load_roi_data)
        self.load_roi_data_button.setEnabled(False)
        layout.addWidget(self.load_roi_data_button)

        layout.addStretch()

        self.load_bouts_button = QPushButton("Load Bout File(s)...")
        self.load_bouts_button.setFixedWidth(150)
        self.load_bouts_button.clicked.connect(self.load_bouts)
        self.load_bouts_button.setEnabled(False)
        layout.addWidget(self.load_bouts_button)

        layout.addStretch()

        self.load_frame_timestamps_button = QPushButton("Load Timestamp File(s)...")
        self.load_frame_timestamps_button.setFixedWidth(190)
        self.load_frame_timestamps_button.clicked.connect(self.load_frame_timestamps)
        self.load_frame_timestamps_button.setEnabled(False)
        layout.addWidget(self.load_frame_timestamps_button)

        layout.addStretch()

        self.calcium_video_loaded_label = QLabel("0 Loaded.")
        # layout.addWidget(self.calcium_video_loaded_label)

        self.top_widget = QWidget()
        self.top_layout = QHBoxLayout(self.top_widget)
        self.main_layout.addWidget(self.top_widget)

        left_plot_groupbox = QGroupBox("Regressor Correlations")
        self.top_layout.addWidget(left_plot_groupbox)
        left_plot_layout = QVBoxLayout(left_plot_groupbox)

        self.left_plot_canvas = PlotCanvas(self, width=4, height=8)
        self.left_plot_canvas.setFixedHeight(600)
        self.left_plot_canvas.setFixedWidth(300)
        left_plot_layout.addWidget(self.left_plot_canvas)
        left_plot_layout.setAlignment(self.left_plot_canvas, Qt.AlignCenter)

        widget = QWidget()
        left_plot_layout.addWidget(widget)
        layout = QHBoxLayout(widget)

        label = QLabel("Regressor: ")
        layout.addWidget(label)

        self.regressor_index_slider = QSlider(Qt.Horizontal)
        self.regressor_index_slider.setRange(0, self.n_regressors)
        self.regressor_index_slider.setValue(self.regressor_index)
        self.regressor_index_slider.valueChanged.connect(self.set_regressor_index)
        layout.addWidget(self.regressor_index_slider)

        self.regressor_index_label = QLabel(str(self.regressor_index))
        layout.addWidget(self.regressor_index_label)

        widget = QWidget()
        left_plot_layout.addWidget(widget)
        layout = QHBoxLayout(widget)

        label = QLabel("Max p-value: ")
        layout.addWidget(label)

        self.max_p_slider = QSlider(Qt.Horizontal)
        self.max_p_slider.setRange(1, 50)
        self.max_p_slider.setValue(int(100*self.max_p))
        self.max_p_slider.valueChanged.connect(self.set_max_p)
        layout.addWidget(self.max_p_slider)

        self.max_p_label = QLabel(str(self.max_p))
        layout.addWidget(self.max_p_label)

        right_plot_groupbox = QGroupBox("Multi-Regressor Analysis")
        self.top_layout.addWidget(right_plot_groupbox)
        right_plot_layout = QVBoxLayout(right_plot_groupbox)

        self.right_plot_canvas = PlotCanvas(self, width=4, height=8)
        self.right_plot_canvas.setFixedHeight(600)
        self.right_plot_canvas.setFixedWidth(300)
        right_plot_layout.addWidget(self.right_plot_canvas)
        right_plot_layout.setAlignment(self.right_plot_canvas, Qt.AlignCenter)

        widget = QWidget()
        right_plot_layout.addWidget(widget)
        layout = QHBoxLayout(widget)

        label = QLabel("Regressors to use for multilinear regression:")
        layout.addWidget(label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        right_plot_layout.addWidget(scroll_area)

        widget = QWidget()
        scroll_area.setWidget(widget)
        self.checkbox_layout = QVBoxLayout(widget)

        self.bottom_widget = QWidget()
        self.bottom_layout = QVBoxLayout(self.bottom_widget)
        self.main_layout.addWidget(self.bottom_widget)

        widget = QWidget()
        self.bottom_layout.addWidget(widget)
        layout = QHBoxLayout(widget)

        self.video_combobox = QComboBox()
        self.video_combobox.currentIndexChanged.connect(self.set_video)
        layout.addWidget(self.video_combobox)

        label = QLabel("Z: ")
        layout.addWidget(label)

        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(0, 0)
        self.z_slider.setValue(self.z)
        self.z_slider.setMaximumWidth(50)
        self.z_slider.valueChanged.connect(self.set_z)
        layout.addWidget(self.z_slider)

        self.z_label = QLabel(str(self.z))
        self.z_label.setMaximumWidth(10)
        layout.addWidget(self.z_label)

        self.remove_video_button = QPushButton("Remove Video")
        self.remove_video_button.clicked.connect(self.remove_selected_video)
        self.remove_video_button.setFixedWidth(120)
        self.remove_video_button.setEnabled(False)
        layout.addWidget(self.remove_video_button)

        widget = QWidget()
        self.bottom_layout.addWidget(widget)
        layout = QHBoxLayout(widget)

        label = QLabel("Frame Offset: ")
        layout.addWidget(label)

        self.tail_calcium_offset_slider = QSlider(Qt.Horizontal)
        self.tail_calcium_offset_slider.setRange(0, 10)
        self.tail_calcium_offset_slider.setValue(self.tail_calcium_offset)
        self.tail_calcium_offset_slider.valueChanged.connect(self.set_tail_calcium_offset)
        layout.addWidget(self.tail_calcium_offset_slider)

        self.tail_calcium_offset_label = QLabel(str(self.tail_calcium_offset))
        layout.addWidget(self.tail_calcium_offset_label)

        label = QLabel("Calcium FPS: ")
        layout.addWidget(label)

        self.calcium_fps_textbox = QLineEdit()
        self.calcium_fps_textbox.setText(str(self.calcium_fps))
        self.calcium_fps_textbox.setFixedWidth(50)
        self.calcium_fps_textbox.editingFinished.connect(self.set_calcium_fps)
        layout.addWidget(self.calcium_fps_textbox)

        label = QLabel("Tail Data FPS: ")
        layout.addWidget(label)

        self.tail_fps_textbox = QLineEdit()
        self.tail_fps_textbox.setText(str(self.tail_fps))
        self.tail_fps_textbox.setFixedWidth(50)
        self.tail_fps_textbox.editingFinished.connect(self.set_tail_fps)
        layout.addWidget(self.tail_fps_textbox)

        self.export_data_button = QPushButton("Export Data...")
        self.export_data_button.setFixedWidth(120)
        self.export_data_button.clicked.connect(self.export_data)
        self.export_data_button.setEnabled(False)
        layout.addWidget(self.export_data_button)

        self.calcium_video_fnames   = []
        self.roi_data_fnames        = []
        self.bout_fnames            = []
        self.frame_timestamp_fnames = []

        self.show()

    def export_data(self):
        pass

    def reset_variables(self):
        self.selected_video      = 0
        self.tail_calcium_offset = 0
        self.calcium_fps         = 3
        self.tail_fps            = 349

        self.z                   = 0
        self.max_p               = 0.05
        self.regressor_index     = 0
        self.n_regressors        = 0
        self.selected_regressors = []

        self.correlation_results     = None
        self.regression_coefficients = None
        self.regression_intercepts   = None
        self.regressors              = None
        self.spatial_footprints      = None
        self.temporal_footprints     = None
        self.calcium_video           = None
        self.mean_images             = None
        self.n_frames                = None
        self.roi_centers             = None

    def update_gui(self):
        if len(self.calcium_video_fnames) > 0:
            self.load_calcium_video_button.setText("✓ Load Calcium Video(s)...")
            self.remove_video_button.setEnabled(True)
        else:
            self.load_calcium_video_button.setText("Load Calcium Video(s)...")
            self.remove_video_button.setEnabled(False)

            self.regressor_index_slider.setRange(0, 0)
            self.regressor_index_label.setText("0")
            self.export_data_button.setEnabled(False)
            self.z_slider.setRange(0, 0)
            self.z_label.setText("0")
            self.calcium_video_loaded_label.setText("0 Loaded.")

            for i in range(self.video_combobox.count()-1, -1, -1):
                self.video_combobox.removeItem(i)

            self.update_plots()

        self.update_regressor_checkboxes()

        if len(self.roi_data_fnames) > 0:
            self.load_roi_data_button.setText("✓ Load ROI Data File(s)...")
        else:
            self.load_roi_data_button.setText("Load ROI Data File(s)...")

        if len(self.bout_fnames) > 0:
            self.load_bouts_button.setText("✓ Load Bout File(s)...")
        else:
            self.load_bouts_button.setText("Load Bout File(s)...")

        if len(self.frame_timestamp_fnames) > 0:
            self.load_frame_timestamps_button.setText("✓ Load Timestamp File(s)...")
        else:
            self.load_frame_timestamps_button.setText("Load Timestamp File(s)...")

    def load_calcium_video(self):
        calcium_video_fnames = QFileDialog.getOpenFileNames(window, 'Select calcium imaging video(s).', '', 'Videos (*.tiff *.tif)')[0]

        if calcium_video_fnames is not None and len(calcium_video_fnames) > 0:
            self.set_fnames(calcium_video_fnames, self.roi_data_fnames, self.bout_fnames, self.frame_timestamp_fnames)

            self.load_roi_data_button.setEnabled(True)
            self.load_bouts_button.setEnabled(True)
            self.load_frame_timestamps_button.setEnabled(True)


            self.roi_data_fnames        = []
            self.bout_fnames            = []
            self.frame_timestamp_fnames = []
            
        self.update_gui()

        self.calcium_video_loaded_label.setText("{} Loaded.".format(len(self.calcium_video_fnames)))

    def load_roi_data(self):
        roi_data_fnames = QFileDialog.getOpenFileNames(window, 'Select ROI data file(s).', '', 'Numpy files (*.npy)')[0]

        if roi_data_fnames is not None and len(roi_data_fnames) == len(self.calcium_video_fnames):
            self.set_fnames(self.calcium_video_fnames, roi_data_fnames, self.bout_fnames, self.frame_timestamp_fnames)

        self.update_gui()

    def load_bouts(self):
        bout_fnames = QFileDialog.getOpenFileNames(window, 'Select labeled bouts file(s).', '', 'CSV files (*.csv)')[0]

        if bout_fnames is not None and len(bout_fnames) == len(self.calcium_video_fnames):
            self.set_fnames(self.calcium_video_fnames, self.roi_data_fnames, bout_fnames, self.frame_timestamp_fnames)

        self.update_gui()

    def load_frame_timestamps(self):
        frame_timestamp_fnames = QFileDialog.getOpenFileNames(window, 'Select frame timestamp file.', '', 'Text files (*.txt)')[0]

        if frame_timestamp_fnames is not None and len(frame_timestamp_fnames) == len(self.calcium_video_fnames): 
            self.set_fnames(self.calcium_video_fnames, self.roi_data_fnames, self.bout_fnames, frame_timestamp_fnames)

        self.update_gui()

    def remove_selected_video(self):
        if len(self.calcium_video_fnames) > 0:
            del self.calcium_video_fnames[self.selected_video]
            del self.roi_data_fnames[self.selected_video]
            if len(self.bout_fnames) > self.selected_video:
                del self.bout_fnames[self.selected_video]
            if len(self.frame_timestamp_fnames) > self.selected_video:
                del self.frame_timestamp_fnames[self.selected_video]

            self.selected_video = max(0, self.selected_video-1)

            if len(self.calcium_video_fnames) > 0:
                self.set_video(self.selected_video)
            else:
                self.reset_variables()

        self.update_gui()

    def set_fnames(self, calcium_video_fnames, roi_data_fnames, bout_fnames, frame_timestamp_fnames):
        self.calcium_video_fnames   = calcium_video_fnames
        self.roi_data_fnames        = roi_data_fnames
        self.bout_fnames            = bout_fnames
        self.frame_timestamp_fnames = frame_timestamp_fnames

        for i in range(self.video_combobox.count(), -1, -1):
            self.video_combobox.removeItem(i)
        
        self.video_combobox.addItems(self.calcium_video_fnames)

    def set_video(self, i):
        self.selected_video = int(i)

        if len(self.calcium_video_fnames) > 0 and len(self.roi_data_fnames) == len(self.calcium_video_fnames) and (len(self.bout_fnames) == len(self.calcium_video_fnames) or len(self.frame_timestamp_fnames) == len(self.calcium_video_fnames)):
            self.do_regressor_analysis()

    def set_tail_calcium_offset(self, i):
        self.tail_calcium_offset = int(i)

        self.tail_calcium_offset_label.setText(str(self.tail_calcium_offset))
        
        if len(self.calcium_video_fnames) > 0 and len(self.roi_data_fnames) == len(self.calcium_video_fnames) and len(self.bout_fnames) == len(self.calcium_video_fnames):
            self.do_regressor_analysis()

    def set_calcium_fps(self):
        self.calcium_fps = float(self.calcium_fps_textbox.text())
        
        if len(self.calcium_video_fnames) > 0 and len(self.roi_data_fnames) == len(self.calcium_video_fnames) and len(self.bout_fnames) == len(self.calcium_video_fnames):
            self.do_regressor_analysis()

    def set_tail_fps(self):
        self.tail_fps = float(self.tail_fps_textbox.text())
        
        if len(self.calcium_video_fnames) > 0 and len(self.roi_data_fnames) == len(self.calcium_video_fnames) and len(self.bout_fnames) == len(self.calcium_video_fnames):
            self.do_regressor_analysis()

    def set_regressor_index(self, i):
        self.regressor_index = int(i)

        self.regressor_index_label.setText(str(self.regressor_index))

        self.update_plots()

    def set_max_p(self, i):
        self.max_p = i/100.0

        self.max_p_label.setText(str(self.max_p))

        self.update_plots()

    def set_z(self, i):
        self.z = int(i)

        self.z_label.setText(str(self.z))

        self.update_plots()

    def update_plots(self, selected_regressors=None):
        if self.correlation_results is not None:
            if selected_regressors is None:
                selected_regressors = self.regressors

            self.left_plot_canvas.plot_correlation(self.correlation_results, self.regressors, self.spatial_footprints, self.temporal_footprints, self.calcium_video, self.mean_images, self.n_frames, self.roi_centers, self.z, self.max_p, self.regressor_index)
            self.right_plot_canvas.plot_multilinear_regression(self.regression_coefficients, self.regression_intercepts, self.regression_scores, selected_regressors, self.spatial_footprints, self.temporal_footprints, self.calcium_video, self.mean_images, self.n_frames, self.roi_centers, self.z)
        else:
            self.left_plot_canvas.clear_plot()
            self.right_plot_canvas.clear_plot()

    def do_regressor_analysis(self):
        if len(self.frame_timestamp_fnames) == len(self.calcium_video_fnames):
            frame_timestamp_fnames = self.frame_timestamp_fnames[self.selected_video]
        else:
            frame_timestamp_fnames = None

        if len(self.bout_fnames) == len(self.calcium_video_fnames):
            bout_fnames = self.bout_fnames[self.selected_video]
        else:
            bout_fnames = None

        self.correlation_results, self.regression_coefficients, self.regression_intercepts, self.regression_scores, self.regressors, self.spatial_footprints, self.temporal_footprints, self.calcium_video, self.mean_images, self.n_frames, self.roi_centers = regressor_analysis(self.calcium_video_fnames[self.selected_video], self.roi_data_fnames[self.selected_video], bout_fnames, frame_timestamp_fnames, self.tail_calcium_offset)
        
        self.regressor_index_slider.setRange(0, len(self.regressors.keys())-1)
        self.z_slider.setRange(0, self.calcium_video.shape[1]-1)
        self.update_plots()
        self.update_regressor_checkboxes()

    def update_multilinear_regression(self):
        keys = list(self.regressors.keys())

        selected_keys = [ keys[i] for i in range(len(keys)) if self.checkboxes[i].isChecked() ]
        selected_regressors = {k:self.regressors[k] for k in selected_keys if k in self.regressors}

        self.regression_coefficients, self.regression_intercepts, self.regression_scores = multilinear_regression(selected_regressors, self.temporal_footprints)

        self.update_plots(selected_regressors=selected_regressors)

    def checkbox_state_changed(self, state):
        self.update_multilinear_regression()

    def update_regressor_checkboxes(self):
        for i in reversed(range(self.checkbox_layout.count())): 
            self.checkbox_layout.itemAt(i).widget().setParent(None)

        self.checkboxes = []

        if self.regressors is not None:
            keys = list(self.regressors.keys())

            for i in range(len(keys)):
                checkbox = QCheckBox(keys[i])
                checkbox.setChecked(True)
                self.checkbox_layout.addWidget(checkbox)
                checkbox.stateChanged.connect(self.checkbox_state_changed)
                self.checkboxes.append(checkbox)

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=16, dpi=30):
        self.fig = plt.figure(0, figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.setStyleSheet("background-color:rgba(0, 0, 0, 0);")

        self.fig.patch.set_alpha(0)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def clear_plot(self):
        self.fig.clear()
        self.draw()

    def plot_correlation(self, correlation_results, regressors, spatial_footprints, temporal_footprints, calcium_video, mean_images, n_frames, roi_centers, z, max_p, regressor_index):
        self.fig.clear()

        plot_correlation(correlation_results, regressors, spatial_footprints, temporal_footprints, calcium_video, mean_images, n_frames, roi_centers, z, max_p, regressor_index, fig=self.fig)

        self.draw()

    def plot_multilinear_regression(self, regression_coefficients, regression_intercepts, regression_scores, selected_regressors, spatial_footprints, temporal_footprints, calcium_video, mean_images, n_frames, roi_centers, z):
        self.fig.clear()

        plot_multilinear_regression(regression_coefficients, regression_intercepts, regression_scores, selected_regressors, spatial_footprints, temporal_footprints, calcium_video, mean_images, n_frames, roi_centers, z, fig=self.fig)

        self.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = PlotWindow()

    app.exec_()