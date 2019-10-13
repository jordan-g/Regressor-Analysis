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
        self.width  = 800
        self.height = 800

        self.selected_video      = 0
        self.tail_calcium_offset = 0

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
        self.load_calcium_video_button.setFixedWidth(200)
        self.load_calcium_video_button.clicked.connect(self.load_calcium_video)
        layout.addWidget(self.load_calcium_video_button)

        layout.addStretch()

        self.load_roi_data_button = QPushButton("Load ROI Data File(s)...")
        self.load_roi_data_button.setFixedWidth(200)
        self.load_roi_data_button.clicked.connect(self.load_roi_data)
        self.load_roi_data_button.setEnabled(False)
        layout.addWidget(self.load_roi_data_button)

        layout.addStretch()

        self.load_bouts_button = QPushButton("Load Bout File(s)...")
        self.load_bouts_button.setFixedWidth(200)
        self.load_bouts_button.clicked.connect(self.load_bouts)
        self.load_bouts_button.setEnabled(False)
        layout.addWidget(self.load_bouts_button)

        layout.addStretch()

        self.load_frame_timestamps_button = QPushButton("Load Timestamp File(s)...")
        self.load_frame_timestamps_button.setFixedWidth(200)
        self.load_frame_timestamps_button.clicked.connect(self.load_frame_timestamps)
        self.load_frame_timestamps_button.setEnabled(False)
        layout.addWidget(self.load_frame_timestamps_button)

        layout.addStretch()

        self.calcium_video_loaded_label = QLabel("0 Loaded.")
        layout.addWidget(self.calcium_video_loaded_label)

        self.top_widget = QWidget()
        self.top_layout = QHBoxLayout(self.top_widget)
        self.main_layout.addWidget(self.top_widget)

        self.plot_canvas = PlotCanvas(self, width=8, height=8)
        self.top_layout.addWidget(self.plot_canvas)

        self.bottom_widget = QWidget()
        self.bottom_layout = QVBoxLayout(self.bottom_widget)
        self.main_layout.addWidget(self.bottom_widget)

        widget = QWidget()
        self.bottom_layout.addWidget(widget)
        layout = QHBoxLayout(widget)

        self.video_combobox = QComboBox()
        self.video_combobox.currentIndexChanged.connect(self.set_video)
        layout.addWidget(self.video_combobox)

        self.remove_video_button = QPushButton("Remove Video")
        self.remove_video_button.clicked.connect(self.remove_selected_video)
        self.remove_video_button.setFixedWidth(150)
        layout.addWidget(self.remove_video_button)

        widget = QWidget()
        self.bottom_layout.addWidget(widget)
        layout = QHBoxLayout(widget)

        label = QLabel("Frame Offset: ")
        layout.addWidget(label)

        self.tail_calcium_offset_slider = QSlider(Qt.Horizontal)
        self.tail_calcium_offset_slider.setRange(0, 10)
        self.tail_calcium_offset_slider.valueChanged.connect(self.set_tail_calcium_offset)
        layout.addWidget(self.tail_calcium_offset_slider)

        self.export_data_button = QPushButton("Export Data...")
        self.export_data_button.setFixedWidth(150)
        self.export_data_button.clicked.connect(self.export_data)
        layout.addWidget(self.export_data_button)

        self.calcium_video_fnames   = []
        self.roi_data_fnames        = []
        self.bout_fnames            = []
        self.frame_timestamp_fnames = []

        self.show()

    def export_data(self):
        pass

    def update_button_text(self):
        if len(self.calcium_video_fnames) > 0:
            self.load_calcium_video_button.setText("✓ Load Calcium Video(s)...")
        else:
            self.load_calcium_video_button.setText("Load Calcium Video(s)...")

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
            
        self.update_button_text()

        self.calcium_video_loaded_label.setText("{} Loaded.".format(len(self.calcium_video_fnames)))

    def load_roi_data(self):
        roi_data_fnames = QFileDialog.getOpenFileNames(window, 'Select ROI data file(s).', '', 'Numpy files (*.npy)')[0]

        if roi_data_fnames is not None and len(roi_data_fnames) == len(self.calcium_video_fnames):
            self.set_fnames(self.calcium_video_fnames, roi_data_fnames, self.bout_fnames, self.frame_timestamp_fnames)

        self.update_button_text()

    def load_bouts(self):
        bout_fnames = QFileDialog.getOpenFileNames(window, 'Select labeled bouts file(s).', '', 'CSV files (*.csv)')[0]

        if bout_fnames is not None and len(bout_fnames) == len(self.calcium_video_fnames):
            self.set_fnames(self.calcium_video_fnames, self.roi_data_fnames, bout_fnames, self.frame_timestamp_fnames)

        self.update_button_text()

    def load_frame_timestamps(self):
        frame_timestamp_fnames = QFileDialog.getOpenFileNames(window, 'Select frame timestamp file.', '', 'Text files (*.txt)')[0]

        if frame_timestamp_fnames is not None and len(bout_fnames) == len(self.calcium_video_fnames): 
            self.set_fnames(self.calcium_video_fnames, self.roi_data_fnames, bout_fnames, frame_timestamp_fnames)

        self.update_button_text()

    def remove_selected_video(self):
        if len(self.calcium_video_fnames) > 0:
            del self.calcium_video_fnames[self.selected_video]
            del self.roi_data_fnames[self.selected_video]
            del self.bout_fnames[self.selected_video]
            del self.frame_timestamp_fnames[self.selected_video]

            self.selected_video = max(0, self.selected_video-1)

            if len(self.calcium_video_fnames) > 0:
                self.set_video(self.selected_video)

        self.update_button_text()

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

        if len(self.calcium_video_fnames) > 0 and len(self.roi_data_fnames) == len(self.calcium_video_fnames) and len(self.bout_fnames) == len(self.calcium_video_fnames):
            self.do_regressor_analysis()

    def set_tail_calcium_offset(self, i):
        self.tail_calcium_offset = int(i)
        
        if len(self.calcium_video_fnames) > 0 and len(self.roi_data_fnames) == len(self.calcium_video_fnames) and len(self.bout_fnames) == len(self.calcium_video_fnames):
            self.do_regressor_analysis()

    def do_regressor_analysis(self):
        if len(self.frame_timestamp_fnames) == len(self.calcium_video_fnames):
            self.correlation_results, self.regression_coefficients, self.regression_intercepts, self.regressors, self.spatial_footprints, self.temporal_footprints, self.calcium_video, self.mean_images, self.n_frames, self.roi_centers = regressor_analysis(self.calcium_video_fnames[self.selected_video], self.roi_data_fnames[self.selected_video], self.bout_fnames[self.selected_video], self.frame_timestamp_fnames[self.selected_video], self.tail_calcium_offset)
        else:
            self.correlation_results, self.regression_coefficients, self.regression_intercepts, self.regressors, self.spatial_footprints, self.temporal_footprints, self.calcium_video, self.mean_images, self.n_frames, self.roi_centers = regressor_analysis(self.calcium_video_fnames[self.selected_video], self.roi_data_fnames[self.selected_video], self.bout_fnames[self.selected_video], None, self.tail_calcium_offset)

        self.plot_canvas.plot(self.correlation_results, self.regression_coefficients, self.regression_intercepts, self.regressors, self.spatial_footprints, self.temporal_footprints, self.calcium_video, self.mean_images, self.n_frames, self.roi_centers)
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = plt.figure(0, figsize=(width, height), dpi=dpi)
        self.fig.patch.set_alpha(0)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.setStyleSheet("background-color:rgba(0, 0, 0, 0);")

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, correlation_results, regression_coefficients, regression_intercepts, regressors, spatial_footprints, temporal_footprints, calcium_video, mean_images, n_frames, roi_centers):
        self.fig.clear()
        self.fig.patch.set_alpha(0)

        plot_regressor_analysis(correlation_results, regression_coefficients, regression_intercepts, regressors, spatial_footprints, temporal_footprints, calcium_video, mean_images, n_frames, roi_centers, fig=self.fig)

        self.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = PlotWindow()

    app.exec_()