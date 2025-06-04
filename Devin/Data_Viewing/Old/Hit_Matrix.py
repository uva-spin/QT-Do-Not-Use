import sys
import os
import numpy as np
import uproot
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                           QLabel, QHBoxLayout, QSizePolicy, QPushButton, 
                           QFileDialog, QMessageBox)
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# Set matplotlib style for better appearance
plt.style.use('seaborn-darkgrid')
mpl.rcParams['figure.facecolor'] = '#f0f0f0'
mpl.rcParams['axes.facecolor'] = '#ffffff'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3


class FileWatcher(FileSystemEventHandler):
    def __init__(self, callback, new_directory_callback):
        super().__init__()
        self.callback = callback
        self.new_directory_callback = new_directory_callback

    def on_created(self, event):
        if event.is_directory:
            if event.src_path.startswith(r'J:\Users\Devin\Desktop\Spin Physics Work\Q-Tracker\Big_Data\sraw\run_'):
                self.new_directory_callback(event.src_path)
        elif event.src_path.endswith('.root'):
            self.callback(event.src_path)


class DetectorPlot(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(12, 8), dpi=100)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.file_paths = []
        self.current_file_index = 0
        self.accumulate_real_time = True
        self.cached_data = {}  # Cache for event data
        self.max_cache_size = 1000  # Maximum number of events to cache

    def create_plot(self):
        print(f"Creating cumulative plot for file: {self.file_paths[self.current_file_index]}")
        self.ax.clear()

        max_detector_id = 61
        max_element_id = 201
        
        cumulative_occupancy = np.zeros((max_detector_id + 1, max_element_id + 1))

        if self.file_paths:
            file_path = self.file_paths[self.current_file_index]
            total_events = self.get_total_events(file_path)

            for event_number in range(total_events):
                detectorid, elementid = self.read_event(file_path, event_number)

                # Accumulate hits
                occupancy, _, _ = np.histogram2d(detectorid, elementid, 
                                               bins=(np.arange(0, max_detector_id + 2), 
                                                     np.arange(0, max_element_id + 2)))
                cumulative_occupancy += occupancy

                if self.accumulate_real_time:
                    self.update_plot(cumulative_occupancy, event_number)

            if not self.accumulate_real_time:
                self.update_plot(cumulative_occupancy)

    def update_plot(self, cumulative_occupancy, event_number=None):
        # Create a custom colormap with better visibility
        colors = ['#000000', '#1a1a1a', '#333333', '#4d4d4d', '#666666', 
                 '#808080', '#999999', '#b3b3b3', '#cccccc', '#e6e6e6', '#ffffff']
        cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)
        
        self.ax.clear()
        cax = self.ax.imshow(cumulative_occupancy.T, origin='lower', cmap=cmap, 
                            interpolation='nearest', aspect='auto')

        # Add colorbar with better formatting
        cbar = self.figure.colorbar(cax, ax=self.ax, label='Occupancy (Total Number of Hits)')
        cbar.ax.tick_params(labelsize=10)
        
        # Improve axis labels and title
        self.ax.set_xlabel('Detector ID', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Element ID', fontsize=12, fontweight='bold')
        
        if event_number is not None:
            self.ax.set_title(f'Hit Occupancy for Event {event_number}', 
                            fontsize=14, fontweight='bold', pad=20)
        else:
            self.ax.set_title('Cumulative Hit Occupancy for All Events', 
                            fontsize=14, fontweight='bold', pad=20)

        self.ax.set_xlim([0, 61])
        self.ax.set_ylim([0, 201])
        
        # Add grid lines
        self.ax.grid(True, linestyle='--', alpha=0.3)
        
        # Improve tick labels
        self.ax.tick_params(axis='both', which='major', labelsize=10)
        
        self.draw()

        if self.accumulate_real_time:
            plt.pause(0.05)  # Reduced pause time for better performance

    def read_event(self, file_path, event_number):
        # Check cache first
        cache_key = f"{file_path}_{event_number}"
        if cache_key in self.cached_data:
            return self.cached_data[cache_key]

        with uproot.open(file_path + ":save") as file:
            detectorid = file["fAllHits.detectorID"].array(library="np")[event_number]
            elementid = file["fAllHits.elementID"].array(library="np")[event_number]
            
            # Cache the result
            self.cached_data[cache_key] = (detectorid, elementid)
            
            # Manage cache size
            if len(self.cached_data) > self.max_cache_size:
                # Remove oldest entries
                oldest_key = next(iter(self.cached_data))
                del self.cached_data[oldest_key]
                
        return detectorid, elementid

    def get_total_events(self, file_path):
        with uproot.open(file_path + ":save") as file:
            total_events = len(file["fAllHits.detectorID"].array(library="np"))
        return total_events

    def update_event(self):
        if self.file_paths:
            file_path = self.file_paths[self.current_file_index]
            with uproot.open(file_path + ":save") as file:
                total_events = len(file["fAllHits.detectorID"].array(library="np"))

            while self.event_number < total_events:
                detectorid, elementid = self.read_event(file_path, self.event_number)
                if len(detectorid) == 0 or len(elementid) == 0:
                    print(f"No hits for event number {self.event_number}")
                    self.event_number = (self.event_number + 1) % total_events
                else:
                    self.create_plot()
                    self.event_number = (self.event_number + 1) % total_events
                    break

            if self.event_number == 0:
                self.current_file_index = (self.current_file_index + 1) % len(self.file_paths)

    def update_files(self, file_paths):
        sorted_files = sorted(file_paths, key=os.path.getctime, reverse=True)
        if len(sorted_files) > 1:
            self.file_paths = [sorted_files[1]]
        else:
            self.file_paths = sorted_files
        self.current_file_index = 0
        self.event_number = 0
        self.create_plot()


class MainWindow(QMainWindow):
    def __init__(self, directory=None):
        super().__init__()

        self.setWindowTitle('Detector Hit Display')
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2d6da3;
            }
            QLabel {
                font-size: 12px;
                font-weight: bold;
            }
        """)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        main_layout = QVBoxLayout(self.main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Status and control layouts
        status_layout = QHBoxLayout()
        self.status_label = QLabel(self)
        self.file_name_label = QLabel(self)
        self.update_status_label(False)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.file_name_label)
        status_layout.addStretch(1)
        main_layout.addLayout(status_layout)

        control_layout = QHBoxLayout()
        self.stop_button = QPushButton("Stop")
        self.pause_button = QPushButton("Pause")
        self.restart_button = QPushButton("Restart")
        self.real_time_button = QPushButton("Real-Time Accumulation: OFF")
        self.select_dir_button = QPushButton("Select Directory")
        
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.restart_button)
        control_layout.addWidget(self.real_time_button)
        control_layout.addWidget(self.select_dir_button)
        main_layout.addLayout(control_layout)

        # Plot widget
        self.plot = DetectorPlot(self.main_widget)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.plot)

        # Set window size
        self.setGeometry(100, 100, 1800, 1000)

        # Timer and observer for filesystem updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(1000)

        self.observer = Observer()
        self.directory = directory
        if self.directory:
            self.setup_observer()
        else:
            self.select_directory()

        # Connect buttons to actions
        self.stop_button.clicked.connect(self.stop_display)
        self.pause_button.clicked.connect(self.pause_display)
        self.restart_button.clicked.connect(self.restart_display)
        self.real_time_button.clicked.connect(self.toggle_real_time)
        self.select_dir_button.clicked.connect(self.select_directory)

    def setup_observer(self):
        event_handler = FileWatcher(self.on_new_file, self.on_new_directory)
        self.observer.schedule(event_handler, self.directory, recursive=False)
        self.observer.start()
        self.check_initial_files()

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.directory = directory
            self.observer.unschedule_all()
            self.setup_observer()
            QMessageBox.information(self, "Directory Selected", 
                                  f"Now monitoring directory: {directory}")

    def closeEvent(self, event):
        self.observer.stop()
        self.observer.join()
        event.accept()

    def check_initial_files(self):
        files = [f for f in os.listdir(self.directory) if f.endswith('.root')]
        if files:
            file_paths = [os.path.join(self.directory, f) for f in files]
            self.update_files(file_paths)

    def on_new_file(self, file_path):
        print(f"New file detected: {file_path}")
        self.update_files(self.plot.file_paths + [file_path])

    def on_new_directory(self, directory_path):
        print(f"New directory detected: {directory_path}")
        self.switch_directory(directory_path)

    def update_files(self, file_paths):
        self.plot.update_files(file_paths)
        self.update_status_label(True)
        self.update_file_name_label()

    def update_display(self):
        if self.plot.event_number == 0:
            self.update_status_label(True)
        else:
            self.update_status_label(False)
        self.plot.update_event()
        self.update_file_name_label()

    def stop_display(self):
        self.timer.stop()

    def pause_display(self):
        if self.timer.isActive():
            self.timer.stop()
            self.pause_button.setText("Resume")
        else:
            self.timer.start(1000)
            self.pause_button.setText("Pause")

    def restart_display(self):
        self.plot.event_number = 0
        self.plot.current_file_index = 0
        self.timer.start(1000)
        self.pause_button.setText("Pause")
        self.update_display()

    def update_status_label(self, is_new_file):
        color = 'green' if is_new_file else 'red'
        self.status_label.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
        self.status_label.setFixedSize(20, 20)
        self.status_label.setAlignment(Qt.AlignCenter)

    def update_file_name_label(self):
        if self.plot.file_paths:
            file_name = os.path.basename(self.plot.file_paths[self.plot.current_file_index])
            event_number = self.plot.event_number
            self.file_name_label.setText(f"Reading file: {file_name} | Event number: {event_number}")
        else:
            self.file_name_label.setText("No file to read")

    def toggle_real_time(self):
        self.plot.accumulate_real_time = not self.plot.accumulate_real_time
        if self.plot.accumulate_real_time:
            self.real_time_button.setText("Real-Time Accumulation: ON")
        else:
            self.real_time_button.setText("Real-Time Accumulation: OFF")

    def switch_directory(self, new_directory):
        print(f"Switching to new directory: {new_directory}")
        self.directory = new_directory
        self.observer.unschedule_all()
        event_handler = FileWatcher(self.on_new_file, self.on_new_directory)
        self.observer.schedule(event_handler, self.directory, recursive=False)
        self.check_initial_files()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detector Hit Display')
    parser.add_argument('--directory', type=str, help='Initial directory to monitor')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    mainWin = MainWindow(args.directory)
    mainWin.show()
    sys.exit(app.exec_())
