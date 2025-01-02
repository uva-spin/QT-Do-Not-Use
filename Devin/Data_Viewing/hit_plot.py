import sys
import os
import numpy as np
import uproot
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QSizePolicy, QPushButton
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class FileWatcher(FileSystemEventHandler):
    def __init__(self, callback, new_directory_callback):
        super().__init__()
        self.callback = callback
        self.new_directory_callback = new_directory_callback

    def on_created(self, event):
        if event.is_directory:
            if event.src_path.startswith(r'/home/devin/Documents/Big_Data/run_005994'):
            #if event.src_path.startswith('/data4/e1039_data/online/sraw/run_'):
                self.new_directory_callback(event.src_path)
        elif event.src_path.endswith('.root'):
            self.callback(event.src_path)

class DetectorPlot(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.file_paths = []
        self.current_file_index = 0

    def create_plot(self):
        print(f"Creating cumulative plot for file: {self.file_paths[self.current_file_index]}")
        self.ax.clear()

        # Set the maximum detector ID and element ID ranges
        max_detector_id = 61   # x-axis range (Detector ID)
        max_element_id = 201   # y-axis range (Element ID)
        
        # Initialize an empty occupancy array for the specified range
        cumulative_occupancy = np.zeros((max_detector_id + 1, max_element_id + 1))

        # Read and accumulate the event data from the ROOT file
        if self.file_paths:
            file_path = self.file_paths[self.current_file_index]
            total_events = self.get_total_events(file_path)

            for event_number in range(total_events):
                detectorid, elementid = self.read_event(file_path, event_number)

                # Accumulate hits into the cumulative occupancy matrix
                occupancy, _, _ = np.histogram2d(detectorid, elementid, bins=(np.arange(0, max_detector_id+2), np.arange(0, max_element_id+2)))
                cumulative_occupancy += occupancy

        # Define a colormap from dark to light yellow
        cmap = LinearSegmentedColormap.from_list('yellow_colormap', ['#000000', '#FFFF00'], N=256)

        # Plot the cumulative occupancy matrix
        cax = self.ax.imshow(cumulative_occupancy.T, origin='lower', cmap=cmap, interpolation='nearest', aspect='auto')

        # Add a single colorbar for the entire plot
        plt.colorbar(cax, ax=self.ax, label='Occupancy (Total Number of Hits)')

        # Set axis labels and title with specified ranges
        self.ax.set_xlabel('Detector ID')
        self.ax.set_ylabel('Element ID')
        self.ax.set_title(f'Cumulative Hit Occupancy for All Events')
        
        # Set the x-axis and y-axis limits
        self.ax.set_xlim([0, max_detector_id])
        self.ax.set_ylim([0, max_element_id])

        self.draw()

    def read_event(self, file_path, event_number):
        with uproot.open(file_path + ":save") as file:
            detectorid = file["fAllHits.detectorID"].array(library="np")[event_number]
            elementid = file["fAllHits.elementID"].array(library="np")[event_number]
        return detectorid, elementid

    def get_total_events(self, file_path):
        # Get the total number of events in the file
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
            self.file_paths = [sorted_files[1]]  # Take the second most recent file
        else:
            self.file_paths = sorted_files  # Fall back to the most recent file if there is only one
        self.current_file_index = 0
        self.event_number = 0
        self.create_plot()

class MainWindow(QMainWindow):
    def __init__(self, directory):
        super().__init__()

        self.setWindowTitle('Detector Hit Display')

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        main_layout = QVBoxLayout(self.main_widget)

        # Horizontal layout for the status indicator and file name
        status_layout = QHBoxLayout()
        self.status_label = QLabel(self)
        self.file_name_label = QLabel(self)
        self.update_status_label(False)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.file_name_label)
        status_layout.addStretch(1)  # Add stretch to push the labels to the left
        main_layout.addLayout(status_layout)

        # Add control buttons
        control_layout = QHBoxLayout()
        self.stop_button = QPushButton("Stop")
        self.pause_button = QPushButton("Pause")
        self.restart_button = QPushButton("Restart")
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.restart_button)
        main_layout.addLayout(control_layout)

        self.plot = DetectorPlot(self.main_widget)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.plot)

        # Set initial size of the window (width, height)
        self.setGeometry(100, 100, 1600, 800)  # Adjust the width and height as needed

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(1000)  # Update every 1000 milliseconds (1 second)

        self.observer = Observer()
        self.directory = directory
        event_handler = FileWatcher(self.on_new_file, self.on_new_directory)
        self.observer.schedule(event_handler, self.directory, recursive=False)
        self.observer.start()

        self.check_initial_files()

        # Connect buttons to their respective functions
        self.stop_button.clicked.connect(self.stop_display)
        self.pause_button.clicked.connect(self.pause_display)
        self.restart_button.clicked.connect(self.restart_display)

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

        # Check for a new run directory
        self.check_for_new_directory()

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

    def check_for_new_directory(self):
        parent_dir = r"/home/devin/Documents/Big_Data/run_005994"
        directories = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith("run_")]
        if directories:
            latest_directory = max(directories, key=lambda d: int(d.split("_")[1]))
            new_directory_path = os.path.join(parent_dir, latest_directory)
            if new_directory_path != self.directory:
                self.switch_directory(new_directory_path)

    def switch_directory(self, new_directory):
        print(f"Switching to new directory: {new_directory}")
        self.directory = new_directory
        self.observer.unschedule_all()
        event_handler = FileWatcher(self.on_new_file, self.on_new_directory)
        self.observer.schedule(event_handler, self.directory, recursive=False)
        self.check_initial_files()

if __name__ == "__main__":
    initial_directory = r"/home/devin/Documents/Big_Data/run_005994"
    app = QApplication(sys.argv)
    mainWin = MainWindow(initial_directory)
    mainWin.show()
    sys.exit(app.exec_())
