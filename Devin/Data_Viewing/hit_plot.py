
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

class FileWatcher(FileSystemEventHandler):
    def __init__(self, callback, new_directory_callback):
        super().__init__()
        self.callback = callback
        self.new_directory_callback = new_directory_callback

    def on_created(self, event):
        if event.is_directory:
            if event.src_path.startswith('../sraw/run_'):
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
        self.event_number = 0

    def create_plot(self):
        print(f"Creating plot for file: {self.file_paths[self.current_file_index]} | Event number: {self.event_number}")
        self.ax.clear()

        # Read the event data from the ROOT file
        if self.file_paths:
            file_path = self.file_paths[self.current_file_index]
            detectorid, elementid = self.read_event(file_path, self.event_number)
        else:
            detectorid, elementid = np.array([]), np.array([])

        # Check if there are hits
        if len(detectorid) == 0 or len(elementid) == 0:
            print(f"No hits for event number {self.event_number}")
            return

        # Define the detector groups and their properties
        detector_groups = [
            {'label': 'Station 1', 'detectors': [
                {'name': 'D0V', 'elements': 201, 'id': 5},
                {'name': 'D0Vp', 'elements': 201, 'id': 6},
                {'name': 'D0Xp', 'elements': 160, 'id': 4},
                {'name': 'D0X', 'elements': 160, 'id': 3},
                {'name': 'D0U', 'elements': 201, 'id': 1},
                {'name': 'D0Up', 'elements': 201, 'id': 2}
            ]},
            {'label': 'Hodo', 'detectors': [
                {'name': 'H1L', 'elements': 20, 'id': 33},
                {'name': 'H1R', 'elements': 20, 'id': 34},
                {'name': 'H1B', 'elements': 23, 'id': 31},
                {'name': 'H1T', 'elements': 23, 'id': 32}
            ]},
            {'label': 'DP-1', 'detectors': [
                {'name': 'DP1TL', 'elements': 80, 'id': 55},
                {'name': 'DP1TR', 'elements': 80, 'id': 56},
                {'name': 'DP1BL', 'elements': 80, 'id': 57},
                {'name': 'DP1BR', 'elements': 80, 'id': 58}
            ]},
            {'label': 'Station 2', 'detectors': [
                {'name': 'D2V', 'elements': 128, 'id': 13},
                {'name': 'D2Vp', 'elements': 128, 'id': 14},
                {'name': 'D2Xp', 'elements': 112, 'id': 15},
                {'name': 'D2X', 'elements': 112, 'id': 16},
                {'name': 'D2U', 'elements': 128, 'id': 17},
                {'name': 'D2Up', 'elements': 128, 'id': 18}
            ]},
            {'label': 'Hodo', 'detectors': [
                {'name': 'H2R', 'elements': 19, 'id': 36},
                {'name': 'H2L', 'elements': 19, 'id': 35},
                {'name': 'H2T', 'elements': 16, 'id': 38},
                {'name': 'H2B', 'elements': 16, 'id': 37}
            ]},
            {'label': 'DP-2', 'detectors': [
                {'name': 'DP2TL', 'elements': 48, 'id': 59},
                {'name': 'DP2TR', 'elements': 48, 'id': 60},
                {'name': 'DP2BL', 'elements': 48, 'id': 61},
                {'name': 'DP2BR', 'elements': 48, 'id': 62}
            ]},
            {'label': 'Station 3+', 'detectors': [
                {'name': 'D3pVp', 'elements': 134, 'id': 19},
                {'name': 'D3pV', 'elements': 134, 'id': 20},
                {'name': 'D3pXp', 'elements': 116, 'id': 21},
                {'name': 'D3pX', 'elements': 116, 'id': 22},
                {'name': 'D3pUp', 'elements': 134, 'id': 23},
                {'name': 'D3pU', 'elements': 134, 'id': 24}
            ]},
            {'label': 'Station 3-', 'detectors': [
                {'name': 'D3mVp', 'elements': 134, 'id': 25},
                {'name': 'D3mV', 'elements': 134, 'id': 26},
                {'name': 'D3mXp', 'elements': 116, 'id': 27},
                {'name': 'D3mX', 'elements': 116, 'id': 28},
                {'name': 'D3mUp', 'elements': 134, 'id': 29},
                {'name': 'D3mU', 'elements': 134, 'id': 30}
            ]},
            {'label': 'Hodo', 'detectors': [
                {'name': 'H3T', 'elements': 16, 'id': 40},
                {'name': 'H3B', 'elements': 16, 'id': 39}
            ]},
            {'label': 'Prop', 'detectors': [
                {'name': 'P1Y1', 'elements': 72, 'id': 47},
                {'name': 'P1Y2', 'elements': 72, 'id': 48}
            ]},
            {'label': 'Hodo', 'detectors': [
                {'name': 'H4Y1R', 'elements': 16, 'id': 42},
                {'name': 'H4Y1L', 'elements': 16, 'id': 41}
            ]},
            {'label': 'Prop', 'detectors': [
                {'name': 'P1X1', 'elements': 72, 'id': 49},
                {'name': 'P1X2', 'elements': 72, 'id': 50}
            ]},
            {'label': 'Hodo', 'detectors': [
                {'name': 'H4Y2R', 'elements': 16, 'id': 44},
                {'name': 'H4Y2L', 'elements': 16, 'id': 43},
                {'name': 'H4T', 'elements': 16, 'id': 46},
                {'name': 'H4B', 'elements': 16, 'id': 45}
            ]},
            {'label': 'Prop', 'detectors': [
                {'name': 'P2X1', 'elements': 72, 'id': 51},
                {'name': 'P2X2', 'elements': 72, 'id': 52},
                {'name': 'P2Y1', 'elements': 72, 'id': 53},
                {'name': 'P2Y2', 'elements': 72, 'id': 54}
            ]}
        ]

        y_max = max([det['elements'] for group in detector_groups for det in group['detectors']])  # Maximum number of elements for scaling

        x_labels = []
        x_ticks = []
        x_offset = 0

        # Plot each group with spacing between them
        for group in detector_groups:
            x_positions = range(x_offset, x_offset + len(group['detectors']))
            for idx, x in enumerate(x_positions):
                detector = group['detectors'][idx]
                y_positions = range(detector['elements'])

                for y in y_positions:
                    # Scale y to match the maximum elements
                    scaled_y = y * y_max / detector['elements']
                    height = y_max / detector['elements']

                    # Check if the element is a hit
                    is_hit = any((detectorid == detector['id']) & (elementid == (y + 1)))
                    color = 'yellow' if is_hit else 'darkblue'

                    self.ax.add_patch(plt.Rectangle((x, scaled_y), 1, height, edgecolor='black', facecolor=color))

                # Add x-axis labels
                x_labels.append(detector['name'])
                x_ticks.append(x + 0.5)

            # Label the group
            self.ax.text(x_offset + len(group['detectors']) / 2, y_max + 15, group['label'], ha='center', va='center')

            # Add spacing between groups
            x_offset += len(group['detectors']) + 2

        # Set the limits and labels
        self.ax.set_xlim(0, x_offset)
        self.ax.set_ylim(0, y_max + 20)
        self.ax.set_xticks(x_ticks)
        self.ax.set_xticklabels(x_labels, rotation=90)
        self.ax.set_xlabel('')
        self.ax.set_ylabel('Element ID')
        self.draw()

    def read_event(self, file_path, event_number):
        with uproot.open(file_path + ":save") as file:
            detectorid = file["fAllHits.detectorID"].array(library="np")[event_number]
            elementid = file["fAllHits.elementID"].array(library="np")[event_number]
        return detectorid, elementid

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
        parent_dir = "../Big_Data/sraw"
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
    initial_directory = r"../Big_Data/sraw/run_005994"
    app = QApplication(sys.argv)
    mainWin = MainWindow(initial_directory)
    mainWin.show()
    sys.exit(app.exec_())
