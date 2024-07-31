import sys
import paramiko
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QLineEdit, 
    QMessageBox, QTextEdit, QComboBox, QDialog
)

class SSHClient(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('SSH Client')
        
        self.layout = QVBoxLayout()

        self.host = 'login.hpc.virginia.edu'

        self.host_label = QLabel('Host (default):')
        self.host_input = QLineEdit(self.host)
        self.layout.addWidget(self.host_label)
        self.layout.addWidget(self.host_input)
        
        self.username_label = QLabel('UVA Computing ID:')
        self.username_input = QLineEdit()
        self.layout.addWidget(self.username_label)
        self.layout.addWidget(self.username_input)
        
        self.password_label = QLabel('UVA Password:')
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)

        self.open_dropdown_button = QPushButton('Open Dropdown Window')
        self.open_dropdown_button.setStyleSheet('QPushButton {background-color: blue; color: white;}')
        self.open_dropdown_button.clicked.connect(self.open_dropdown_window)
        self.layout.addWidget(self.open_dropdown_button)
        
        self.setLayout(self.layout)
    
    def open_dropdown_window(self):
        username = self.username_input.text()
        password = self.password_input.text() 
        self.hide()
        self.dropdown_window = DropdownWindow(username, password, self)
        self.dropdown_window.show()

class DropdownWindow(QDialog):
    def __init__(self, username, password, parent=None):
        super().__init__(parent)
        self.username = username
        self.password = password
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Select Option and Run Script')
        
        self.layout = QVBoxLayout()
        self.instructions = ('To generate MC events, select Target Channel and Target Vertex. '
                             'Then enter the number of events and jobs (events x jobs = total events created).')

        self.command_label = QLabel('Instructions:')
        self.command_input = QTextEdit(self.instructions)
        self.command_input.setReadOnly(True)
        self.command_input.setStyleSheet("QTextEdit {background-color: #f0f0f0; font-family: Courier; font-size: 14px;}")
        self.layout.addWidget(self.command_label)
        self.layout.addWidget(self.command_input)

        self.channel_label = QLabel('Target Channel:')
        self.channel_dropdown = QComboBox()
        self.channel_dropdown.addItems(['DY', 'JPsi', 'Pion', 'MultiMuon'])  # Add your options here
        self.layout.addWidget(self.channel_label)
        self.layout.addWidget(self.channel_dropdown)

        self.vertex_label = QLabel('Target Vertex:')
        self.vertex_dropdown = QComboBox()
        self.vertex_dropdown.addItems(['All', 'Cylinder', 'Dump', 'Target', 'TargetDumpGap', 'Manual'])  # Add your options here
        self.layout.addWidget(self.vertex_label)
        self.layout.addWidget(self.vertex_dropdown)

        self.num_events_label = QLabel('Number of Events:')
        self.num_events_input = QLineEdit()
        self.layout.addWidget(self.num_events_label)
        self.layout.addWidget(self.num_events_input)

        self.num_jobs_label = QLabel('Number of Jobs:')
        self.num_jobs_input = QLineEdit()
        self.layout.addWidget(self.num_jobs_label)
        self.layout.addWidget(self.num_jobs_input)
        
        self.run_button = QPushButton('Run Script')
        self.run_button.setStyleSheet('QPushButton {background-color: red; color: white;}')
        self.run_button.clicked.connect(self.run_script)
        self.layout.addWidget(self.run_button)
        
        self.setLayout(self.layout)
    
    def run_script(self):
        channel = self.channel_dropdown.currentText()  # Get the selected option
        vertex = self.vertex_dropdown.currentText()  # Get the selected option

        try:
            num_events = int(self.num_events_input.text())  # Get the number of events
            num_jobs = int(self.num_jobs_input.text())  # Get the number of jobs
        except ValueError:
            QMessageBox.critical(self, 'Error', 'Number of Events and Number of Jobs must be integers.')
            return

        total_events = num_events * num_jobs
        
        host = 'login.hpc.virginia.edu'
        username = self.username
        password = self.password

        # Command can be customized as needed
        command = f'''cd /project/ptgroup/work/MC_Generation/{channel}_{vertex}_script; 
        source /project/ptgroup/spinquest/this-e1039.sh; 
        ./jobscript.sh {channel}_{vertex}_{total_events} {num_events} {num_jobs}'''
        
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            print(f"Connecting to {host} with username {username}")
            ssh.connect(host, username=username, password=password)
            print("Connection successful")

            stdin, stdout, stderr = ssh.exec_command(command)
            print(f"Executing command: {command}")

            output = stdout.read().decode()
            error = stderr.read().decode()

            ssh.close()
            print("Connection closed")

            if output:
                QMessageBox.information(self, 'Output', output)
            if error:
                QMessageBox.warning(self, 'Error', error)
        except paramiko.AuthenticationException:
            QMessageBox.critical(self, 'Error', 'Authentication failed, please verify your credentials')
        except paramiko.SSHException as sshException:
            QMessageBox.critical(self, 'Error', f'Unable to establish SSH connection: {sshException}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Operation failed: {str(e)}')
        finally:
            if ssh:
                ssh.close()
            self.parent.show()
            self.close()

def main():
    app = QApplication(sys.argv)
    client = SSHClient()
    client.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
