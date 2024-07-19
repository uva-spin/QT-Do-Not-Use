import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QLineEdit, QMessageBox
import subprocess

class SSHClient(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('SSH Client')
        
        self.layout = QVBoxLayout()
        
        self.host_label = QLabel('Host:')
        self.host_input = QLineEdit()
        self.layout.addWidget(self.host_label)
        self.layout.addWidget(self.host_input)
        
        self.username_label = QLabel('Username:')
        self.username_input = QLineEdit()
        self.layout.addWidget(self.username_label)
        self.layout.addWidget(self.username_input)
        
        self.password_label = QLabel('Password:')
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)
        
        self.button = QPushButton('Run Script')
        self.button.setStyleSheet('QPushButton {background-color: red; color: white;}')
        self.button.clicked.connect(self.run_script)
        self.layout.addWidget(self.button)
        
        self.setLayout(self.layout)
    
    def run_script(self):
        host = self.host_input.text()
        username = self.username_input.text()
        password = self.password_input.text()
        
        command = '/path/to/your/script.sh'  # Place your command here
        
        sshpass_command = [
            'sshpass', '-p', password, 'ssh', 
            f'{username}@{host}', command
        ]
        
        try:
            result = subprocess.run(sshpass_command, capture_output=True, text=True)
            
            output = result.stdout
            error = result.stderr
            
            if output:
                QMessageBox.information(self, 'Output', output)
            if error:
                QMessageBox.warning(self, 'Error', error)
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

def main():
    app = QApplication(sys.argv)
    client = SSHClient()
    client.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

