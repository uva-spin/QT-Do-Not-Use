import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QLineEdit, QMessageBox, QTextEdit
import paramiko

class SSHClient(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('SSH Client')
        
        self.layout = QVBoxLayout()

        self.host = 'login.hpc.virginia.edu'
        self.instructions = 'To generate MC events, please input your UVA Computing ID and associated password, and then press "Run Script"! \t\t Please also make sure that you are connected to the UVA Anywhere VPN in order to access Rivanna\'s HPC.'

        self.host_label = QLabel('Host:')
        self.host_input = QLineEdit(self.host)
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

        self.command_label = QLabel('Instructions:')
        self.command_input = QTextEdit(self.instructions)
        self.command_input.setReadOnly(True)
        self.command_input.setStyleSheet("QTextEdit {background-color: #f0f0f0; font-family: Courier; font-size: 14px;}")
        self.layout.addWidget(self.command_label)
        self.layout.addWidget(self.command_input)
        
        self.button = QPushButton('Run Script')
        self.button.setStyleSheet('QPushButton {background-color: red; color: white;}')
        self.button.clicked.connect(self.run_script)
        self.layout.addWidget(self.button)
        
        self.setLayout(self.layout)
    
    def run_script(self):
        host = self.host_input.text()
        username = self.username_input.text()
        password = self.password_input.text()
        
        command = './jobscript.sh'  # We need to put our command here once we've figured out how to connect
        
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host, username=username, password=password)
            
            stdin, stdout, stderr = ssh.exec_command(command)
            
            output = stdout.read().decode()
            error = stderr.read().decode()
            
            ssh.close()
            
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


