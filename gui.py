import sys
from PyQt5 import QtCore, QtGui, uic,QtWidgets

qtFile = 'simple.ui' # Enter file here.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtFile)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle('LBP GUI')
        self.setWindowIcon(QtGui.QIcon('myicon.png'))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())