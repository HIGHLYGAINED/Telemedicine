import sys
import re
from wordcloud import WordCloud

from keras.optimizers import *
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2 as cv2

from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QVBoxLayout, QFileDialog , QDialog, QPushButton
from PyQt5.QtGui import QIcon, QImage, QPalette, QBrush
from PyQt5.QtCore import QSize, QDir
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class Menu(QMainWindow):

    def __init__(self):
        super().__init__()

        oImage = QImage("tt.png")

        sImage = oImage.scaled(QSize(300, 200))  # resize Image to widgets size

        palette = QPalette()

        palette.setBrush(10, QBrush(sImage))  # 10 = Windowrole

        self.setPalette(palette)

        self.initUI()

        self.show()

    def initUI(self):
        self.setGeometry(300, 400, 300, 200)
        self.setWindowTitle('Simple menu')
        self.statusBar()

        exitAct = QAction(QIcon('exit.png'), ' &Quit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('File')
        fileMenu.addAction(exitAct)

        Demo1 = menubar.addMenu('WordCloudDemo')

        Demo = QAction("Demo1", self)
        Demo.setStatusTip("Text Analyzer")
        Demo.triggered.connect(lambda: Mm())

        Demo1.addAction(Demo)

        Demo2 = menubar.addMenu('Facial Emotion Demo')

        Fer = QAction("FER", self)
        Fer.setStatusTip("Facial emotion Recognition")
        Fer.triggered.connect(lambda: Predict())

        Demo2.addAction(Fer)
        self.show()


class Mm(QDialog):

    def __init__(self, parent=None):
        super(Mm, self).__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.button = QPushButton('Select File')
        self.button.move(300, 200)
        self.button.clicked.connect(self.getfile)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        self.setLayout(layout)
        self.show()

    def getfile(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Single File', QDir.rootPath())
        self.word_cloud(filename)
        # print('The file name is...', filename)

    def word_cloud(self, filename):
        long_string = []
        with open(filename, 'r') as file:
            file_content = file.read()
            p = re.compile("[\d:\d:\d\n]").split(file_content)
            for i in range(len(p)):
                if p[i] is not '':
                    long_string.append(p[i])
        long_string = ','.join(long_string)
        # Create a WordCloud object
        word_cloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
        # Generate a word cloud
        word_cloud.generate(long_string)
        # Visualize the word cloud
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(word_cloud, interpolation='bilinear')
        ax.axis("off")
        self.canvas.draw()


class Predict(QDialog):

    def __init__(self, parent=None):
        super(Predict, self).__init__(parent)

        # self.filename = filename
        self.model = load_model('modelFg.h5')

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.button = QPushButton('Select File')
        self.button.move(300, 200)
        self.button.clicked.connect(self.getfile)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        self.setLayout(layout)
        self.show()

    def getfile(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Single File', QDir.rootPath())
        self.emotion_analysis(filename)
        # print('The file name is...', filename)

    def emotion_analysis(self, filename):
        imagePath = filename
        original = cv2.imread(imagePath)
        gray = cv2.imread(imagePath, 0)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15, minSize=(30, 30),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

        Labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        for (x, y, w, h) in faces:
            fimage = gray[y:y + h, x:x + w]
            fm = cv2.resize(fimage, (48, 48))
            fm = fm.astype("float") / 255.0
            fm = img_to_array(fm)
            fm = np.expand_dims(fm, axis=0)
            prediction = self.model.predict(fm)
            emotion_probability = np.max(prediction)
            label = Labels[prediction.argmax()]
            #     plt.gray()
            self.figure.clear()
            ax1 = self.figure.add_subplot(111)
            ax1.set_title("Emotion Probability is " + str(emotion_probability) + " and " + "Emotion is " + str(label).upper())
            ax1.imshow(original)
            # ax1.axis("off")
            self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Menu()
    sys.exit(app.exec_())
