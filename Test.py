import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import os as os
import keras
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import confusion_matrix

Labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def getData(filename):
    # images are 48x48
    # N = 35887
    hk = np.identity(7)
    Y = []
    X = []
    T = []
    first = True
    for line in open(filename):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            T.append(hk[int(row[0])])
            X.append([int(p) for p in row[1].split()])

    X, Y, T = np.array(X).reshape(len(X), 48, 48, 1) / 255.0, np.array(Y), np.array(T)
    return X, Y, T


x_test, y_test, ytest_ = getData("Testing.csv")
num_class = len(set(y_test))
y_test_ = keras.utils.to_categorical(y_test, num_class)

model = load_model('modelFg.h5')
# score = model.evaluate(x_test, y_test_,
#                        batch_size=64, verbose=1)
#
# print('Test accuracy:', score[1])

y_pred_class = model.predict_classes(x_test,batch_size=64)

cm = confusion_matrix(y_test,y_pred_class)
print(cm)

conf = pd.DataFrame(cm,index = Labels, columns = Labels)

conf = conf.apply(lambda x: x/sum(x) ,axis = 1)

plt.subplots(figsize=(15,15))
hm = sns.heatmap(conf, cmap="summer_r",annot=True,fmt='.2f', annot_kws={'size': 20})
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(),fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), fontsize=20)
# plt.savefig('Conf.pdf')
plt.show()