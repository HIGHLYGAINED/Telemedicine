import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras 


data = pd.read_csv("Training.csv")

d = pd.DataFrame(data.emotion.value_counts())
d.reset_index(inplace = True)
d.rename(columns ={"index": "label", "emotion": "Count"}, inplace=True)

# def weights(data):
#     h = {}
#     t = data['Count']
#     l = data['label']
#     for i in range(0,len(t)):
#         ratio = 1/ (t[i]/max(t))
#         h[l[i]] = ratio
#     return h


# class_weight = weights(d)


def getData(filename):
    # images are 48x48
    # N = 35887
    hk=np.identity(7)
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
            X.append([int(p) for p in row[1].split()] )


    X, Y, T = np.array(X).reshape(len(X),48,48,1)/255.0, np.array(Y), np.array(T)
    return X, Y, T


X, Y, Y_  = getData("Training.csv")
num_class = len(set(Y))

X_val, Y_val, Yval_ = getData("Validation.csv")
X_test, Y_test, Ytest_ = getData("Testing.csv")

X0, Y0, Y0_ = X[Y!=1, :], Y[Y!=1, ], Y_[Y!=1, : ]
X1 = X[Y==1, :]
Y1_ = Y_[Y==1, :]

X1 = np.repeat(X1, 15, axis=0)
Y1_ = np.repeat(Y1_, 15, axis=0)

XM = np.vstack([X0, X1])
YM_ = np.vstack([Y0_, Y1_])

YM = np.concatenate((Y0, [1]*len(X1)))

# Random Shuffle
print(XM.shape, YM.shape, YM_.shape)
perm = np.arange(len(YM))
np.random.shuffle(perm)
XM, YM, YM_ =XM[perm], YM[perm], YM_[perm]

x_train = X
y_train = Y
x_val = X_val
x_test = X_test
y_val = Y_val
y_test = Y_test
xm_train = XM
ym_train = YM
y_train = keras.utils.to_categorical(y_train, num_class)
y_val =  keras.utils.to_categorical(y_val, num_class)
y_test = keras.utils.to_categorical(y_test, num_class)
ym_train = keras.utils.to_categorical(ym_train, num_class)

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(#rotation_range= 30,
                   zoom_range= 0.20,
                   shear_range= 0.20,
                   #horizontal_flip=True,
                   #vertical_flip= True,
                   fill_mode="nearest")


from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.layers import Input


from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced',
                                                 np.unique(Y),
                                                 Y)
class_weight = dict(enumerate(class_weight))


def my_model():
    model = Sequential()
    input_shape = (48, 48, 1)
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(7))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adagrad(lr=0.01))
    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE
    # model.summary()

    return model


model = my_model()
model.summary()



# path_model='modelFf.h5' # save model at this location after each epoch
# modelF = my_model() # create the model
# h = modelF.fit(x_train,y_train,
#                 batch_size = 64,
#                 epochs= 20,
#                 verbose=1,
#                 validation_data=(x_val,y_val),
#                 shuffle=True,
#                 # class_weight = class_weight,
#                 callbacks=[
#                 ModelCheckpoint(filepath = path_model, mode = 'auto', verbose =1, save_best_only = True),
#                 ReduceLROnPlateau(),
#                 EarlyStopping(mode= 'auto',verbose = 1)]
#             )


path_model='modelFg.h5' # save model at this location after each epoch
modelF = my_model() # create the model
# fit the model
h = modelF.fit_generator(datagen.flow(x_train,y_train, batch_size = 64),
              epochs= 20,
              verbose=1,
              validation_data = (x_val,y_val),
              shuffle=True,
              class_weight = class_weight,
              callbacks=[
                ModelCheckpoint(filepath = path_model, mode = 'auto', verbose =1, save_best_only = True),
                ReduceLROnPlateau(),
                EarlyStopping(mode= 'auto',verbose = 1)]
            )

