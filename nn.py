from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from collections import Counter
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def categorical_to_numpy(labels_in):
  labels = []
  for label in labels_in:
    if label == 'dog':
      labels.append(np.array([1, 0]))
    else:
      labels.append(np.array([0, 1]))
  return np.array(labels)

def one_hot_encoding(input):
  output = np.array(input)
  output = np.zeros((input.size, input.max()+1))
  output[np.arange(input.size),input] = 1
  
  return output


def load_data():
  # Run this cell to download our data into a file called 'cifar_data'
  !wget -O cifar_data https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%204%20_%205%20-%20Neural%20Networks%20_%20CNN/dogs_v_roads

  # now load the data from our cloud computer
  import pickle
  data_dict = pickle.load(open( "cifar_data", "rb" ));
  
  data   = data_dict['data']
  labels = data_dict['labels']
  
  return data, labels

def plot_one_image(data, labels, img_idx):
  from google.colab.patches import cv2_imshow
  import cv2
  import matplotlib.pyplot as plt
  my_img   = data[img_idx, :].squeeze().reshape([32,32,3]).copy()
  my_label = labels[img_idx]
  print('label: %s'%my_label)
  fig, ax = plt.subplots(1,1)

  img = ax.imshow(my_img, extent=[-1,1,-1,1])

  x_label_list = [0, 8, 16, 24, 32]
  y_label_list = [0, 8, 16, 24, 32]

  ax.set_xticks([-1, -0.5, 0, 0.5, 1])
  ax.set_yticks([-1, -0.5, 0, 0.5, 1])

  ax.set_xticklabels(x_label_list)
  ax.set_yticklabels(y_label_list)

  fig.show(img)
  
def CNNClassifier(num_epochs=30, layers=4, dropout=0.5):
  def create_model():
    model = Sequential()
    model.add(Reshape((32, 32, 3)))
    
    for i in range(layers):
      model.add(Conv2D(32, (3, 3), padding='same'))
      model.add(Activation('relu'))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model
  return KerasClassifier(build_fn=create_model, epochs=num_epochs, batch_size=10, verbose=2)

def plot_acc(history, ax = None, xlabel = 'Epoch #'):
    history = history.history
    history.update({'epoch':list(range(len(history['val_accuracy'])))})
    history = pd.DataFrame.from_dict(history)

    best_epoch = history.sort_values(by = 'val_accuracy', ascending = False).iloc[0]['epoch']

    if not ax:
      f, ax = plt.subplots(1,1)
    sns.lineplot(x = 'epoch', y = 'val_accuracy', data = history, label = 'Validation', ax = ax)
    sns.lineplot(x = 'epoch', y = 'accuracy', data = history, label = 'Training', ax = ax)
    ax.axhline(0.5, linestyle = '--',color='red', label = 'Chance')
    ax.axvline(x = best_epoch, linestyle = '--', color = 'green', label = 'Best Epoch')  
    ax.legend(loc = 7)    
    ax.set_ylim([0.4, 1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy (Fraction)')
    
    plt.show()

def main_function():
    data, labels = load_data()
    img_height =  32#@param {type:"integer"}
    img_width =  32#@param {type:"integer"}
    color_channels =  3#@param {type:"integer"}

    if img_height == 32 and img_width == 32 and color_channels == 3:
        print("Correct!")
        print ("Each image is", img_height, 'x', img_width, 'pixels.')
        print ("Each pixel has", color_channels, "channels for red, green, blue.")
        print ("This gives a total of", img_height * img_width * color_channels, "intensity values per image.")
    else:
        print("Those aren't quite the values.")
        print("Your values give a total of", img_height * img_width * color_channels, "intensity values per image.") 
        print("Discuss with your group and try again!")
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.2)

    # Initialize our model
    knn_model = KNeighborsClassifier() # Change this!

    # Train our model
    knn_model.fit(X_train, y_train)
    # Test our model
    y_pred = knn_model.predict(X_test)
        # Print the score on the testing data
    print(accuracy_score(y_test, y_pred))

    k_values = [1, 3, 5, 10, 20, 30]
    for k in k_values:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.2)
        # Initialize our model
        knn_model = KNeighborsClassifier(n_neighbors = k) # Change this!
        # Train our model
        knn_model.fit(X_train, y_train)
        # Test our model
        y_pred = knn_model.predict(X_test)
        # Print the score on the testing data
        print(k, ":", accuracy_score(y_test, y_pred))
    
    nnet = MLPClassifier(hidden_layer_sizes=(3), random_state=1, max_iter=10000000)  ## How many hidden layers? How many neurons does this have?
    nnet.fit(X_train, y_train)

    # Predict what the classes are based on the testing data
    predictions = nnet.predict(X_test)

    values = [(1, 1), (3, 3), (5, 5), (8, 8), (10, 10)]
    for value in values:
        nnet = MLPClassifier(hidden_layer_sizes = value, max_iter=10000000)  ## How many hidden layers? How many neurons does this have?
        nnet.fit(X_train, y_train)

        # Predict what the classes are based on the testing data
        predictions = nnet.predict(X_test)

        # Print the score on the testing data
        print(value,  "Accuracy:")
        print(accuracy_score(y_test, predictions)*100)