from model.functions import model_LSTM
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
import pickle
import numpy as np


location_video = np.load(r'storage\keypoints.npy')
numeric_labels = np.load(r'storage\numeric_labels.npy')

window_size = 12
batch_size = 16
num_feature = 132

X_train, X_test, y_train, y_test = train_test_split(location_video, 
                                                    numeric_labels, 
                                                    test_size = 0.3, 
                                                    random_state = 42)

model_lstm = model_LSTM(window_size, 
                   num_feature,
                   batch_size, 
                   X_train, 
                   y_train)

model_lstm.summary()

plot_model(model_lstm, 
           to_file = 'model\model_structure.png', 
           show_shapes = True, 
           show_layer_names = True)

pickle.dump(model_lstm, open('model\LSTM.sav', 'wb'))

np.save('storage\X_test.npy', X_test)
np.save('storage\y_test.npy', y_test)







