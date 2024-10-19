import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

X_test = np.load('storage/X_test.npy')
y_test = np.load('storage/y_test.npy')

model = pickle.load(open('model\LSTM.sav', 'rb'))

y_pred = model.predict(X_test)  # X_test là dữ liệu kiểm tra
y_pred = [1 if value >= 0.5 else 0 for value in y_pred]

matrix = confusion_matrix(y_test, y_pred)

# Hiển thị confusion matrix
matrix = ConfusionMatrixDisplay(confusion_matrix=matrix)
matrix.plot()
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print()
print('Chỉ số Accuracy là: ', accuracy)
print('Chỉ số Precision là: ', precision)
print('Chỉ số Recall là: ', recall)
print('Chỉ số F1-score là: ', f1)
print()





