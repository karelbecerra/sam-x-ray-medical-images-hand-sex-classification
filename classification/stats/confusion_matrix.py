from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(confusion_matrix):
  disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels = ['Female', 'Male'])
  disp.plot()
  plt.show()

def get_confusion_matrix(y_data, predicted_data):
  y_pred = np.argmax(predicted_data, axis=1)
  y_true = np.argmax(y_data, axis=1)
  conf_mat = confusion_matrix(y_true, y_pred)
  return conf_mat, y_pred, predicted_data
