import numpy as np # linear algebra
import matplotlib.pyplot as plt # showing and rendering figures

def plot(trace_file, print_out=True):
  delimiter = ','
  trace_data = np.genfromtxt(trace_file, delimiter=delimiter, skip_header=1)

  epochs = len(trace_data)
  if print_out: print('Epochs:', epochs)
  training_accuracy   = trace_data[:, 1]
  training_loss       = trace_data[:, 3]
  validation_accuracy = trace_data[:, 2]
  validation_loss     = trace_data[:, 4]

  max_validation_accuracy_epoch = np.argmax(validation_accuracy)
  if print_out: print('Best epoch (regarding validation accuracy) {:}, training accuracy: {:.4f}, validation accuracy: {:.4f}'.format(max_validation_accuracy_epoch+1, training_accuracy[max_validation_accuracy_epoch], validation_accuracy[max_validation_accuracy_epoch]))

  min_validation_loss_epoch = np.argmin(validation_loss)
  if print_out: print('Best epoch (regarding validation loss) {:}, training loss: {:.4f}, validation loss: {:.4f}'.format(min_validation_loss_epoch+1, training_loss[min_validation_loss_epoch], validation_loss[min_validation_loss_epoch]))

  # training curves
  plt.figure(figsize=(12, 6))
  plt.plot(range(1, epochs + 1), training_accuracy)
  plt.plot(range(1, epochs + 1), validation_accuracy)
  plt.plot(range(1, epochs + 1), training_loss)
  plt.plot(range(1, epochs + 1), validation_loss)
  plt.legend(['training_accuracy','validation_accuracy','training_loss','validation_loss'])
  plt.ylim(0, 1)
  plt.xlim(1, epochs)
  plt.yticks(np.arange (0.5, 1.01, 0.1))
  plt.xticks(range(0, epochs + 1, 4))
  plt.show()