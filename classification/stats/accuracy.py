import numpy as np

def accuracy_one_by_one(data_array):
  # Extract values from the two columns
  column1_values = data_array[:, 2]
  column2_values = data_array[:, 3]
  comparison_result = np.where(column1_values == column2_values, 1, 0)
  return np.column_stack((data_array, comparison_result))

def accuracy(data):
  return np.mean(data[:, 0] == data[:, 1], axis=0) * 100

def accuracy_old(data_array):
  # Extract actual and predicted values
  
  actual_values = data_array[:, 0]
  predicted_values = data_array[:, 1]
  # Calculate accuracy
  return np.mean(actual_values == predicted_values) * 100
