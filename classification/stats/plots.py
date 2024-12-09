import numpy as np
import matplotlib.pyplot as plt

def display_image(img_array, count):
  count = min(count, img_array.shape[0])
  fig, axes = plt.subplots(1, count, figsize=(count*2, 2))
  
  for i in range(count):
    image_to_plot = np.transpose(img_array[i], (1, 2, 0))
    axes[i].imshow(image_to_plot)
  plt.show()

def display_sample(img_array, n_img):
  n_img = min(n_img, img_array.shape[0])
  fig, ax = plt.subplots(1, n_img, figsize=(n_img*2, 2))
  for i in range(n_img):
      ax[i].imshow(img_array[i], cmap='gray')
  plt.show()