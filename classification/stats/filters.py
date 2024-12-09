import numpy as np
from stats.accuracy import accuracy_one_by_one

def indexes(data, column, value):
   return data[:, column] == value

def filter_by_column(data, column, value):
    return data[ indexes(data, column, value) ]

def filter_by_sex(data, sex):
    return filter_by_column(data, 0, sex)

def filter_by_result(data, value):
    return filter_by_column(data, 4, value)

def filter_wrong_imgs(x_data, metadata, sex):
  # filter images by sex
  filtered_imgs = filter_by_sex(x_data, sex)
  # filter predictions by sex
  filtered = filter_by_sex(metadata, sex)
  flag_results = accuracy_one_by_one(filtered)
  filtered_wrong_imgs = filtered_imgs[ flag_results[:, 4] == 0 ]
  return filtered_wrong_imgs

def filter_top_imgs(x_data, new_meta_data, sex, top_count):
  # get indexes by sex
  #print('new_meta_data ', new_meta_data.shape)
  filtered_indexes = indexes(new_meta_data, 0, sex)
  # filter images by sex
  filtered_imgs = x_data[ filtered_indexes ]
  # filter predictions by sex
  filtered_pred_data = new_meta_data[ filtered_indexes ]
  # max predictions by sex
  amax = np.argmax(filtered_pred_data, axis=1)
  #print('amax ', amax)
  # top indexes by sex
  top_indexes = np.argsort(amax)[::-1][:top_count]
  # top images by sex
  top_imgs = filtered_imgs[top_indexes]
  # top values by sex
  top_values = filtered_pred_data[top_indexes]
  return top_imgs, top_values