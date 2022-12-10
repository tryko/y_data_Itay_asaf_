
import numpy  as np 
def clip_array(arr, max_val):
  # Create a copy of the original array
  arr_clipped = arr.copy()
  # Find rows where the sum of the values is greater than max_val
  rows_to_clip = arr[arr.sum(axis=1) > max_val]
  # For each of these rows, replace the largest value with a value
  # that will make the sum of the row equal to max_val
  row_sums = rows_to_clip.sum(axis=1)
  v = rows_to_clip.max(axis=1)
  arr_clipped[np.arange(len(row_sums)), rows_to_clip.argmax(axis=1)] = v - (row_sums - max_val)

  return arr_clipped

  a = np.array([[1, 5, 4], [-3, 2, 8]])
  print(clip_array(a,8))
  