from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from scipy import ndimage
from six.moves import xrange
import tensorflow as tf
IMAGE_SIZE = 320

NUM_CLASSES = 10
NUM_DATA_PER_EPOCH_FOR_TRAIN = 50000
NUM_DATA_PER_EPOCH_FOR_EVAL = 10000

original_path = "/Users/stevenchen/OneDrive/Documents/UCF/sophomore_2016-2017/Undergraduate Research/RV"
list_files_original = []
for dirName, subdirList, fileList in os.walk(original_path):
	for filename in fileList:
		if "dcm" in filename.lower():
			list_files_original.append(os.path.join(dirName,filename))
list_files_original.sort()

print list_files_original[1 : range(len(list_files_original))]
