import dicom
import os
import numpy as np
import png
import itk_util as itk
from scipy.misc import imsave
import cv2
import imageio
import nibabel as nb
import imutils


from scipy import ndimage
from scipy import misc
from PIL import Image

def get_patient_number(i):
    if (i < 10):
        return "0{}".format(i)
    return i

def check_to_create_dir(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def dicom_to_png(dicom_file, png_file):
	#get data from dicom file
	dicom_data = dicom.read_file(dicom_file)
	shape = dicom_data.pixel_array.shape
	#print dicom_data.pixel_array.dtype
	#to numpy array
	image = np.array(dicom_data.pixel_array)
	#print image
	#print image.shape
	#print png_file
	#print dicom_file
	imsave(png_file, image)

def png_to_binary(png_file, binary_file):
	png_data = cv2.imread(png_file)
	(thresh, binary_image) = cv2.threshold(png_data, 129, 255, cv2.THRESH_BINARY)

	#print png_data.shape
	cv2.imwrite(binary_file, binary_image)

def hdr_to_png(hdr_file, png_file, index):
	#print hdr_file
	image = nb.load(hdr_file) #annotation
	#print image
	image_data = image.get_data() #img3d
	#print 'data shape lol'
	shape = image_data.shape #img.shape
	#print 'shape'
	#print shape
	#print 'data lol'
	#print image_data
	#insert backup code here
	x = shape[0]
	y = shape[1]
	s = shape[2]


	image_3d = np.zeros(x*y*s).reshape((x, y, s))

	for S in range(s):
		for X in range(x):
			for Y in range(y):
				image_3d[X, Y, S] = image_data[X, Y, S]
	X = image_3d.shape[0]
	Y = image_3d.shape[1]
	S = image_3d.shape[2]
	i = 0
	for s in range(S):
		image_2d = np.zeros(X*Y).reshape((X, Y))
		for x in range(X):
			for y in range(Y):
				image_2d[x, y] = image_3d[x, y, s]
		png_file_name = png_file + 'P%d-%04d' % (index, i) + '.png'
		i = i + 1
		imsave(png_file_name, image_2d)
		image_rotate = cv2.imread(png_file_name)
		new_image = imutils.rotate_bound(image_rotate, -90)
		imsave(png_file_name, new_image)



def convert_to_png_from_dicom(dicom_file_path, png_file_path, index):
	#check that dicom file exists
	if not os.path.exists(dicom_file_path):
		raise Exception('File "%s" does not exist' % dicom_file_path)

	#check that png file doesn't exist
	if os.path.exists(png_file_path):
		pass

	dicom_files = []

	for file in os.listdir(dicom_file_path):
		if file.endswith(".dcm"):
			dicom_files.append(os.path.join(dicom_file_path, file))

	dicom_files.sort()

	for i in range(len(dicom_files)):
		dicom_file = dicom_files[i]
		check_to_create_dir(png_file_path)
		png_file = png_file_path + 'P%d-%04d' % (index, i) + '.png'
		#print dicom_file
		dicom_to_png(dicom_file, png_file)

def convert_to_png_from_hdr(hdr_file_path, png_file_path, index):
	#check that dicom file exists
	if not os.path.exists(hdr_file_path):
		raise Exception('File "%s" does not exist' % hdr_file_path)

	#check that png file doesn't exist
	if os.path.exists(png_file_path):
		pass

	hdr_files = []

	for file in os.listdir(hdr_file_path):
		if file.endswith(".hdr"):
			hdr_files.append(os.path.join(hdr_file_path, file))

	hdr_files.sort()

	for i in range(len(hdr_files)):
		hdr_file = hdr_files[i]
		check_to_create_dir(png_file_path)
		png_file = png_file_path
		#print png_file
		hdr_to_png(hdr_file, png_file, index)

def convert_to_binary(png_file_path, binary_file_path, index):
	#check that dicom file exists
	if not os.path.exists(png_file_path):
		raise Exception('File "%s" does not exist' % png_file_path)

	#check that png file doesn't exist
	if os.path.exists(binary_file_path):
		pass

	png_files = []

	for file in os.listdir(png_file_path):
		if file.endswith(".png"):
			png_files.append(os.path.join(png_file_path, file))

	png_files.sort()

	for i in range(len(png_files)):
		png_file = png_files[i]
		check_to_create_dir(binary_file_path)
		binary_file = binary_file_path + 'P%d-%04d' % (index, i) + '.png'
		png_to_binary(png_file, binary_file)

def setup(path):
	#create folders for png
	for i in range(1, 17):
		print 'Working on Patient {}'.format(i)

		p = get_patient_number(i)

		dicom_sub = path + 'P{}/P{}dicom/'.format(p, p)
		dicom_sub_alt = path + 'P{}/patient{}/P{}dicom/'.format(p, p, p)

		if os.path.exists(dicom_sub):
			real_dicom_dir = dicom_sub
			real_hdr_dir = real_dicom_dir
		else:
			real_dicom_dir = dicom_sub_alt
			real_hdr_dir = real_dicom_dir

		#train
		png_original_sub = 'P{}/P{}original/'.format(p, p)
		png_original_dir = '/Users/stevenchen/OneDrive/Documents/UCF/sophomore_2016-2017/Undergraduate Research/RV_CNN_data/train/' + png_original_sub

		convert_to_png_from_dicom(real_dicom_dir, png_original_dir, i)

		png_segmented_sub = 'P{}/P{}segmented/'.format(p, p)
		png_segmented_dir = '/Users/stevenchen/OneDrive/Documents/UCF/sophomore_2016-2017/Undergraduate Research/RV_CNN_data/train/' + png_segmented_sub

		convert_to_png_from_hdr(real_hdr_dir, png_segmented_dir, i)

		binary_sub = 'P{}/P{}binary/'.format(p, p)
		binary_dir = '/Users/stevenchen/OneDrive/Documents/UCF/sophomore_2016-2017/Undergraduate Research/RV_CNN_data/train/' + binary_sub
		convert_to_binary(png_segmented_dir, binary_dir, i)

		#test
		png_original_sub1 = 'P{}/P{}original/'.format(p, p)
		png_original_dir1 = '/Users/stevenchen/OneDrive/Documents/UCF/sophomore_2016-2017/Undergraduate Research/RV_CNN_data/test/' + png_original_sub1

		convert_to_png_from_dicom(real_dicom_dir, png_original_dir1, i)

		png_segmented_sub1 = 'P{}/P{}segmented/'.format(p, p)
		png_segmented_dir1 = '/Users/stevenchen/OneDrive/Documents/UCF/sophomore_2016-2017/Undergraduate Research/RV_CNN_data/test/' + png_segmented_sub1

		convert_to_png_from_hdr(real_hdr_dir, png_segmented_dir1, i)

		binary_sub1 = 'P{}/P{}binary/'.format(p, p)
		binary_dir1 = '/Users/stevenchen/OneDrive/Documents/UCF/sophomore_2016-2017/Undergraduate Research/RV_CNN_data/test/' + binary_sub1
		convert_to_binary(png_segmented_dir1, binary_dir1, i)

		#valid
		png_original_sub2 = 'P{}/P{}original/'.format(p, p)
		png_original_dir2 = '/Users/stevenchen/OneDrive/Documents/UCF/sophomore_2016-2017/Undergraduate Research/RV_CNN_data/valid/' + png_original_sub2

		convert_to_png_from_dicom(real_dicom_dir, png_original_dir2, i)

		png_segmented_sub2 = 'P{}/P{}segmented/'.format(p, p)
		png_segmented_dir2 = '/Users/stevenchen/OneDrive/Documents/UCF/sophomore_2016-2017/Undergraduate Research/RV_CNN_data/valid/' + png_segmented_sub2

		convert_to_png_from_hdr(real_hdr_dir, png_segmented_dir2, i)

		binary_sub2 = 'P{}/P{}binary/'.format(p, p)
		binary_dir2 = '/Users/stevenchen/OneDrive/Documents/UCF/sophomore_2016-2017/Undergraduate Research/RV_CNN_data/valid/' + binary_sub2
		convert_to_binary(png_segmented_dir2, binary_dir2, i)

		print 'Finished with Patient {}'.format(i)

setup('/Users/stevenchen/OneDrive/Documents/UCF/sophomore_2016-2017/Undergraduate Research/RV_segmentation/')
