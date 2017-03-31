import dicom
import os
import numpy as np
import png
import itk_util as itk
from scipy.misc import imsave
import cv2
import imageio
import nibabel as nb

from scipy import ndimage
from scipy import misc

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

	'''
	#rescale between 0-255
	scaled_images = []
	for row in images:
		scaled_row = []
		for col in row:
			scaled_col = int((float(col) / float(max_value)) * 255.0)
			scaled_row.append(scaled_col)
		scaled_images.append(scaled_row)

	print scaled_images

	#write png files
	f = png.Writer(shape[1], shape[0], greyscale = True)
	f.write(png_file, scaled_images)
	'''

def png_to_binary(png_file, binary_file):
	png_data = cv2.imread(png_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	(thresh, binary_image) = cv2.threshold(png_data, 128, 255, cv2.THRESH_BINARY)

	cv2.imwrite(binary_file, binary_image)

def hdr_to_png(hdr_file, png_file):
	print hdr_file
	image = nb.load(hdr_file)
	print image
	data = image.get_data()
	print 'data shape lol'
	shape = data.shape
	print 'shape'
	print shape
	print 'data lol'
	x = shape[0]
	y = shape[1]
	#print x, y, z
	#z = hdr_data[3]
	t = 20
	s = shape[2]/t

	image_array = np.zeros(x*y*s).reshape((x, y, s))

	for T in range(t):
		for S in range(s):
			for X in range(x):
				for Y in range(y):
					st = S*t + T
					image_array[S, T, y - 1 - Y] = data[X, Y, st]
	print image_array

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
		png_file = png_file_path + 'P%d-%04d' % (index, i) + '.png'
		#print png_file
		hdr_to_png(hdr_file, png_file)

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
	for i in range(1, 16):
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

		png_sub = 'P{}/P{}png/'.format(p, p)
		png_dir = path + png_sub

		convert_to_png_from_hdr(real_hdr_dir, png_dir, i)

		binary_sub = 'P{}/P{}binary/'.format(p, p)
		binary_dir = path + binary_sub
		convert_to_binary(png_dir, binary_dir, i)

		print 'Finished with Patient {}'.format(i)

setup('/Users/stevenchen/OneDrive/Documents/UCF/sophomore_2016-2017/Undergraduate Research/RV_segmentation/')
