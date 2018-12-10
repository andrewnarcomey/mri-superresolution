import pydicom
import os
import math
import numpy as np
import png
from config import config

def load_dir(curr_folder):
	files = []
	num_dcm = len([name for name in os.listdir(curr_folder) if name.endswith('.dcm')])
	for i in range(1, num_dcm+1):
		int_fill = ""
		length = len(str(i))
		int_fill += "0"*(4-length) + str(i)
		fp = curr_folder + "IM-0001-" + int_fill + ".dcm"
		dcm = pydicom.dcmread(fp)
		files.append(dcm)
	return files

def read_all_data(data_folder):
	X = []
	Y = []

	for i in range(len(config.patient_numbers)):
		patient_num = config.patient_numbers[i]
		patient_folder = data_folder + "/anon" + str(patient_num)
		l_patient = patient_folder + "l/"
		h_patient = patient_folder + "h/"
		x = load_dir(l_patient)
		y = load_dir(h_patient)
		X.append(x)
		Y.append(y)

	return X, Y

def match_dcm(X, Y):
	pairs = []
	for patient in range(len(X)):
		patient_pairs = [] # tuples containing (index of low res imag, index of closest high res img)
		possible_matches = list(range(len(Y[patient])))
		curr_index = 0
		for i in range(len(X[patient])):
			l_x, l_y, l_z = X[patient][i].ImagePositionPatient
			possible_y = []
			j = curr_index
			while len(possible_y) < 10:
				if j in possible_matches: possible_y.append(j)
				j += 1
				if j > possible_matches[-1]: break
			min_dist = float('inf')
			min_index = float('inf')
			# greedy matching
			for index in possible_y:
				h_x, h_y, h_z = Y[patient][index].ImagePositionPatient
				dist = math.sqrt((h_x-l_x)**2 + (h_y-l_y)**2 + (h_z-l_z)**2)
				if dist < min_dist:
					min_dist = dist
					min_index = index
			curr_index = min_index
			possible_matches.remove(min_index)
			patient_pairs.append((i, min_index))
		pairs.append(patient_pairs)
	return pairs

def split_pairs(pairings):
	X = []
	Y = []
	for patient_pairs in pairings:
		X_patient = []
		Y_patient = []
		for pair in patient_pairs:
			X_patient.append(pair[0])
			Y_patient.append(pair[1])
		X.append(X_patient)
		Y.append(Y_patient)
	return X, Y

# def convert_to_png(path):
# 	#ds = pydicom.dcmread(path)
# 	#ds = dcm_list[0]
# 	ds = pydicom.dcmread(path)
# 	shape = ds.pixel_array.shape
# 	# Convert to float to avoid overflow or underflow losses.
# 	image_2d = ds.pixel_array.astype(float)
# 	# Rescaling grey scale between 0-255
# 	image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
# 	# Convert to uint
# 	image_2d_scaled = np.uint8(image_2d_scaled)
# 	# Write the PNG file
# 	destination = 'test3.png'
# 	with open(destination, 'wb') as png_file:
# 		w = png.Writer(shape[1], shape[0], greyscale=True)
# 		w.write(png_file, image_2d_scaled)

def write_pairs_to_file(path, data_pairs):
	with open(path, 'w') as f:
		for i, patient in enumerate(data_pairs):
			f.write("{}\n".format("Patient " + str(config.patient_numbers[i])))
			for dcm_index in patient:
				f.write("{}\n".format(dcm_index))


def load_scan(path):
    all_slices = [pydicom.read_file(path + '/' + s, force = True) for s in os.listdir(path)]
    slice_list = []
    incomplete_slices = []
    for slice in all_slices:
        if hasattr(slice, 'InstanceNumber'):
            slice_list.append(slice)
        else:
            incomplete_slices.append(slice)
    slice_list.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slice_list[0].ImagePositionPatient[2] - slice_list[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slice_list[0].SliceLocation - slice_list[1].SliceLocation)
        
    for s in slice_list:
        s.SliceThickness = slice_thickness
        
    return slice_list

def select_data(X_dcm, X_indices, Y_dcm, Y_indices):
	# X = [X_dcm[patient][index].pixel_array for patient in range(len(X_indices)) for index in X_indices[patient]]
	# Y = [Y_dcm[patient][index].pixel_array for patient in range(len(Y_indices)) for index in Y_indices[patient]]
	X = []
	Y = []
	#assert(len(Y_indices)==len(X_indices))
	for patient in range(len(Y_indices)):
		for index in Y_indices[patient]:
			Y.append(Y_dcm[patient][index].pixel_array)
		for index in X_indices[patient]:
			X.append(X_dcm[patient][index].pixel_array)
	return X, Y
