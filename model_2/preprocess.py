from utils import *
from config import config
import pickle
import tensorlayer as tl

### This file should be run locally prior to running main.py ###

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
    patients_lr = []
    patients_hr = []
    for i in range(len(config.patient_numbers)):
        patient_num = config.patient_numbers[i]
        patient_folder = data_folder + "/anon" + str(patient_num)
        l_patient = patient_folder + "l/"
        h_patient = patient_folder + "h/"
        lr = load_dir(l_patient)
        hr = load_dir(h_patient)
        patients_lr.append(lr)
        patients_hr.append(hr)
    return patients_lr, patients_hr

def match_dcm(lr, hr):
    pairs = []
    for patient in range(len(lr)):
        patient_pairs = [] # tuples containing (index of low res imag, index of closest high res img)
        possible_matches = list(range(len(hr[patient])))
        curr_index = 0
        for i in range(len(lr[patient])):
            l_x, l_y, l_z = lr[patient][i].ImagePositionPatient
            possible_hr = []
            j = curr_index
            while len(possible_hr) < 10:
                if j in possible_matches: possible_hr.append(j)
                j += 1
                if j > possible_matches[-1]: break
            min_dist = float('inf')
            min_index = float('inf')
            # greedy matching
            for index in possible_hr:
                h_x, h_y, h_z = hr[patient][index].ImagePositionPatient
                dist = math.sqrt((h_x-l_x)**2 + (h_y-l_y)**2 + (h_z-l_z)**2)
                if dist < min_dist:
                    min_dist = dist
                    min_index = index
            curr_index = min_index
            possible_matches.remove(min_index)
            patient_pairs.append((i, min_index))
        pairs.append(patient_pairs)
    return pairs

def select_and_save_data(lr_dcm, lr_indices, hr_dcm, hr_indices):
    folder = 'preprocessed_data2'
    tl.files.exists_or_mkdir(folder)
    print("[*] Processing patients")
    for patient in range(len(hr_indices)):
        patient_str = str(config.patient_numbers[patient])
        image_shape = lr_dcm[patient][0].pixel_array.shape
        n_images = len(lr_indices[patient])
    
        subfolder_l = folder + "/" + patient_str + "l"
        subfolder_h = folder + "/" + patient_str + "h"
        tl.files.exists_or_mkdir(subfolder_l)
        tl.files.exists_or_mkdir(subfolder_h)

        for i in range(n_images):
            index = lr_indices[patient][i]
            np.save(subfolder_l + "/" + patient_str + "l_" + str(i) + ".npy", lr_dcm[patient][index].pixel_array)
        for j in range(n_images):
            index = hr_indices[patient][j]
            multi_array = np.zeros((3, 512, 512))
            base_array = hr_dcm[patient][index].pixel_array
            if index == 0:
                multi_array[0] = base_array
                multi_array[1] = hr_dcm[patient][index+1].pixel_array
                multi_array[2] = hr_dcm[patient][index+2].pixel_array
            elif index == hr_indices[patient][-1]:
                multi_array[0] = hr_dcm[patient][index-2].pixel_array
                multi_array[1] = hr_dcm[patient][index-1].pixel_array
                multi_array[2] = base_array
            else:
                multi_array[0] = hr_dcm[patient][index-1].pixel_array
                multi_array[1] = base_array
                multi_array[2] = hr_dcm[patient][index+1].pixel_array
            np.save(subfolder_h + "/" + patient_str + "h_" + str(j) + ".npy", multi_array)
        print('patient ' + patient_str + " complete")

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

if __name__ == '__main__':
    patients_lr_dcm, patients_hr_dcm = read_all_data(config.data_path)
    pairings = match_dcm(patients_lr_dcm, patients_hr_dcm)
    lr_indices, hr_indices = split_pairs(pairings)
    select_and_save_data(patients_lr_dcm, lr_indices, patients_hr_dcm, hr_indices)