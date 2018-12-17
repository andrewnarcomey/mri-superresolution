import pydicom
import os
import math
import numpy as np
import png
from config import config
import tensorlayer as tl
import matplotlib.pyplot as plt

## Extra functions used in main.py ##

def load_info(path):
    data = {}
    all_patient_folders = tl.files.load_folder_list(path=path)
    all_patient_folders = [folder for folder in all_patient_folders if folder[len(path)+1:-1] in str(config.patient_numbers)]
    all_patient_folders.sort(key=tl.files.natural_keys)
    nfolders = len(all_patient_folders)
    if nfolders % 2 != 0: nfolders -= 1
    print("[*] Unpackaging patient files (" + str(nfolders) + ")")
    for i in range(0,nfolders,2):
        patient_str = str(config.patient_numbers[int(i/2)])
        print("Loading patient: " + patient_str)
        hr_folder = all_patient_folders[i]
        print(hr_folder)
        lr_folder = all_patient_folders[i+1]
        print(lr_folder)
        
        hr_data = []
        hr_files = tl.files.load_file_list(path=hr_folder, regx='.*.npy', printable=False, keep_prefix=False)
        hr_files.sort(key=tl.files.natural_keys)
        for file in hr_files:
            hr_data.append(tl.files.load_npy_to_any(path=hr_folder + "/", name=file))

        lr_data = []
        lr_files = tl.files.load_file_list(path=lr_folder, regx='.*.npy', printable=False, keep_prefix=False)
        lr_files.sort(key=tl.files.natural_keys)
        for file in lr_files:
            lr_data.append(tl.files.load_npy_to_any(path=lr_folder + "/", name=file))

        data[patient_str] = [lr_data, hr_data]
    return data


def convert_to_png(path):
    ds = pydicom.dcmread(path)
    shape = ds.pixel_array.shape
    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)
    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)
    # Write the PNG file
    destination = 'test.png'
    with open(destination, 'wb') as png_file:
        w = png.Writer(shape[1], shape[0], greyscale=True)
        w.write(png_file, image_2d_scaled)

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


def plot_iterative_losses(losses):
    n = list(range(len(losses['mse_loss'])))
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(n, losses['d_loss'])
    axarr[0].set_title('Discriminator Loss')
    axarr[1].plot(n, losses['g_loss'])
    axarr[1].set_title('Generator Loss')
    axarr[1].set(xlabel='Iteration')
    for ax in axarr.flat:
        ax.set(ylabel='Loss')
    plt.savefig("iter_main_losses.png")

    plt.figure()
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(n, losses['mse_loss'])
    axarr[0].set_title('MSE Loss')
    axarr[1].plot(n, losses['adv_loss'])
    axarr[1].set_title('Adversarial Loss')
    axarr[2].plot(n, losses['vgg_loss'])
    axarr[2].set_title('VGG Loss')
    axarr[2].set(xlabel='Iteration')
    for ax in axarr.flat:
        ax.set(ylabel='Loss')
    plt.savefig("iter_all_losses.png")


def plot_total_losses(losses):
    n = list(range(len(losses['d_loss'])))
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(n, losses['d_loss'])
    axarr[0].set_title('Discriminator Loss')
    axarr[1].plot(n, losses['g_loss'])
    axarr[1].set_title('Generator Loss')
    axarr[1].set(xlabel='Epoch')
    for ax in axarr.flat:
        ax.set(ylabel='Loss')
    plt.savefig('epoch_total_losses')

def mse(gen, target):
    err = np.sum((gen.astype("float") - target.astype("float")) ** 2)
    err /= float(gen.shape[0] * gen.shape[1] * gen.shape[2])
    return err
    