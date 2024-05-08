import numpy as np
from typing import List
from numpy.fft import rfft, fftshift, fft2
from scipy.signal import savgol_filter, welch
from dataclasses import dataclass
import os

import matplotlib.pyplot as plt

@dataclass
class Chunks:
    chunk_amount: int
    padding: int
    chunk_length_px: int
    xidx: List
    yidx: List

class HelperFunctions:
    @staticmethod
    def process_2d_fft(surface_profile, window = None): 
        # Subtract the mean
        surface_profile -= np.mean(surface_profile)
        
        # Apply a window
        if window is not None:
            surface_profile *= window        
        
        # Do the fourier transformed
        img_fft = np.abs(fft2(surface_profile))
        
        # Shift the k-space to set the origin in the center
        img_fft = fftshift(img_fft)
        
        return img_fft

    @staticmethod
    def process_kspace_fft(fft_profile, window = None):
        '''
        Process the fourier transform of an array which is of the (,frames) shape and already transformed in k-space
        this method is used to compute the dispersion relation of the dataset
        '''
        # Filter signal
        fft_profile -= np.mean(fft_profile)
        
        # Create dataset which is multiplied by a Hann window and fourier transform
        fft_profile = fft_profile.astype(np.float64)
        if window is not None:
            fft_profile *= window 
    
        # Do the fourier transform
        img_fft = np.abs(rfft(fft_profile)) 
         
        return img_fft

    @staticmethod
    def process_temporal_fft(surface_profile, window = None): 
        '''
        Helper function computes the fft of an array which is of the (,frames) shape (so a temporal fft)
        this method is used to compute the power spectrum of the dataset
        '''
        # Filter signal
        surface_profile -= np.mean(surface_profile)
        #surface_profile = savgol_filter(surface_profile, window_length=5, polyorder=3)
    
        # Create dataset which is multiplied by a Hann window and fourier transform
        surface_profile = surface_profile.astype(np.float64)
        if window is not None:
            surface_profile *= window
    
        img_fft = np.abs(rfft(surface_profile))
        return img_fft

    @staticmethod
    def process_temporal_welch(surface_profile, fs):
        # Remove the mean
        surface_profile -= np.mean(surface_profile)

        # Do welch transform
        freq, surface_profile = welch(surface_profile, fs, nperseg=512)

        return freq, surface_profile

    @staticmethod
    def gen_synthetic_img(img):
        '''
        Generate a synthetic image to prepare for the 2d spatial fourier transform
        the synthetic image adds copies of the provided image with a padding one fourth the qmage size
        '''
        img_size = np.size(img, axis=0)
        synth_size = int(1.5 * img_size)
        synth_padd = int(img_size / 4) 
        synth_img = np.zeros((synth_size, synth_size))

        # Flip horizontally
        img_hor   = np.flip(img, axis=1)
        hor_left  = img_hor[:,-synth_padd:]
        hor_right = img_hor[:, :synth_padd] 
        
        # Flip vertically
        img_ver = np.flip(img, axis=0)
        ver_top = img_ver[-synth_padd:, :]
        ver_bot = img_ver[:synth_padd, :]
        
        # The middle
        synth_img[synth_padd:-synth_padd, synth_padd:-synth_padd] = img
        
        # Left and right sides
        synth_img[synth_padd:-synth_padd, :synth_padd] = hor_left
        synth_img[synth_padd:-synth_padd, -synth_padd:] = hor_right

        # Top and bottom sides
        synth_img[:synth_padd, synth_padd:-synth_padd] = ver_top
        synth_img[-synth_padd:, synth_padd:-synth_padd] = ver_bot

        # The corners, top left, top right, bottom left, bottom right
        synth_img[:synth_padd, :synth_padd] = np.flip(np.flip(hor_left[:synth_padd, :]), axis=1)
        synth_img[:synth_padd, -synth_padd:] = np.flip(np.flip(hor_right[:synth_padd]), axis=1)
        synth_img[-synth_padd:, :synth_padd] = np.flip(np.flip(hor_left[-synth_padd:]), axis=1)
        synth_img[-synth_padd:, -synth_padd:] = np.flip(np.flip(hor_right[-synth_padd:]), axis=1)
        
        return synth_img


    @staticmethod 
    def grav_dispersion_sq(k, h0, g = 9.81):
        return g * k * np.tanh(k * h0)

    @staticmethod
    def cap_dispersion_sq(k, h0, gamma = 7.18E-2, rho = 998):
        return gamma * k**3 / rho * np.tanh(k * h0)

    @staticmethod
    def gravcap_dispersion_sq(k, h0, rho = 998, g = 9.81, gamma = 0.07):
        '''
        Returns omega**2 associated with provided k-values, according to 
        the complete gravity capillary dispersion relation
        '''
        return g * k * np.tanh(k * h0) * (1 + gamma * k**2 / (rho * g))
    
    @staticmethod
    def crop_image(img, nx1, nx2, ny1, ny2):
        return img[nx1:nx2, ny1:ny2]

    @staticmethod
    def square_img(img):
        shape = img.shape
        croplen = np.abs(shape[0] - shape[1]) // 2
        if shape[0] > shape[1]: 
            return img[croplen:-croplen]
        elif shape[1] > shape[0]:
            return img[:, croplen:-croplen]
        else:
            return img 

    @staticmethod
    def load_folders(input_folder):
        return [x[0] for x in os.walk(input_folder)][1:]

    @staticmethod
    def load_images(input_folder, header='npy'):
        '''
        Helper function used to load in .npy files)
        '''
        # Load in image paths
        images = np.sort(os.listdir(input_folder))

        # find npy files
        image_paths = []
        image_names = []
        for entry in images:
            split_filename = entry.split('.')
            if (split_filename[-1] == header):
                image_names.append(entry)
                image_paths.append(os.path.join(input_folder, entry))
        print(f'input directory {input_folder} contains {len(image_paths)} images ...')

        return np.sort(image_paths), np.sort(image_names)

    @staticmethod
    def create_output_folder(output_folder):
        '''
        Helper function used to generate an output folder for the processed data
        '''
        if not os.path.isdir(output_folder):
            print(f'Directory {output_folder} does not exist, making directory ...')
            os.makedirs(output_folder)
        elif os.listdir(output_folder) == []:
            print(Warning(f'Directory {output_folder} already exists, but does not contain files, continuing ...'))
        elif any([('.' in item) for item in os.listdir(output_folder)]):
            raise Exception(f'Directory {output_folder} already exists, and contains files, check that you do not overwrite anything!')
        else:
            print(Warning(f'Directory {output_folder} already exists, but does not contain files, so nothing will get overwritten, continuing ..'))
        
    @staticmethod
    def print(message, mode = 'n'):
        '''
        Helper function used to have more control over the print messages
        '''
        # modes are: newline 'n' (default), overwrite 'o'
        if mode == 'n':
            print(message, end='\n')
        elif mode == 'o':
            print(message, end='\r')
        else:
            print('Bruh, check je print statement')
