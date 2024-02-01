import numpy as np
import os
import matplotlib.pyplot as plt 
import ftp
from helper_functions import HelperFunctions as hp

class FtpClass:
    def __init__(self, sample_size = 30):
        self.mean = np.zeros(sample_size)
        self.std  = np.zeros(sample_size)
        self.height_sample = np.zeros(sample_size)
    
    def get_wavelength(self, img, conv_factor):
        from scipy.fft import rfft, rfftfreq

        rows, _ = img.shape
        center_line = img

        fft_center_line = np.zeros(np.size(center_line, axis=1) // 2 + 1)
        i = 0 
        for ctr in center_line:
            fft_center_line += np.real(rfft(ctr))
            i += 1

        fft_center_line /= i
        kspace = rfftfreq(np.size(center_line, axis=1), d=conv_factor)
        cutoff = 10
        idx = np.argmax(fft_center_line[cutoff:])
        p_value = 1/kspace[cutoff:][idx]
        print(f'from reference image a carrier wavelength of {p_value} cm was extracted ...')
        
        return p_value

    def run_ftp(self, output_folder, background_folder, reference_folder, input_folder):
        '''
        Inputs: 
        - output_folder     : folder where the computed height profiles will be stored
        - background_folder : folder containing background images of the interrogation window
        - reference_folder  : image containing the undisturbed profile of the projection
        - input_folder      : folder containing the .npy files of the heightprofiles

        Outputs:
        - output folders containing the phase maps and height maps of all the measured data
        '''        
        # Create general output folder
        hp.create_output_folder(output_folder)
        
        # Create directory for height maps
        dir_height_maps = 'height_maps'
        path_dir_height_maps = os.path.join(output_folder, dir_height_maps, '')
        hp.create_output_folder(path_dir_height_maps)

    #    # Create directory for phase maps
    #    dir_phase_maps = 'phase_maps'
    #    path_dir_phase_maps = os.path.join(output_folder, dir_phase_maps, '')
    #    hp.create_output_folder(path_dir_phase_maps)
        
        # Creating a background image by averaging a few frames from the background
        print('generating background image ...')
        image_paths, _ = hp.load_images(background_folder)
        background_img = np.zeros(np.load(image_paths[0]).shape)
        for image in image_paths:
            background_img += np.load(image)
        background_img /= len(image_paths)

        # Load in the reference image, subtracting background
        print('generating reference image ...')
        image_paths, _ = hp.load_images(reference_folder)
        reference_img = np.zeros(np.load(image_paths[0]).shape)
        for image in image_paths:
            reference_img += np.load(image)
        reference_img /= len(image_paths)
        ref_img = reference_img - background_img

        # Load in the perturbed images
        image_paths, _ = hp.load_images(input_folder)
        frames = len(image_paths)
        
        #ADD Define experimental params manually
        L = 80 #cm 
        D = 39 #cm 
        conv_factor = 0.303 * 1E-1 #cm/px
        p = self.get_wavelength(ref_img, conv_factor) #cm 

        #ADD Define parameters for the processing code
        th = .3
        ns = .7

        #ADD Load in first image as reference for the unwrapping algorithm
        last_img = np.load(image_paths[0]) - background_img
        
        center_idx = tuple([900, 500])
        
        # Compute the first height and phase map
        last_phase_map = ftp.calculate_phase_diff_map_1D(last_img, ref_img, th, ns)
        height_map     = ftp.height_map_from_phase_map(last_phase_map, L, D, p)
        
        # Save height and phase maps
        filename = 'height_map_' + str(1).zfill(10) + '.npy'
        path = os.path.join(path_dir_height_maps, filename)
        np.save(path, height_map)

    #    filename = 'phase_map_' + str(1).zfill(10) + '.npy'
    #    path = os.path.join(path_dir_phase_maps, filename)
    #    np.save(path, last_phase_map)
        
        # Run through the rest of the images
        for i, image in enumerate(image_paths[1:]):
            # Loading image and finding phase map
            img = np.load(image) - background_img
            new_phase_map = ftp.calculate_phase_diff_map_1D(img, ref_img, th, ns)
            
            # Unwrapping the phase in time
            appended_lst = np.append(last_phase_map[center_idx], new_phase_map[center_idx])
            shift = (np.unwrap(appended_lst) - appended_lst)[1] * np.ones(new_phase_map.shape)
            new_phase_map = new_phase_map + shift

            # Generating height map
            height_map = ftp.height_map_from_phase_map(new_phase_map, L, D, p)            

            # Saving files
            filename = 'height_map_' + str(i + 2).zfill(10) + '.npy'
            path = os.path.join(path_dir_height_maps, filename)
            np.save(path, height_map)

    #        filename = 'phase_map_' + str(i + 2).zfill(10) + '.npy'
    #        path = os.path.join(path_dir_phase_maps, filename)
    #        np.save(path, new_phase_map)
             
            # Setting the last phase map as the new one
            last_phase_map = new_phase_map
            
            if (i % int(frames / 10) == 0):
                hp.print(f'  {int(np.ceil(i / frames * 100))}% ...', mode='o')
