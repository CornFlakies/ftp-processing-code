import os
import ftp
import numpy as np 
import skimage as sk
import conversion_factor as cf
import matplotlib.pyplot as plt
from helper_functions import HelperFunctions as hp

class FtpClass:
    def get_wavelength(self, img, conv_factor):
        from scipy.fft import rfft, rfftfreq

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

    def run_ftp(self, yaml_file):
        '''
        Inputs:
        - calibration_folder : folder with a calibration image, using the qr code thingy
        - output_folder     : folder where the computed height profiles will be stored
        - background_folder : folder containing background images of the interrogation window
        - reference_folder  : image containing the undisturbed profile of the projection
        - input_folder      : folder containing the .npy files of the heightprofiles

        Outputs:
        - output folders containing the phase maps and height maps of all the measured data
        '''

        # Unpack yaml parameters
        cur_dir = os.path.dirname(__file__)
        calibration_folder  = hp.gen_path(cur_dir, yaml_file['INPUT']['CALIBRATION_IMG_PATH'])
        background_folder   = hp.gen_path(cur_dir, yaml_file['INPUT']['BACKROUND_IMG_PATH'])
        reference_folder    = hp.gen_path(cur_dir, yaml_file['INPUT']['REFERENCE_IMG_PATH'])
        input_folder        = hp.gen_path(cur_dir, yaml_file['INPUT']['INPUT_IMG_PATH'])
        image_mask_folder   = hp.gen_path(cur_dir, yaml_file['INPUT']['IMG_MASK_PATH'])
        output_folder       = hp.gen_path(cur_dir, yaml_file['OUTPUT']['OUTPUT_PATH'])
        
        first_img_idx = int(yaml_file['PROCESSING']['FIRST_IMG'])          
        last_img_idx  = int(yaml_file['PROCESSING']['LAST_IMG'])

        D = float(yaml_file['PARAMETERS']['SETUP']['DIST_PROJCAM']) 
        L = float(yaml_file['PARAMETERS']['SETUP']['DIST_CAM_FS'])  
        th = float(yaml_file['PARAMETERS']['FILTERING']['WINDOW_SIZE'])
        ns = float(yaml_file['PARAMETERS']['FILTERING']['WINDOW_TAPER'])

        # Create directory for height maps
        dir_height_maps = 'height_maps'
        path_dir_height_maps = os.path.join(output_folder, dir_height_maps, '')
        hp.create_output_folder(path_dir_height_maps)

        # Load image mask files
        if image_mask_folder is not None:
            image_mask_files, _ = hp.load_images(image_mask_folder, header='tiff')

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
       
        # Load in the calibration image, just the first one
        calib_paths, _ = hp.load_images(calibration_folder)
        img = np.load(calib_paths[0])
        _, conv_factor = cf.calculate_conversion_factor(img)
        print("Conversion factor: mm/px = {factor}".format(factor=conv_factor))

        # Get wavelength from ref image
        conv_factor *= 1E-1 #to centimeters
        p = self.get_wavelength(ref_img, conv_factor) #cm 
                
        # Create output folder for the height_maps
        hp.create_output_folder(output_folder)        

        # Load in the perturbed images
        image_paths, image_names = hp.load_images(input_folder)

        if (last_img_idx == -1): 
            print(f'running from images {image_names[first_img_idx]} to {image_names[last_img_idx]} ...')
            last_img_idx = len(image_paths)
        elif (((last_img_idx >= 0) & (last_img_idx > len(image_paths))) & (last_img_idx != -1)):
            print(Warning('Supplied last image index is larger than the provided dataset ...\n running through the whole dataset ..'))
            last_img_idx = len(image_paths)
        elif (last_img_idx < first_img_idx):
            raise Exception('First image index is larger than the last image index')
        else:
            print(f'running from images {image_names[first_img_idx]} to {image_names[last_img_idx - 1]} ...')
        frames = len(image_names[first_img_idx:last_img_idx])

        # Load in first image as reference for the unwrapping algorithm
        last_img = np.load(image_paths[first_img_idx]) - background_img
       
        #ADD Select pixel to unwrap the phase in time
        center_idx = tuple([250, 500])
        
        # Compute the first height and phase map
        last_phase_map = ftp.calculate_phase_diff_map_1D(last_img, ref_img, th, ns)
        height_map     = ftp.height_map_from_phase_map(last_phase_map, L, D, p)
       
        # Save height map
        filename = 'height_map_' + str(1).zfill(10) + '.npy'
        path = os.path.join(path_dir_height_maps, filename)
        np.save(path, height_map)

        # Run through the rest of the images
        for i, image in enumerate(image_paths[first_img_idx:last_img_idx]):
            # Loading mask, if provided
            if image_mask_folder is not None:
                mask = sk.io.imread(image_mask_files[i])
                plt.figure()
                plt.imshow(mask)
                mask[np.argwhere(mask == 0)] = 1 
                plt.figure()
                plt.imshow(mask)
                plt.show()
                

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

            # Setting the last phase map as the new one
            last_phase_map = new_phase_map
            
            if (i % int(frames / 10) == 0):
                hp.print(f'  {int(np.ceil(i / frames * 100))}% ...', mode='o')

