import numpy as np
import numpy.ma as ma
from scipy import signal
from skimage.restoration import unwrap_phase as unwrap
import matplotlib.pyplot as plt

def calculate_phase_diff_map_1D(dY, dY0, th, ns, mask_for_unwrapping=None):
    """
    # Basic FTP treatment.
    # This function takes a deformed and a reference image and calculates the phase difference map between the two.
    # 
    # INPUTS:
    # dY	= deformed image
    # dY0	= reference image
    # ns	= size of gaussian filter
    # th    = threshold of the filter
    # 
    # OUTPUT:
    # dphase = phase difference map between images
    """
    ny, nx = np.shape(dY)
    phase0 = np.zeros([nx, ny])
    phase = np.zeros([nx, ny])

    #plt.figure()
    #plt.imshow(dY0)
    #plt.figure()
    #plt.imshow(dY)
    #plt.show()
    
    for lin in range(nx):
        fY0 = np.fft.fft(dY0[lin, :])
        fY = np.fft.fft(dY[lin, :])
        
        dfy = 1.0 / ny
        #fy = np.arange(dfy, 1, dfy)

        imax = np.argmax(np.abs(fY0[9 : int(np.floor(nx / 2))]))
        ifmax = imax + 9
      
        HW = np.round(ifmax * th)
        W = 2 * HW
        win = signal.windows.tukey(int(W), ns)

        gaussfilt1D = np.zeros(nx)
        gaussfilt1D[int(ifmax - HW - 1) : int(ifmax - HW + W - 1)] = win

        # Multiplication by the filter
        Nfy0 = fY0 * gaussfilt1D
        Nfy = fY * gaussfilt1D

        ##plt.plot(gaussfilt1D)
        #plt.plot(gaussfilt1D[9:] * 5000)
        #plt.plot(np.abs(fY0[9:]))
        #plt.plot(np.abs(fY[9:]))
        #plt.figure()
        #plt.plot(np.abs(Nfy))
        #plt.show()
  
        # Inverse Fourier transform of both images
        Ny0 = np.fft.ifft(Nfy0)
        Ny = np.fft.ifft(Nfy)

        phase0[lin, :] = np.angle(Ny0)
        phase[lin, :] = np.angle(Ny)

    # 2D-unwrapping is available with masks (as an option), using 'unwrap' library
    # unwrap allows for the use of wrapped_arrays, according to the docs:
    # "[...] in this case masked entries are ignored during the phase unwrapping process. This is useful if the wrapped phase data has holes or contains invalid entries. [...]"

    if mask_for_unwrapping is not None:
        phase0 = ma.masked_array(phase0, mask=mask_for_unwrapping)
        phase = ma.masked_array(phase, mask=mask_for_unwrapping)
    
    phase0 = unwrap(phase0)
    phase = unwrap(phase)

    # Definition of the phase difference map
    dphase = phase - phase0
    return dphase

def height_map_from_phase_map(dphase, L, D, p):
    """
    Converts a phase difference map to a height map using the phase to height
    relation.

    INPUTS:
         dphase    = phase difference map (already unwrapped)
         L         = distance between the reference surface and the plane of the entrance  pupils
         D         = distance between centers of entrance pupils
         p         = wavelength of the projected pattern (onto the reference surface)

         OUTPUT:
            h         = height map of the surface under study
    """
    return -L * dphase / (2 * np.pi / p * D - dphase)
