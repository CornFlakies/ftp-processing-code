import cv2
import qrcode
import numpy as np
from scipy.special import comb
from itertools import combinations
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def load_image_to_grayscale(imagefile):

    # Load the image
    image = cv2.imread(imagefile)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def detect_and_read_qr(gray, plot=False):
    """
    Detects and read the contents of a QR code in the image.
    The contents are returned in a Python dictionary.
    """

    # QR DETECTION
    # Initialize the QRCodeDetector
    qr_detector = cv2.QRCodeDetector()

    # Detect and decode the QR code
    decoded_string, points, array = qr_detector.detectAndDecode(gray)

    # Split the string into key-value pairs
    pairs = [item.strip() for item in decoded_string.split(",")]

    # Create a dictionary from key-value pairs
    decoded_info_dict = {}
    for pair in pairs:
        key, value = pair.split(":")
        decoded_info_dict[key.strip()] = value.strip()

    # Optional: plot if is True (False by default)
    if plot is True:
        points = points.reshape(4, 2)
        points = np.vstack((points, points[0, :]))

        plt.figure()
        plt.imshow(gray)
        plt.set_cmap("gray")
        plt.plot(points[:, 0], points[:, 1], "b-")
        plt.xlabel("px")
        plt.ylabel("px")
        plt.title("QRdata: {data}".format(data=decoded_info_dict))

    return decoded_info_dict


def compute_distances_between_points(corners):
    """
    Calculates distances between points.
    """

    # Number of internal corners
    num_internal_corners = int(np.sqrt(corners.shape[0]))

    # Compute distances between adjacent neighbors, in pixels
    # using all combinations of distances available
    # and generate array with corresponding indices
    all_elements = list(range(num_internal_corners**2))
    combinations_list = list(combinations(all_elements, 2))
    unit_corners = [
        (x, y) for x in range(num_internal_corners) for y in range(num_internal_corners)
    ]
    dists = np.zeros(len(combinations_list))
    for i, elem in enumerate(combinations_list):
        corner0 = corners[elem[0]]
        corner1 = corners[elem[1]]
        ucorner0 = np.array(unit_corners[elem[0]])
        ucorner1 = np.array(unit_corners[elem[1]])
        d = np.linalg.norm(corner0 - corner1)
        du = np.linalg.norm(ucorner0 - ucorner1)
        dists[i] = d / du

    return dists

def calculate_conversion_factor(image, plot=False):

    image = (image / image.max() * 255).astype(np.uint8)
    _, gray = cv2.threshold(image, 75, 255, cv2.THRESH_BINARY)
    
    # Detect and read QR code info
    qrdata = detect_and_read_qr(gray, plot=plot)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Get the number of squares in the checkerboard
    # (assuming the checkerboard is a square)
    num_squares = int(qrdata["num_squares"])
    num_internal_corners = num_squares - 1
    internal_corners_shape = (num_internal_corners, num_internal_corners)

    # Find the chessboard corners in the image
    ret, corners = cv2.findChessboardCorners(gray, internal_corners_shape, None)

    # If corners are indeed found, continue
    if ret:
        # Reshape the corners output for convenience.
        corners = corners.reshape(corners.shape[0], 2)

        # Calculate distances between internal corners of a
        # checkerboard of prescribed size.
        dists = compute_distances_between_points(corners)

        # Calculate the mm_per_px conversion factor
        pixel_distance_on_checkerboard_image = np.mean(dists)
        physical_distance_on_checkerboard_print = float(qrdata["square_size_mm"])
        mm_per_px = (
            physical_distance_on_checkerboard_print
            / pixel_distance_on_checkerboard_image
        )

        # Optional: draw the corners' positions and number them in order.
        if plot is True:
            plt.plot(corners[:, 0], corners[:, 1], "ro")
            for i in range(corners.shape[0]):
                plt.annotate(str(i), (corners[i, 0], corners[i, 1]), color="r")
        return dists, mm_per_px
    else:
        raise Exception("Error: checkerboard not found in the image.")

if __name__=='__main__': 
    import argparse

    # Load in image paths
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('input_folder', type=str)
    args = argparser.parse_args()

    image = np.load(args.input_folder)

    dists, mm_per_px = calculate_conversion_factor(image, plot=True)
    print("Conversion factor: mm/px = {factor}".format(factor=mm_per_px))

