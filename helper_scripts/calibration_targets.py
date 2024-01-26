import cv2
import qrcode
import numpy as np
from io import BytesIO
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


def draw_checkerboard(pdf_filename, square_size_mm=5, num_squares=8, margin_mm=10):
    # Calculate the page size based on A4
    page_width, page_height = A4

    # Calculate the size of the checkerboard in points
    checkerboard_size = square_size_mm * num_squares * mm

    # Calculate the position with a specified margin
    x_offset = margin_mm * mm
    y_offset = page_height - margin_mm * mm - checkerboard_size

    # Create a PDF canvas with A4 page size
    c = canvas.Canvas(pdf_filename, pagesize=A4)

    # Draw the checkerboard pattern
    for i in range(num_squares):
        for j in range(num_squares):
            if (i + j) % 2 == 0:
                c.setFillGray(0)
            else:
                c.setFillGray(1)
            x = x_offset + i * square_size_mm * mm
            y = y_offset + j * square_size_mm * mm
            c.rect(x, y, square_size_mm * mm, square_size_mm * mm, fill=True, stroke=0)

    # Save the PDF
    # c.save()
    return c


def generate_qrcode(info, size_cm):
    # Calculate the size in points (1 cm = 28.3464567 points)
    size = int(size_cm * 28.3464567)

    qr = qrcode.QRCode(
        version=5,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,  # You can adjust this if needed
        border=0,
    )
    qr.add_data(info)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Resize the image to the desired physical size
    # img = img.resize((size, size))

    return img


def read_qr_info_from_image(image_path):

    # Read the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the QRCodeDetector
    qr_detector = cv2.QRCodeDetector()

    # Detect and decode the QR code
    retval, decoded_info, points, array = qr_detector.detectAndDecodeMulti(gray)

    # Grab first string from decoded info
    # (as only one QR code is expected to be found in the image)
    decoded_string = decoded_info[0]

    # Split the string into key-value pairs
    pairs = [item.strip() for item in decoded_string.split(",")]

    # Create a dictionary from key-value pairs
    decoded_info_dict = {}
    for pair in pairs:
        key, value = pair.split(":")
        decoded_info_dict[key.strip()] = value.strip()
    return decoded_info_dict


# para generar el qr
# info = "L: 5, U: mm, N: 10"
# pdf_path = "output.pdf"
# add_qrcode_to_pdf(info, pdf_path, size_cm=5)
# print(f"PDF with QR code created at {pdf_path}")


def generate_calibration_target(
    pdf_filename,
    square_size_mm=5,
    num_squares=15,
    add_qr_code=True,
    margin_mm=7.5 * mm,
):
    """
    Generates a pdf with a target and qr info.
    """

    # Draw checkerboard with specified dimensions
    c = draw_checkerboard(
        pdf_filename,
        square_size_mm=square_size_mm,
        num_squares=num_squares,
        margin_mm=margin_mm,
    )

    if add_qr_code is True:

        # Generate information string that is coded into the QR code
        qr_info = f"square_size_mm: {square_size_mm}, num_squares: {num_squares}"

        # Generate QR code with specified dimensions (50x50 mm)
        qr_size_mm = 50
        qr_size = qr_size_mm * mm
        qr_code_img = generate_qrcode(qr_info, qr_size)

        # Set position of QR code in the page, to be to the right of the
        # checkerboard, centered vertically with respect to the checkerboard
        page_width, page_height = A4
        checkerboard_size_mm = square_size_mm * num_squares
        x_offset = checkerboard_size_mm * mm + 2 * margin_mm * mm
        y_offset = (
            page_height
            - checkerboard_size_mm * mm
            - margin_mm * mm
            + (checkerboard_size_mm - qr_size_mm) * mm / 2
        )

        # Set the width and height of the image
        c.drawInlineImage(
            qr_code_img,
            x_offset,
            y_offset,
            width=qr_size,
            height=qr_size,
        )

    # Add text below the checkerboard
    baseline_text_height = 12
    c.drawString(
        margin_mm * mm,
        page_height
        - checkerboard_size_mm * mm
        - margin_mm * mm
        - 2 * baseline_text_height,
        f"square_size_mm: {square_size_mm} mm // num_squares: {num_squares}",
    )

    # Save to file
    c.save()


# Specify the filename for the PDF
pdf_filename = "checkerboard_top_left_4.pdf"

# Generate the checkerboard PDF with 5 mm squares in the top-left corner
generate_calibration_target(pdf_filename, square_size_mm=2.5, num_squares=35)

print(f"Checkerboard saved as {pdf_filename}.")
