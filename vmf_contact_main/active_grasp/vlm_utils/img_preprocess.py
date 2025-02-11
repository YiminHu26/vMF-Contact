import cv2
import numpy as np

def crop_max_circle(image):
    """
    Crop the largest possible circle from the center of the image.
    """
    height, width = image.shape[:2]
    radius = min(height, width) // 2  # Maximum radius possible

    # Compute the center of the image
    center_x, center_y = width // 2, height // 2

    # Create a circular mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, thickness=-1)

    # Apply mask
    result = cv2.bitwise_and(image, image, mask=mask)

    # Crop the square region containing the circle
    cropped_circle = result[center_y - radius:center_y + radius, center_x - radius:center_x + radius]

    # Create transparent background for cropped circle
    b, g, r = cv2.split(cropped_circle)
    alpha = mask[center_y - radius:center_y + radius, center_x - radius:center_x + radius]
    cropped_circle_rgba = cv2.merge([b, g, r, alpha])

    return cropped_circle_rgba

def rotate_circle(image, angle):
    """
    Rotate only the circular cropped part, keeping the background transparent.
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Compute rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate only the cropped circle, keeping transparency
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

    return rotated_image

def crop_max_and_rotate(image, angle, size=(700, 700)):
    """
    Crop the largest possible circle from the center of the image and rotate it.
    """
    cropped_circle = crop_max_circle(image)
    rotated_circle = rotate_circle(cropped_circle, angle)

    return cv2.resize(rotated_circle, size, interpolation=cv2.INTER_LINEAR)[..., :3]