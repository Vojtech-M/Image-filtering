"""
Module for image filtering operations using convolution.
"""

import numpy as np

def apply_filter(image: np.array, kernel: np.array) -> np.array:
    """
    Apply a 2D convolution operation on the input image using the input kernel.

    Args:
        image (np.array): Input image as a NumPy array (2D for grayscale, 3D for RGB).
        kernel (np.array): Convolution kernel as a square NumPy array (2D).

    Returns:
        np.array: Filtered image as a NumPy array.

    Raises:
        AssertionError: If input image or kernel does not meet required dimensions.
    """

    # Validate dimensions of image and kernel
    assert image.ndim in [2, 3], "Image must be 2D (grayscale) or 3D (RGB)"
    assert kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1], "Kernel must be a square 2D array"

    padding = kernel.shape[0] // 2

    if image.ndim == 2:  # Grayscale image
        height, width = image.shape
        padded_image = np.pad(image, padding, mode='constant')
        filtered_image = np.zeros_like(image, dtype=np.float32)
    else:  # RGB image
        height, width, channels = image.shape
        padded_image = np.pad(image, [(padding, padding), (padding, padding), (0, 0)], mode='constant')
        filtered_image = np.zeros_like(image, dtype=np.float32)

    # Perform 2D convolution
    for y in range(height):
        for x in range(width):
            if image.ndim == 2:  # Grayscale case
                patch = padded_image[y:y + kernel.shape[0], x:x + kernel.shape[1]]
                filtered_image[y, x] = np.sum(patch * kernel)
            else:  # RGB case
                for c in range(channels):  # Loop through RGB channels
                    patch = padded_image[y:y + kernel.shape[0], x:x + kernel.shape[1], c]
                    filtered_image[y, x, c] = np.sum(patch * kernel)

    # Clip and convert filtered image int
    filtered_image = np.clip(filtered_image, 0, 255)
    filtered_image = filtered_image.astype(np.uint8)

    return filtered_image
