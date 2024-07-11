import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# def convolve2d(image, kernel, stride=(1, 1), padding=0):
#     # Convert the input image to a NumPy array
#     image_np = np.array(image)
    
#     # Add padding to the input image
#     image_padded = cv2.copyMakeBorder(image_np, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    
#     # Perform convolution
#     output = cv2.filter2D(image_padded, -1, kernel, delta=0, borderType=cv2.BORDER_CONSTANT)
    
#     # Apply stride
#     output = output[::stride[0], ::stride[1]]
    
#     return output

def convolve2d(image, kernel, stride=(1, 1), padding=0):
    # Convert the input image to a NumPy array
    image_np = np.array(image)
    
    # Add padding to the input image
    image_padded = cv2.copyMakeBorder(image_np, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    
    # Perform convolution
    output = cv2.filter2D(image_padded, -1, kernel, delta=0, borderType=cv2.BORDER_CONSTANT)
    
    # Apply ReLU activation function
    output = np.maximum(output, 0)
    
    # Apply stride
    output = output[::stride[0], ::stride[1]]
    
    return output

def max_pooling(input_data, pool_size=(2, 2), stride=(2, 2)):
    input_height, input_width, input_channels = input_data.shape
    pool_height, pool_width = pool_size
    stride_height, stride_width = stride

    output_height = (input_height - pool_height) // stride_height + 1
    output_width = (input_width - pool_width) // stride_width + 1

    pooled_output = np.zeros((output_height, output_width, input_channels))

    for h in range(output_height):
        for w in range(output_width):
            h_start = h * stride_height
            h_end = h_start + pool_height
            w_start = w * stride_width
            w_end = w_start + pool_width
            for c in range(input_channels):
                patch = input_data[h_start:h_end, w_start:w_end, c]
                pooled_output[h, w, c] = np.max(patch)

    return pooled_output

def max_unpooling(input_data, mask, stride=(2, 2), output_shape=None):
    input_height, input_width, input_channels = input_data.shape
    stride_height, stride_width = stride
    output_height, output_width, _ = mask.shape

    if output_shape is None:
        output_shape = (input_height, input_width, input_channels)

    unpooled_output = np.zeros(output_shape)

    for h in range(output_height):
        for w in range(output_width):
            for c in range(input_channels):
                h_start = h * stride_height
                h_end = h_start + stride_height
                w_start = w * stride_width
                w_end = w_start + stride_width
                patch = unpooled_output[h_start:h_end, w_start:w_end, c]
                idx = np.unravel_index(mask[h, w, c], patch.shape)
                patch[idx] = input_data[h, w, c]

    return unpooled_output