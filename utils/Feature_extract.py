from utils import conv
import numpy as np

# Example kernels
gaussian_kernel = np.array([[21, 31, 21],
                            [31, 48, 31],
                            [21, 31, 21]])/200*1.05
gaussian_kernel = gaussian_kernel*1.05
ori_gaussian_kernel = np.array([[21, 31, 21],
                                [31, 48, 31],
                                [21, 31, 21]])/256
# Define the list of kernels
kernel = [ori_gaussian_kernel]
def extract(image):
    k = 0
    conv1 = conv.convolve2d(image, kernel[k])
    conv2 = conv.convolve2d(conv1, kernel[k])
    pool1 = conv.max_pooling(conv2).astype(np.uint8)
    # block encoder 2
    conv3 = conv.convolve2d(pool1, kernel[k])
    conv4 = conv.convolve2d(conv3, kernel[k])
    pool2 = conv.max_pooling(conv4).astype(np.uint8)
    # block encoder 3
    conv5 = conv.convolve2d(pool2, kernel[k])
    conv6 = conv.convolve2d(conv5, kernel[k])
    conv7 = conv.convolve2d(conv6, kernel[k])
    pool3 = conv.max_pooling(conv7).astype(np.uint8)
    # block encoder 4
    conv8 = conv.convolve2d(pool3, kernel[k])
    conv9 = conv.convolve2d(conv8, kernel[k])
    conv10 = conv.convolve2d(conv9, kernel[k])
    pool4 = conv.max_pooling(conv10).astype(np.uint8)
    # block encoder 5
    conv11 = conv.convolve2d(pool4, kernel[k])
    conv12 = conv.convolve2d(conv11, kernel[k])
    conv13 = conv.convolve2d(conv12, kernel[k])
    pool5 = conv.max_pooling(conv13).astype(np.uint8)
    # block decoder 1
    # Generating mask for max unpooling
    pool_size = (2, 2)
    stride = (2, 2)
    mask1 = np.random.randint(0, pool_size[0]*pool_size[1], size=pool5.shape)
    # Max unpooling
    unpooled1 = conv.max_unpooling(pool5, mask1, stride, conv13.shape).astype(np.uint8)
    conv14 = conv.convolve2d(unpooled1, kernel[k])
    conv15 = conv.convolve2d(conv14, kernel[k])
    conv16 = conv.convolve2d(conv15, kernel[k])
    # block decoder 2
    mask2 = np.random.randint(0, pool_size[0]*pool_size[1], size=pool4.shape)
    unpooled2 = conv.max_unpooling(pool4, mask2, stride, conv10.shape).astype(np.uint8)
    conv17 = conv.convolve2d(unpooled2, kernel[k])
    conv18 = conv.convolve2d(conv17, kernel[k])
    conv19 = conv.convolve2d(conv18, kernel[k])
    # block decoder 3
    mask3 = np.random.randint(0, pool_size[0]*pool_size[1], size=pool3.shape)
    unpooled3 = conv.max_unpooling(pool3, mask3, stride, conv7.shape).astype(np.uint8)
    conv20 = conv.convolve2d(unpooled3, kernel[k])
    conv21 = conv.convolve2d(conv20, kernel[k])
    conv22 = conv.convolve2d(conv21, kernel[k])
    # block decoder 4
    mask4 = np.random.randint(0, pool_size[0]*pool_size[1], size=pool2.shape)
    unpooled4 = conv.max_unpooling(pool2, mask4, stride, conv4.shape).astype(np.uint8)
    conv23 = conv.convolve2d(unpooled4, kernel[k])
    conv24 = conv.convolve2d(conv23, kernel[k])
    # block decoder 5
    mask5 = np.random.randint(0, pool_size[0]*pool_size[1], size=pool1.shape)
    unpooled5 = conv.max_unpooling(pool1, mask5, stride, conv2.shape).astype(np.uint8)
    conv25 = conv.convolve2d(unpooled5, kernel[k])
    conv26 = conv.convolve2d(conv25, kernel[k])
    Feature_map = conv26
    return Feature_map