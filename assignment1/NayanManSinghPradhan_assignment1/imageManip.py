# Submitted By: Nayan Man Singh Pradhan

import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path) # using io.imread() function
    pass
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = 0.5 * (image**2) # making the image dimmer
    pass
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### YOUR CODE HERE
    out = color.rgb2gray(image) # converting rgb to grey
    pass
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    out = image.copy() # need copy of image to manipulate it
    # 3rd attribute of image represents R,G,B value as 0,1,2 respectively
    if (channel == 'R'):
        out[:,:,0] = 0 # red color is removed 
    elif (channel == 'G'):
        out[:,:,1] = 0 # green color is removed
    elif (channel == 'B'):
        out[:,:,2] = 0 # blue color is removed
    pass
    ### END YOUR CODE

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    out = lab.copy() # need copy of image to manipulate it

    if (channel == 'L'):
        out = lab[:,:,0] # only L
    elif (channel == 'A'):
        out = lab[:,:,1] # only A
    elif (channel == 'B'):
        out = lab[:,:,2] # only B
    pass
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    out = hsv.copy() # need copy of image to manipulate it

    if (channel == 'H'):
        out = hsv[:,:,0] # only H
    elif (channel == 'S'):
        out = hsv[:,:,1] # only S
    elif (channel == 'V'):
        out = hsv[:,:,2] # only V
    pass
    ### END YOUR CODE

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### YOUR CODE HERE
    image1_copy = rgb_exclusion(image1, channel1)
    image2_copy = rgb_exclusion(image2, channel2)
    out = np.concatenate((image1_copy[:,:image2_copy.shape[1]//2], image2_copy[:,image2_copy.shape[1]//2:]), axis = 1)
    # using np.concatenate to concatinate two images
    # image.shape[0] : no of columns
    # image.shape[1] : no of rows
    # we divide using // so that we get an integer value
    # axis = 1 : acts on all columns in each row
    pass
    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    top_left = rgb_exclusion(image, 'R') 
    top_right = dim_image(image)
    # bottom_left = brighten_image(image) # made an additional function
    bottom_left = image**0.5
    bottom_right = rgb_exclusion(image, 'R')

    # made slices according to question
    n_tl = top_left[:top_left.shape[0]//2, :top_left.shape[1]//2]
    n_tr = top_right[:top_right.shape[0]//2, top_right.shape[1]//2:]
    n_bl = bottom_left[bottom_left.shape[0]//2:, :bottom_left.shape[1]//2]
    n_br = bottom_right[bottom_right.shape[0]//2:, bottom_right.shape[1]//2:]
    # using np.concatenate to concatinate two images
    # image.shape[0] : no of columns
    # image.shape[1] : no of rows
    # we divide using // so that we get an integer value

    # I implemented two ways of performing the final concatination:
    
    # 1.
    top = np.concatenate((n_tl, n_tr), axis = 1)
    bottom = np.concatenate((n_bl, n_br), axis = 1)
    out = np.concatenate((top, bottom), axis = 0)
    
    # 2.
    # left = np.concatenate((n_tl, n_bl), axis = 0)
    # right = np.concatenate((n_tr, n_br), axis = 0)
    # out = np.concatenate((left, right), axis = 1)

    # axis = 1 : acts on all columns in each row
    # axis = 0 : acts on all rows in each column

    pass
    ### END YOUR CODE

    return out

# # custom function to make image brighter for mix_quadrants function
# def brighten_image(image):
#     """Change the value of every pixel by following

#                         x_n = x_p^0.5

#     where x_n is the new value and x_p is the original value.

#     Args:
#         image: numpy array of shape(image_height, image_width, 3).

#     Returns:
#         out: numpy array of shape(image_height, image_width, 3).
#     """

#     out = None

#     out = image**0.5 # making the image brighter
#     pass
#     return out