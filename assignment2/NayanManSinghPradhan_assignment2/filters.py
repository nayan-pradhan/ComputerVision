# Done and Submitted by: Nayan Man Singh Pradhan

import numpy as np
from scipy import signal
from scipy import misc


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    ### YOUR CODE HERE

    # First, we need to address the edge of the image
    # I will use zero padding for solving this

    # First make a zero matrix with extra dimentions
    padded_img = np.zeros((Hi+Hk-1, Wi+Wk-1)) 

    padded_img[:-Hk//2,:-Wk//2] = image 

    for l in range(Hi):
        for b in range(Wi):
            for i in range(Hk):
                for j in range(Wk):
                    out[l,b] += padded_img[l-i+1, b-j+1] * kernel[i,j] 

    pass
    ### END YOUR CODE

    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.pad(image, ((pad_height, pad_height),(pad_width, pad_width)), 'constant', constant_values = 0)
    pass
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE

    # First, start by flipping the kernel (for convolution)
    temp_flipped_ker = np.fliplr(kernel)
    flipped_ker = np.flipud(temp_flipped_ker)

    # Initializing some variables
    i = Hk//2
    j = Wk//2

    # Zero padding
    padded_img = zero_pad(image, i, j)

    # Making changes on copy of image
    out = np.copy(out)
    
    # run the loop
    for m in range (i, i + Hi):
        for n in range (j, j + Wi):
            temp_box = padded_img[m-i:m+i+1, n-j:n+j+1]
            out[m-i, n-j] = (temp_box * flipped_ker).sum()

    pass
    ### END YOUR CODE

    return out


# Bonus Function
def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    out = signal.convolve2d(image, kernel, boundary='fill', mode = 'same')
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hi, Wi = f.shape # For larger image
    Hk, Wk = g.shape # For item we need to search for

    # Since this is correlation, we dont need to flip kernel ie. g

    # Initializing some variables
    i = Hk//2
    j = Wk//2

    # Add padding to image
    padded_img = zero_pad(f, i, j)

    # Making changes in copy of image
    out = np.copy(f)

    # Looping
    for m in range(i, Hi + i):
        for n in range(j, Wi + j):
            temp_box = padded_img[m-i:m+i, n-j:n+j+1]
            out[m-i, n-j] = (temp_box * g).sum()

    pass
    ### END YOUR CODE

    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    mean_g = np.mean(g)
    new_g = g - mean_g
    out = cross_correlation(f, new_g)
    pass
    ### END YOUR CODE

    return out


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    
    # dimension of image and kernel
    Hi, Wi = f.shape
    Hk, Wk = g.shape

    # Initializing some variabels
    i = Hk//2
    j = Wk//2

    # Padding original image
    padded_img = zero_pad(f, i, j)

    # Making copy of image to work on
    out = np.copy(f)

    for m in range (i, Hi+i):
        for n in range (j, Wi+j):
            temp = padded_img[m-i:m+i, n-j:n+j+1]
            out[m-i, n-j] = ((1/np.std(temp) * (temp - np.mean(temp)) * (1/np.std(g)) * (g - np.mean(g)))).sum()

    pass
    ### END YOUR CODE

    return out
