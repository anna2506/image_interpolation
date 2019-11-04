import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def load_image(path):
	return plt.imread(path)  #function imported from matplotlib, loading the picture as numpy array

#function to save image
def save_image(path, img):
    plt.imsave(path, img)

def rgb2bayer(image):
    """Convert image to bayer pattern:
    [B G]
    [G R]

    Args:
        image: Input image as (H,W,3) numpy array

    Returns:
        bayer: numpy array (H,W,3) of the same type as image
        where each color channel retains only the respective 
        values given by the bayer pattern
    """
    assert image.ndim == 3 and image.shape[-1] == 3
    # otherwise, the function is in-place
    bayer = image.copy()
    w, h, _ = bayer.shape  #the width and the height of the image
    red = np.zeros((w, h, 3))  #an array with red filter
    green = np.zeros((w, h, 3))  #an array with green filter
    blue = np.zeros((w, h, 3))  #an array with blue filter
    for i in range(w):
        for j in range(h):
            #even row and even column element
            if  i%2 == 0 and j %2 == 0:
                # putting the blue value
                blue[i, j, 0] = bayer[i, j, 0] * 0
                blue[i, j, 1] = bayer[i, j, 1] * 0
                blue[i, j, 2] = bayer[i, j, 2] * 1
                #uneven raw and uneven column element
            elif i%2 != 0 and j % 2 != 0:
                #putting the red value
                red[i, j, 0] = bayer[i, j, 0] * 1
                red[i, j, 1] = bayer[i, j, 1] * 0
                red[i, j, 2] = bayer[i, j, 2] * 0
                #other cases
            else:
                #putting the green value
                green[i, j, 0] = bayer[i, j, 0] * 0
                green[i, j, 1] = bayer[i, j, 1] * 1
                green[i, j, 2] = bayer[i, j, 2] * 0
    #sum of red, green and blue arrays
    bayer = red + green + blue
    assert bayer.ndim == 3 and bayer.shape[-1] == 3
    return bayer

def bayer2rgb(bayer):
    """Interpolates missing values in the bayer pattern.
    Note, green uses bilinear upsampling; red and blue nearest neighbour.

    Args:
        bayer: 2D array (H,W,C) of the bayer pattern
    
    Returns:
        image: 2D array (H,W,C) with missing values interpolated
        green_K: 2D array (3, 3) of the interpolation kernel used for green channel
        redblue_K: 2D array (3, 3) using for interpolating red and blue channels
    """
    assert bayer.ndim == 3 and bayer.shape[-1] == 3
    w, h, _ = bayer.shape  #the width and the height of the image
    # 3 filters
    red_filter = bayer[:, :, 0]
    green_filter = bayer[:, :, 1]
    blue_filter = bayer[:, :, 2]
    red_i, red_j, blue_i, blue_j, delta_i, delta_j = 0, 0, 0, 0, 0, 0
    #we check if the pattern of bayer image is as it was at the beginning or maybe we need to make a bias
    for i in range(w):
        for j in range(h):
            if(red_filter[i][j] != 0):
                if i%4 < 2 and j%4 < 2:
                    delta_i, delta_j = 2, 2
                elif i%4 > 1 and j%4 < 2:
                    delta_i, delta_j = 0, 2
                elif i%4 < 2 and j%4 > 1:
                    delta_i, delta_j = 2, 0
    if (delta_i == 0 and delta_j == 0) or (delta_i == 2 and delta_j == 2):
        for i in range(w):
            for j in range(h):
                """   
                |B B | G G|
                |B B | G G|
                |---------|
                |G G | R R|
                |G G | R R|
                -----------
                """
                #if it's a normal pattern without bias
                # making a nearest neighbour interpolation
                if(delta_i == 0):
                    # we check if we're not in the cell for a red filter
                    if red_filter[i][j] == 0 and (i % 4 < 2 or j % 4 < 2):
                        red_i = 3 - i%4
                        red_j = 3 - j%4
                        # if it's not the corner
                        if i + red_i < w or j + red_j < h:
                            red_filter[i][j] = red_filter[i + red_i][j + red_j]
                        # otherwise
                        else:
                            red_filter[i][j] = red_filter[i - 1 - i%4][j - 1 - i%4]
                    # we check if we're not in the cell for a blue filter    
                    if blue_filter[i][j] == 0 and (i % 4 > 1 or j % 4 > 1):
                        blue_i = i%4 - 1
                        blue_j = j%4 - 1
                        # if it's not the corner
                        if i - blue_i >= 0 or j - blue_j >= 0:
                            blue_filter[i][j] = blue_filter[i - blue_i][j - blue_j]
                        # otherwise
                        else:
                            blue_filter[i][j] = blue_filter[i + 4 - i%4][j + 4 - j%4]
                    """ 
                    |R R | G G|
                    |R R | G G|
                    |-----------|
                    |G G | B B|
                    |G G | B B|
                    """
                    #if it's a  pattern with bias          
                else:
                    # we check if we're not in the cell for a blue filter
                    if blue_filter[i][j] == 0 and (i % 4 < 2 or j % 4 < 2):
                        blue_i = 3 - i%4
                        blue_j = 3 - j%4
                        # if it's not the corner
                        if i + blue_i < w or j + blue_j < h:
                            blue_filter[i][j] = blue_filter[i + blue_i][j + blue_j]
                        # otherwise
                        else:
                            blue_filter[i][j] = blue_filter[i - 1 - i%4][j - 1 - i%4]
                    # we check if we're not in the cell for a red filter
                    if red_filter[i][j] == 0 and (i % 4 > 1 or j % 4 > 1):
                        red_i = i%4 - 1
                        red_j = j%4 - 1
                        # if it's not the corner
                        if i - red_i >= 0 or j - red_j >= 0:
                            red_filter[i][j] = red_filter[i - red_i][j - red_j]
                        # otherwise
                        else:
                            red_filter[i][j] = red_filter[i + 4 - i%4][j + 4 - j%4]
        # bilinear interpolation for a green filter
        for i in range(0, w, 2):
            for j in range(0, h, 2):
                # we check if we're not in the cell for a green filter
                if green_filter[i][j] == 0:
                    # if we're in the upper left square
                    if i%4 < 2 and j%4 < 2:
                        # if it's not the corner
                        if i + 2 < w and j + 2 < h:
                            green_filter[i][j] = (green_filter[i+2][j] + green_filter[i][j+2])/2
                            green_filter[i+1][j] = (green_filter[i+2][j] + green_filter[i][j+2])/2
                            green_filter[i][j+1] = (green_filter[i+2][j] + green_filter[i][j+2])/2
                            green_filter[i+1][j+1] = (green_filter[i+2][j] + green_filter[i][j+2])/2
                        # otherwise
                        else:
                            green_filter[i][j] = (green_filter[i-1][j] + green_filter[i][j-1])/2
                            green_filter[i+1][j] = (green_filter[i-1][j] + green_filter[i][j-1])/2
                            green_filter[i][j+1] = (green_filter[i-1][j] + green_filter[i][j-1])/2
                            green_filter[i+1][j+1] = (green_filter[i-1][j] + green_filter[i][j-1])/2
                    # if we're in the lower right square
                    if i%4 > 1 and j%4 > 1:
                        # if it's not the corner
                        if i - 1 >= 0 and j - 1 >= 0:
                            green_filter[i][j] = (green_filter[i-1][j] + green_filter[i][j-1])/2
                            green_filter[i+1][j] = (green_filter[i-1][j] + green_filter[i][j-1])/2
                            green_filter[i][j+1] = (green_filter[i-1][j] + green_filter[i][j-1])/2
                            green_filter[i+1][j+1] = (green_filter[i-1][j] + green_filter[i][j-1])/2
                        # otherwise
                        else:
                            green_filter[i][j] = (green_filter[i+2][j] + green_filter[i][j+2])/2
                            green_filter[i+1][j] = (green_filter[i+2][j] + green_filter[i][j+2])/2
                            green_filter[i][j+1] = (green_filter[i+2][j] + green_filter[i][j+2])/2
                            green_filter[i+1][j+1] = (green_filter[i+2][j] + green_filter[i][j+2])/2
    else:
        for i in range(0, w, 2):
            for j in range(0, h, 2):
                if delta_i == 0:
                    """
                    | G G | B B |
                    | G G | B B |
                    |-----------|
                    | R R | G G |
                    | R R | G G |
                    """
                    # we check if we're not in the cell for a red filter
                    if red_filter[i][j] == 0:
                        # if we're in the upper left square
                        if (i % 4 < 2 and j % 4 < 2):
                            # if it's not the corner
                            if i + 2 < w:
                                red_filter[i][j] = red_filter[i + 2][j]
                                red_filter[i+1][j] = red_filter[i + 2][j]
                                red_filter[i][j+1] = red_filter[i + 2][j]
                                red_filter[i+1][j+1] = red_filter[i + 2][j]
                            # otherwise
                            else:
                                red_filter[i][j] = red_filter[i - 1][j]
                                red_filter[i+1][j] = red_filter[i - 1][j]
                                red_filter[i][j+1] = red_filter[i - 1][j]
                                red_filter[i+1][j+1] = red_filter[i - 1][j]
                        # if we're in the upper right square
                        elif (i%4 < 2 and j%4 > 1):
                            if i+2 < w and j-2 >= 0:
                                red_filter[i][j] = red_filter[i + 2][j - 2]
                                red_filter[i+1][j] = red_filter[i + 2][j - 2]
                                red_filter[i][j+1] = red_filter[i + 2][j - 2]
                                red_filter[i+1][j+1] = red_filter[i + 2][j - 2]
                            elif i+2 > w and j-2 >= 0:
                                red_filter[i][j] = red_filter[i - 2][j - 2]
                                red_filter[i+1][j] = red_filter[i - 2][j - 2]
                                red_filter[i][j+1] = red_filter[i - 2][j - 2]
                                red_filter[i+1][j+1] = red_filter[i - 2][j - 2]
                            elif i+2 < w and j-2 < 0:
                                red_filter[i][j] = red_filter[i + 2][j + 2]
                                red_filter[i+1][j] = red_filter[i + 2][j + 2]
                                red_filter[i][j+1] = red_filter[i + 2][j + 2]
                                red_filter[i+1][j+1] = red_filter[i + 2][j + 2]
                            elif i+2 > w and j-2 < 0:
                                red_filter[i][j] = red_filter[i - 2][j + 2]
                                red_filter[i+1][j] = red_filter[i - 2][j + 2]
                                red_filter[i][j+1] = red_filter[i - 2][j + 2]
                                red_filter[i+1][j+1] = red_filter[i - 2][j + 2]
                        # if we're in the lower right square
                        elif(i%4 > 1 and j%4 >1):
                            if j - 2 >= 0:
                                red_filter[i][j] = red_filter[i][j - 2]
                                red_filter[i+1][j] = red_filter[i][j - 2]
                                red_filter[i][j+1] = red_filter[i][j - 2]
                                red_filter[i+1][j+1] = red_filter[i][j - 2]
                            else:
                                red_filter[i][j] = red_filter[i][j + 2]
                                red_filter[i+1][j] = red_filter[i][j + 2]
                                red_filter[i][j+1] = red_filter[i][j + 2]
                                red_filter[i+1][j+1] = red_filter[i][j + 2]
                    # we check if we're not in the cell for a blue filter
                    if blue_filter == 0:
                        # if we're in the upper left square
                        if (i % 4 < 2 and j % 4 < 2):
                            if j + 2 < w:
                                blue_filter[i][j] = blue_filter[i][j+2]
                                blue_filter[i+1][j] = blue_filter[i][j+2]
                                blue_filter[i][j+1] = blue_filter[i][j+2]
                                blue_filter[i+1][j+1] = blue_filter[i][j+2]
                            else:
                                blue_filter[i][j] = blue_filter[i][j-2]
                                blue_filter[i+1][j] = blue_filter[i][j-2]
                                blue_filter[i][j+1] = blue_filter[i][j-2]
                                blue_filter[i+1][j+1] = blue_filter[i][j-2]
                        elif (i%4 > 1 and j%4 < 2):
                            if i+2 < w and j+2 < h:
                                blue_filter[i][j] = blue_filter[i + 2][j + 2]
                                blue_filter[i+1][j] = blue_filter[i + 2][j + 2]
                                blue_filter[i][j+1] = blue_filter[i + 2][j + 2]
                                blue_filter[i+1][j+1] = blue_filter[i + 2][j + 2]
                            else:
                                blue_filter[i][j] = blue_filter[i - 2][j - 2]
                                blue_filter[i+1][j] = blue_filter[i - 2][j - 2]
                                blue_filter[i][j+1] = blue_filter[i - 2][j - 2]
                                blue_filter[i+1][j+1] = blue_filter[i - 2][j - 2]
                        elif(i%4 > 1 and j%4 >1):
                            if i - 2 >= 0:
                                blue_filter[i][j] = blue_filter[i-2][j]
                                blue_filter[i+1][j] = blue_filter[i-2][j]
                                blue_filter[i][j+1] = blue_filter[i-2][j]
                                blue_filter[i+1][j+1] = blue_filter[i-2][j]
                            else:
                                blue_filter[i][j] = blue_filter[i+2][j]
                                blue_filter[i+1][j] = blue_filter[i+2][j]
                                blue_filter[i][j+1] = blue_filter[i+2][j]
                                blue_filter[i+1][j+1] = blue_filter[i+2][j]
                else:
                    """
                    | G G | R R |
                    | G G | R R |
                    |-----------|
                    | B B | G G |
                    | B B | G G |
                    """
                    # we check if we're not in the cell for a red filter
                    if red_filter[i][j] == 0:
                        # if we're in the upper left square
                        if (i % 4 < 2 and j % 4 < 2):
                            if j + 2 < w:
                                red_filter[i][j] = red_filter[i][j+2]
                                red_filter[i+1][j] = red_filter[i][j+2]
                                red_filter[i][j+1] = red_filter[i][j+2]
                                red_filter[i+1][j+1] = red_filter[i][j+2]
                            else:
                                red_filter[i][j] = red_filter[i][j-2]
                                red_filter[i+1][j] = red_filter[i][j-2]
                                red_filter[i][j+1] = red_filter[i][j-2]
                                red_filter[i+1][j+1] = red_filter[i][j-2]
                        elif (i%4 > 1 and j%4 < 2):
                            if i+2 < w and j+2 < h:
                                red_filter[i][j] = red_filter[i + 2][j + 2]
                                red_filter[i+1][j] = red_filter[i + 2][j + 2]
                                red_filter[i][j+1] = red_filter[i + 2][j + 2]
                                red_filter[i+1][j+1] = red_filter[i + 2][j + 2]
                            else:
                                red_filter[i][j] = red_filter[i - 2][j - 2]
                                red_filter[i+1][j] = red_filter[i - 2][j - 2]
                                red_filter[i][j+1] = red_filter[i - 2][j - 2]
                                red_filter[i+1][j+1] = red_filter[i - 2][j - 2]
                        elif(i%4 > 1 and j%4 >1):
                            if i - 2 >= 0:
                                red_filter[i][j] = red_filter[i-2][j]
                                red_filter[i+1][j] = red_filter[i-2][j]
                                red_filter[i][j+1] = red_filter[i-2][j]
                                red_filter[i+1][j+1] = red_filter[i-2][j]
                            else:
                                red_filter[i][j] = red_filter[i+2][j]
                                red_filter[i+1][j] = red_filter[i+2][j]
                                red_filter[i][j+1] = red_filter[i+2][j]
                                red_filter[i+1][j+1] = red_filter[i+2][j]
                    # we check if we're not in the cell for a blue filter
                    if blue_filter[i][j] == 0:
                        # if we're in the upper left square
                        if (i % 4 < 2 and j % 4 < 2):
                            if i + 2 < w:
                                blue_filter[i][j] = blue_filter[i + 2][j]
                                blue_filter[i+1][j] = blue_filter[i + 2][j]
                                blue_filter[i][j+1] = blue_filter[i + 2][j]
                                blue_filter[i+1][j+1] = blue_filter[i + 2][j]
                            else:
                                blue_filter[i][j] = blue_filter[i - 1][j]
                                blue_filter[i+1][j] = blue_filter[i - 1][j]
                                blue_filter[i][j+1] = blue_filter[i - 1][j]
                                blue_filter[i+1][j+1] = blue_filter[i - 1][j]
                        elif (i%4 < 2 and j%4 > 1):
                            if i+2 < w and j-2 >= 0:
                                blue_filter[i][j] = blue_filter[i + 2][j - 2]
                                blue_filter[i+1][j] = blue_filter[i + 2][j - 2]
                                blue_filter[i][j+1] = blue_filter[i + 2][j - 2]
                                blue_filter[i+1][j+1] = blue_filter[i + 2][j - 2]
                            elif i+2 > w and j-2 >= 0:
                                blue_filter[i][j] = blue_filter[i - 2][j - 2]
                                blue_filter[i+1][j] = blue_filter[i - 2][j - 2]
                                blue_filter[i][j+1] = blue_filter[i - 2][j - 2]
                                blue_filter[i+1][j+1] = blue_filter[i - 2][j - 2]
                            elif i+2 < w and j-2 < 0:
                                blue_filter[i][j] = blue_filter[i + 2][j + 2]
                                blue_filter[i+1][j] = blue_filter[i + 2][j + 2]
                                blue_filter[i][j+1] = blue_filter[i + 2][j + 2]
                                blue_filter[i+1][j+1] = blue_filter[i + 2][j + 2]
                            elif i+2 > w and j-2 < 0:
                                blue_filter[i][j] = blue_filter[i - 2][j + 2]
                                blue_filter[i+1][j] = blue_filter[i - 2][j + 2]
                                blue_filter[i][j+1] = blue_filter[i - 2][j + 2]
                                blue_filter[i+1][j+1] = blue_filter[i - 2][j + 2]
                        elif(i%4 > 1 and j%4 >1):
                            if j - 2 >= 0:
                                blue_filter[i][j] = blue_filter[i][j - 2]
                                blue_filter[i+1][j] = blue_filter[i][j - 2]
                                blue_filter[i][j+1] = blue_filter[i][j - 2]
                                blue_filter[i+1][j+1] = blue_filter[i][j - 2]
                            else:
                                blue_filter[i][j] = blue_filter[i][j + 2]
                                blue_filter[i+1][j] = blue_filter[i][j + 2]
                                blue_filter[i][j+1] = blue_filter[i][j + 2]
                                blue_filter[i+1][j+1] = blue_filter[i][j + 2]
        # bilinear interpolation for a green filter
        for i in range(0, w, 2):
            for j in range(0, h, 2):
                if green_filter[i][j] == 0:
                    # if we're in the upper right square
                    if i%4 < 2 and j%4 > 1: 
                        # if it's not the corner
                        if i + 2 < w and j - 2 >= 0:
                            green_filter[i][j] = (green_filter[i][j-2] + green_filter[i+2][j])/2
                            green_filter[i+1][j] = (green_filter[i][j-2] + green_filter[i+2][j])/2
                            green_filter[i][j+1] = (green_filter[i][j-2] + green_filter[i+2][j])/2
                            green_filter[i+1][j+1] = (green_filter[i][j-2] + green_filter[i+2][j])/2
                        elif i + 2 > w and j - 2 >= 0:
                            green_filter[i][j] = (green_filter[i][j-2] + green_filter[i-2][j])/2
                            green_filter[i+1][j] = (green_filter[i][j-2] + green_filter[i-2][j])/2
                            green_filter[i][j+1] = (green_filter[i][j-2] + green_filter[i-2][j])/2
                            green_filter[i+1][j+1] = (green_filter[i][j-2] + green_filter[i-2][j])/2
                        elif i + 2 < w and j - 2 < 0:
                            green_filter[i][j] = (green_filter[i][j+2] + green_filter[i+2][j])/2
                            green_filter[i+1][j] = (green_filter[i][j+2] + green_filter[i+2][j])/2
                            green_filter[i][j+1] = (green_filter[i][j+2] + green_filter[i+2][j])/2
                            green_filter[i+1][j+1] = (green_filter[i][j+2] + green_filter[i+2][j])/2
                        elif i + 2 > w and j - 2 < 0:
                            green_filter[i][j] = (green_filter[i][j+2] + green_filter[i-2][j])/2
                            green_filter[i+1][j] = (green_filter[i][j+2] + green_filter[i-2][j])/2
                            green_filter[i][j+1] = (green_filter[i][j+2] + green_filter[i-2][j])/2
                            green_filter[i+1][j+1] = (green_filter[i][j+2] + green_filter[i-2][j])/2
                    # if we're in the lower left square
                    if i%4 > 1 and j%4 < 2: 
                        # if it's not the corner
                        if i - 2 >= 0 and j + 2 < h:
                            green_filter[i][j] = (green_filter[i-2][j] + green_filter[i][j+2])/2
                            green_filter[i+1][j] = (green_filter[i-2][j] + green_filter[i][j+2])/2
                            green_filter[i][j+1] = (green_filter[i-2][j] + green_filter[i][j+2])/2
                            green_filter[i+1][j+1] = (green_filter[i-2][j] + green_filter[i][j+2])/2
                        elif i - 2 >= 0 and j + 2 > h:
                            green_filter[i][j] = (green_filter[i-2][j] + green_filter[i][j-2])/2
                            green_filter[i+1][j] = (green_filter[i-2][j] + green_filter[i][j-2])/2
                            green_filter[i][j+1] = (green_filter[i-2][j] + green_filter[i][j-2])/2
                            green_filter[i+1][j+1] = (green_filter[i-2][j] + green_filter[i][j-2])/2
                        elif i - 2 < 0 and j + 2 < h:
                            green_filter[i][j] = (green_filter[i+2][j] + green_filter[i][j+2])/2
                            green_filter[i+1][j] = (green_filter[i+2][j] + green_filter[i][j+2])/2
                            green_filter[i][j+1] = (green_filter[i+2][j] + green_filter[i][j+2])/2
                            green_filter[i+1][j+1] = (green_filter[i+2][j] + green_filter[i][j+2])/2
                        elif i - 2 < 0 and j + 2 > h:
                            green_filter[i][j] = (green_filter[i+2][j] + green_filter[i][j-2])/2
                            green_filter[i+1][j] = (green_filter[i+2][j] + green_filter[i][j-2])/2
                            green_filter[i][j+1] = (green_filter[i+2][j] + green_filter[i][j-2])/2
                            green_filter[i+1][j+1] = (green_filter[i+2][j] + green_filter[i][j-2])/2
    # complete the image with all three filters 
    bayer[:, :, 0] = red_filter
    bayer[:, :, 1] = green_filter
    bayer[:, :, 2] = blue_filter
    image = bayer.copy()
    assert image.ndim == 3 and image.shape[-1] == 3
    return image

def scale_and_crop_x2(bayer):
    """Upscamples a 2D bayer pattern by factor 2 and takes the central crop.

    Args:
        bayer: 2D array (H, W) containing bayer pattern

    Returns:
        image_zoom: 2D array (H, W) corresponding to x2 zoomed and interpolated 
        one-channel image
    """
    assert bayer.ndim == 2
    w, h = bayer.shape #the width and the height of the image
    scaled_img = np.zeros((2 * w, 2 * h))  #scaled image with the width and height twice bigger
    #scaling the image
    for i in range(0, 2 * w, 2):
        for j in range(0, 2 * h, 2):
            scaled_img[i, j], scaled_img[i, j + 1], scaled_img[i + 1, j], scaled_img[i + 1, j + 1] = bayer[i // 2, j // 2], bayer[i // 2, j // 2], bayer[i // 2, j // 2], bayer[i // 2, j // 2]
    for i in range(2 * w):
        for j in range(2 * h):
            if 0 <= i - w / 2 < w and 0 <= j - h / 2 < h:
                bayer[i - int(w / 2), j - int(h / 2)] = scaled_img[i, j]
    cropped = bayer.copy()
    assert cropped.ndim == 2
    return cropped

def problem2():
    im = load_image("data/castle.png")
    bayer = rgb2bayer(im)
	#scale and crop each channel
    bayer[:, :, 0] = scale_and_crop_x2(bayer[:, :, 0])
    bayer[:, :, 1] = scale_and_crop_x2(bayer[:, :, 1])
    bayer[:, :, 2] = scale_and_crop_x2(bayer[:, :, 2])
	# interpolate
    im_zoom = bayer2rgb(bayer)
    save_image("data/castle_out.png", im_zoom)
problem2()
