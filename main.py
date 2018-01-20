from PIL import Image # to install that use pip install Pillow
import numpy as np
from PCG import PCG

def get_min_patch_channel(x0, x1, size_patch, A_ev,  width, height, img):
    #" width et height sont inverses c normal"
    """ Return the min on the patch centered on (x0,x1) in img and on the different channel"""
    half = int(size_patch/2)
    startx0 = x0 - half
    startx1 = x1 - half
    valuesR = []
    valuesG= []
    valuesB = []
    for i in range(startx0, startx0 + size_patch):
        for j in range(startx1, startx1 + size_patch):
            if i >= half and j >= half and i <= width - 1 - half and j <= height - 1 - half:
                valuesR.append(img[i,j][0])
                valuesG.append(img[i,j][1])
                valuesB.append(img[i,j][2])
    return min(float(min(valuesR))/float(A_ev["R"]), float(min(valuesG))/float(A_ev["G"]), float(min(valuesB))/float(A_ev["B"]))

def get_vector_place(i,j, width):
    """ return the place of a pixel initially in [i,j] in a vector version of the matrix """
    return i*width + j

def get_matrix_place(v, width):
    """ the contrary of the previous function"""
    return (int(v/width), v%width)

def get_list_windows(placei, placej, width, height, window_size):
    """ Return the list of windows (actually extreme left up point) that contains placei and placej"""
    list_window = []
    minx = max(0, placei[0] - window_size + 1, placej[0] - window_size + 1 )
    maxx = min(height - window_size - 1,  placei[0] + 1, placej[0] + 1)

    miny = max(0, placei[1] - window_size + 1, placej[1] - window_size + 1 )
    maxy = min(width - window_size - 1,  placei[1] + 1, placej[1] + 1)
    for i in range(minx, maxx):
        for j in range(miny, maxy):
            list_window.append((i,j))
    return list_window

def get_mu_sigma(window, window_size, img):
    """ Return value of the mean and the covariance over a window of img"""
    sigma = np.zeros([3, 3])
    valuesR = [img[i,j][0] for i in range(window[0], window[0] + window_size) for j in range(window[1], window[1] + window_size)] # red values
    valuesG = [img[i, j][1] for i in range(window[0], window[0] + window_size) for j in range(window[1], window[1] + window_size)]  # green values
    valuesB = [img[i,j][2] for i in range(window[0], window[0] + window_size) for j in range(window[1], window[1] + window_size)] # blue values
    # We compute the mean for each channel
    mu = np.asarray([[np.mean(valuesR)], [np.mean(valuesG)], [np.mean(valuesB)]])
    # We compute the covariances
    covRG = np.cov(valuesR, valuesG)
    covRB = np.cov(valuesR, valuesB)
    covGB = np.cov(valuesG, valuesB)
    # We fill the sigma matrix
    sigma[0,0] = covRG[0,0] # RR
    # RG
    sigma[0,1] = covRG[1,0]
    sigma[1,0] = covRG[1,0]
    #RB
    sigma[0, 2] = covRB[0,1]
    sigma[2,0] = covRB[0,1]
    # GG
    sigma[1,1] = covRG[1,1]
    ## GB
    sigma[1,2] = covGB[0,1]
    sigma[2,1] = covGB[0,1]
    # BB
    sigma[2,2] = covGB[1,1]

    return mu, sigma


def get_laplacian_value(i, j, window_size, width, height,  img, regul):
    placei = get_matrix_place(i,width)
    placej = get_matrix_place(j, width)
    krock = 0. # kronecker value
    if i == j:(placei[0] - placej[0]) < window_size and abs(placei[1] - placej[1]) < window_size: # we check they have at least one window in common
        # We get all the windows of size window_size that contain the two pixels
        list_windows = get_list_windows(placei, placej, width, height, window_size)
        colori = np.asarray(
            [[img[placei[0], placei[1]][0]], [img[placei[0], placei[1]][1]], [img[placei[0], placei[1]][2]]])
        colorj = np.asarray(
            [[img[placej[0], placej[1]][0]], [img[placej[0], placej[1]][1]], [img[placej[0], placej[1]][2]]])
        nbPixels = float(window_size ** 2)
        value = 0.
        for window in list_windows:
            mu, sigma = get_mu_sigma(window, window_size, img)
            value += krock - (1./nbPixels)*(1. + np.asscalar((colori - mu).T.dot(np.linalg.inv(sigma + (regul/nbPixels)*np.identity(3)).dot(colorj -mu))))
        return value
    else:
        return 0.


###################"
# Parameters
size = (100, 100)
w= 0.95
size_patch = 9
lamb = 10.**(-4)
window_size = 3
eps = 0.01 # regularization for the laplacian terms
###################


# We load the image and we create a gray version of it to have the intensity evaluated on each pixels
myimage = Image.open("Data/image3.jpeg") # the real image
mydarkImage = Image.open("Data/image3.jpeg") # the gray version
#myimage.show()
myimage.thumbnail(size)
myimage.show()
mydarkImage.thumbnail(size)
mydark = mydarkImage.convert('L')
myim = myimage.convert('RGB')
img = myim.load()
imgGray = mydark.load()

# We create the dark channel
dark_channel  = Image.new('L', myimage.size)
height, width = myimage.size
dark_channel_pix = dark_channel.load()
to_rank = []
locations = []
for i in range(height):
    for j in range(width):
        value = min([img[i,j][t] for t in range(3)])
        dark_channel_pix[i,j] = value
        to_rank.append(value)
        locations.append((i,j))
dark_channel.show()

## To recover A ##############################################################"
#First get 1 percent with highest intensity in dark channel
percent = int(width*height/100)
indices = np.argpartition(np.asarray(to_rank), -percent)[-percent:]
pixels_selected = [locations[ind] for ind in indices]
# Then get highest intensity in original image for those pixels
values = np.asarray([imgGray[i,j] for i,j in pixels_selected])
maxi = int(np.argmax(values))
pixel_selected = pixels_selected[maxi]
A = dict()
A["R"] = img[pixel_selected[0], pixel_selected[1]][0]
A["G"] = img[pixel_selected[0], pixel_selected[1]][1]
A["B"] = img[pixel_selected[0], pixel_selected[1]][2]
print A

## To recover t constant (called ttild in the paper) ##########################
t_constant = Image.new('L', myimage.size)
t_constant_pix = t_constant.load()
t_array = np.zeros([height, width])

# Compute t for each pixel
for i in range(height):
    for j in range(width):
        new_value = get_min_patch_channel(i,j,size_patch, A, height, width, img)
        t_constant_pix[i,j] = int(255*(1. - w*new_value))
        t_array[i,j] = 1.  - w*new_value
t_constant.show()

## To recover t (soft matting) ################################################
t_tild = t_array.reshape([t_array.shape[0]*t_array.shape[1], 1])
t = np.zeros(t_tild.shape)
# Creation of the laplacian matrix
L = np.zeros([t.shape[0], t.shape[0]])
for i in range(t.shape[0]):
    if i%100 == 0:
        print "Laplacian calculation", i
    for j in range(t.shape[0]):
        L[i,j] = get_laplacian_value(i = i, j= j, window_size=window_size,width=width, height=height, img = img, regul = eps)

t_final = PCG(L + lamb*np.identity(t.shape[0]), lamb*t_tild, t, 10.**(-3))
t_final = t_final.reshape(t_array.shape)
## Recover A (no t soft matting for the moment)
t0 = 0.1
t_array = t_final
final_image  = Image.new('RGB', myimage.size)
fin_pixel = final_image.load()
for i in range(height):
    for j in range(width):
        fin_pixel[i,j] = (int(float(img[i,j][0] - A["R"])/max(t0, t_array[i,j]) + A["R"]),
                          int(float(img[i,j][1] - A["G"])/max(t0, t_array[i,j]) + A["G"]),
                          int(float(img[i,j][2] - A["B"])/max(t0, t_array[i,j]) + A["B"]))
final_image.show()




