from PIL import Image # to install that use pip install Pillow
import numpy as np

myimage = Image.open("Data/image2.jpeg")
mydarkImage = Image.open("Data/image2.jpeg")
#myimage.show()
myimage.thumbnail((500,500))
myimage.show()
mydarkImage.thumbnail((500,500))
mydark = mydarkImage.convert('L')
myim = myimage.convert('RGB')
img = myim.load()
imgGray = mydark.load()

# Create the dark channel
dark_channel  = Image.new('L', myimage.size)
width, height = myimage.size
dark_channel_pix = dark_channel.load()
to_rank = []
locations = []
for i in range(width):
    for j in range(height):
        value = min([img[i,j][t] for t in range(3)])
        dark_channel_pix[i,j] = value
        to_rank.append(value)
        locations.append((i,j))

dark_channel.show()

## To recover A
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

## To recover t constant (called ttild in the paper)
w= 0.95
size_patch = 15
lamb = 10.**(-4)

t_constant = Image.new('L', myimage.size)
t_constant_pix = t_constant.load()
t_array = np.zeros([width, height])

def get_min_patch_channel(x0, x1, size_patch, A_ev,  width, height, img):
    """ Return the min on the patch centered on (x0,x1) and on the different channel"""
    half = int(size_patch/2)
    startx0 = x0 - half
    startx1 = x1 - half
    valuesR = []
    valuesG= []
    valuesB = []
    for i in range(startx0, startx0 + 15):
        for j in range(startx1, startx1 + 15):
            if i >= half and j >= half and i <= width - 1 - half and j <= height - 1 - half:
                valuesR.append(img[i,j][0])
                valuesG.append(img[i,j][1])
                valuesB.append(img[i,j][2])
    return min(float(min(valuesR))/float(A_ev["R"]), float(min(valuesG))/float(A_ev["G"]), float(min(valuesB))/float(A_ev["B"]))


# Compute t for each pixel
for i in range(width):
    for j in range(height):
        new_value = get_min_patch_channel(i,j,size_patch, A, width, height, img)
        t_constant_pix[i,j] = int(255*(1. - w*new_value))
        t_array[i,j] = 1.  - w*new_value
t_constant.show()

## To recover t (soft matting)

## Recover A (no t soft matting for the moment)
t0 = 0.1
final_image  = Image.new('RGB', myimage.size)
fin_pixel = final_image.load()
for i in range(width):
    for j in range(height):
        fin_pixel[i,j] = (int(float(img[i,j][0] - A["R"])/max(t0, t_array[i,j]) + A["R"]), int(float(img[i,j][1] - A["G"])/max(t0, t_array[i,j]) + A["G"]), int(float(img[i,j][2] - A["B"])/max(t0, t_array[i,j]) + A["B"]))
final_image.show()




