import os
import cv2
rootdir = '/Users/user1/OneDrive/OneDrive iOS/' \
    '_course_sirajMLmakemoney/week7/dataset'
import numpy as np


# ran the bit of code below to get the count of images and the largest
# image dimensionality.

# totfiles = 0
# maxdim = 0
# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         if not(len(files) == 1 and files[0] == '.DS_Store'):
#             totfiles += 1
#             filewithpath = subdir + '/' + file
#             im = cv2.imread(filewithpath, cv2.IMREAD_COLOR)
#             currdim = im.shape[0] * im.shape[1] * im.shape[2]
#             if totfiles % 1000 == 0:
#                 print(totfiles)
#             if currdim > maxdim:
#                 maxdim = currdim

# totfiles = 15336
# maxdim = 413952

# no need to run the above code each time to get the file count and max dims
N = 15336
# extra column for classification
D = 413952 + 1

images = np.zeros((N, D))

currimg = 0
picclasses = {}
totclasses = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if not(len(files) == 1 and files[0] == '.DS_Store'):
            filewithpath = subdir + '/' + file
            picclass = subdir.split('/')[-1]
            if picclass not in picclasses.keys():
                picclasses[picclass] = totclasses
                totclasses += 1
            im = cv2.imread(filewithpath, cv2.IMREAD_COLOR)
            im = np.reshape(im, -1)
            filewithpath = subdir + '/' + file
            im = cv2.imread(filewithpath, cv2.IMREAD_COLOR)
            im = np.reshape(im, -1)
            images[currimg, 0] = picclasses[picclass]
            images[currimg, 1:(im.shape[0] + 1)] = im
            currimg += 1
            if currimg % 100 == 0:
                print(currimg)

np.savetxt(fname='data/images.csv', X=im,
           delimiter=',', header='img_class,pixels')
