import os
import cv2
import numpy as np
from datetime import datetime


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
def getbeandata(squareside=50):
    rootdir = '/Users/user1/OneDrive/OneDrive iOS/' \
        '_course_sirajMLmakemoney/week7/dataset'

    N = 15336
    D = squareside ** 2

    X = np.zeros((N, D))
    Y = np.zeros(N)

    currimg = 0
    picclasses = {}
    totclasses = 0
    t0 = datetime.now()
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if not(len(files) == 1 and files[0] == '.DS_Store'):
                filewithpath = subdir + '/' + file
                picclass = subdir.split('/')[-1]
                if picclass not in picclasses.keys():
                    picclasses[picclass] = (totclasses, picclass)
                    totclasses += 1
                filewithpath = subdir + '/' + file
                im = cv2.imread(filewithpath, cv2.IMREAD_GRAYSCALE)
                im = cv2.resize(im, (squareside, squareside))
                im = np.reshape(im, -1)
                Y[currimg] = picclasses[picclass][0]
                X[currimg, :im.shape[0]] = im
                if currimg % 100 == 0:
                    t1 = datetime.now()
                    print(currimg, t1 - t0)
                    t0 = t1
                currimg += 1

    return X, Y, picclasses
