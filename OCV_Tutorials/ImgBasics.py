# OpenCV Tutorials on rading an image, writing to an image
# and showing an image

import numpy as np
import cv2

#Loading a color image in grayscale
img = cv2.imread('pink_open_11.jpg',0)

#Display the loaded grayscale image
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
