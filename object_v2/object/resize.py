import numpy as np
import cv2
import glob

filename = "/home/horvath/Dokumentumok/szakdoga/object_v2/object/qr/train/images/image-035.png"

img = cv2.imread(filename,3)

img = cv2.resize(img, (36,48))

cv2.imwrite(filename,img)
