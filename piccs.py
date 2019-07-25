import cv2
import numpy as np


specs=cv2.imread('glasses.png',1)
cv2.imshow("Specs",specs)

cv2.waitKey(0)
cv2.destroyAllWindows()