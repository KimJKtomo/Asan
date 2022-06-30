import numpy as np
import cv2

img =cv2.imread("./Left_side/Left_side_9_data.jpg", cv2.IMREAD_GRAYSCALE)
img_array = np.array(img)
print(np.shape(img_array))
color= cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)

cv2.imwrite('./supine_color/Left_side_9_data.jpg', im_color)

