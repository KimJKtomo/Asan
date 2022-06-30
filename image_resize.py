import cv2
from glob import glob
import os

import glob
import cv2
images = [cv2.imread(file) for file in glob.glob("./output/All_tilt_image/*.jpg")]
new_images=[]
# print(images)
for i in range(0, len(images)):
    print(i)
    new_image = cv2.resize(images[i], dsize=(32, 64), interpolation=cv2.INTER_AREA)
    cv2.imwrite('./output/All_tilt_image/new/%d.jpg' % i , new_image)
