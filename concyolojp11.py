import cv2
import glob
import os
import numpy as np

files = glob.glob("C:/yolo/AI-Pred-Data/step4/" + "/*.png", recursive=True)
dir_path = "C:/yolo/AI-Pred-Data/step3/"
PathOut = "C:/yolo/AI-Pred-Data/step5/"

for file in files:

    filename = os.path.basename(file)
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    ohight, owidth = img.shape    
#    ohight, owidth, oColorimg = img.shape
    filenameXPos = filename.rfind("PMX")
    filenameYPos = filename.rfind("MY")
    filenameEPos = filename.rfind("--")
    filenameNM = filename[:filenameXPos]
    adjustMX = int(filename[filenameXPos +3 :filenameYPos])
    adjustMY = int(filename[filenameYPos +2 :filenameEPos])

    print(filenameNM, adjustMX, adjustMY)

    width, height = img.shape[:2]
    rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    print(rgba.shape)
    rgba[:, :, 3] = np.where(np.all(rgba == 255, axis=-1), 0, 255)
    x1, y1, x2, y2 = adjustMX, adjustMY, rgba.shape[1] + adjustMX, rgba.shape[0] + adjustMY

    dst = cv2.imread(dir_path + filenameNM + "--q.jpeg")
 
# 合成!
    dst[y1:y2, x1:x2] = dst[y1:y2, x1:x2] * (1 - rgba[:, :, 3:] / 255) + \
                      rgba[:, :, :3] * (rgba[:, :, 3:] / 255)

    cv2.imwrite(PathOut + filenameNM + "-a.jpeg", dst)