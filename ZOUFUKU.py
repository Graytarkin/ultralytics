import cv2
import glob
import os
import numpy as np
import random
import matplotlib.pyplot as plt
#%matplotlib inline

files = glob.glob("C:/workspace/yolowork/nextbatterbox" + "/*.jpeg", recursive=True)
input_dir = "C:/workspace/yolowork/nextbatterbox"
output_dir = "C:/workspace/yolowork/batterbox"

for file in files:

    filename = os.path.basename(file)
    print('File:' + filename)
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
#img = cv2.imread(input_dir + "/N0N273N1N125N23N101N14N268N75.jpeg",0)
    rows,cols=img.shape[:2]
#sumimg = img/255.0
    for i in range(10):
        var=6
        pts1 = np.float32([[0,0],[0,rows],[cols,rows],[cols,0]])
        pts2 = np.float32([[random.randint(-var,var),random.randint(-var,var)],
                       [random.randint(-var,var), rows+random.randint(-var,var)],
                       [cols+random.randint(-var,var),rows+random.randint(-var,var)],
                       [cols+random.randint(-var,var),random.randint(-var,var)]])

        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(cols,rows))
        posDot = filename.rfind(".")
        imgfile = filename[: posDot]
        cv2.imwrite(output_dir + "/" + imgfile + "-" + str(i) + ".jpeg", dst) 
#    sumimg+=dst/255.0

#plt.imshow(sumimg)
#plt.colorbar()
#plt.show()