from pathlib import Path
import os
import cv2
import numpy as np
import glob
import statistics
import math
from scipy import stats

types = ('jpg','jpeg')
files = []
for t in types:
    files += glob.glob("C:/yolo/AI-Pred-Data/step0/*." + t, recursive=True)
print(len(files))

output_dir = "C:/yolo/AI-Pred-Data/step1/"

for file in files:

    filename = os.path.basename(file)
    print('File:' + filename)
    img = cv2.imread(file)
    ohight, owidth, oColorimg = img.shape
    if ohight < 1000 or owidth < 1000:
        result = cv2.resize(img, (owidth * 2, ohight * 2)) 
        img = result
        ohight, owidth, oColorimg = img.shape 
        
    if ohight < 1000 or owidth < 1000:
        result = cv2.resize(img, (owidth * 2, ohight * 2)) 
        img = result
        ohight, owidth, oColorimg = img.shape

    filenamedPos = filename.rfind(".")
    filenameNM = filename[:filenamedPos]
 
    size = (50, 50)  # 分割後の大きさ
    rows = int(np.ceil(img.shape[0] / size[0]))  # 行数
    cols = int(np.ceil(img.shape[1] / size[1]))  # 列数

    print(filenameNM, rows, cols)

    mrows = int(np.ceil(img.shape[0] / size[0]))  # 行数
    mcols = int(np.ceil(img.shape[1] / size[1]))  # 列数

    msize = [mrows, mcols]
    print(msize)

    chunks = []
    chunks.append(msize)
    for mrow_img in np.array_split(img, mrows, axis=0):
        for chunk in np.array_split(mrow_img, mcols, axis=1):
            chunks.append(chunk)
    print(len(chunks))

    popped_item = chunks.pop(0)
    Rows, Cols = popped_item
    print(Rows, Cols,  len(chunks))
    colsImage = []
    rowImage = []  
    j = 0
    for imga in chunks:
        gimg = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
        ret, imgT = cv2.threshold(gimg, 0, 255, cv2.THRESH_OTSU)
        std = np.std(gimg)
        median = np.median(gimg)
        if abs(median - ret) < 5 and std < 5:
            ret2, imgT = cv2.threshold(gimg, 50, 255, cv2.THRESH_BINARY)
        rowImage.append(imgT)
        j = j + 1
        if j == Cols:
            image_v = cv2.hconcat(rowImage) 
            rowImage = [] 
            j = 0  
            colsImage.append(image_v)
    image_H = cv2.vconcat(colsImage)
    if len(chunks) > 0:    
            print(filenameNM)
            cv2.imwrite(output_dir + filenameNM + "P-N" + ".png", image_H)