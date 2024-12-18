import os
import glob
import cv2
import statistics
import numpy as np
from io import BytesIO

from PIL import Image as Imgpl

from openpyxl import Workbook
from openpyxl.drawing.image import Image as Imgxl

## Good !!

user = os.getlogin()

types = ('jpg','jpeg')
files = []
for t in types:
    files += glob.glob("C:/yolo/ultralytics/runs/detect/predict/*." + t, recursive=True)
print(len(files))


dir_path = "C:/yolo/AI-Pred-Data/step4/"

filesPNG = [s for s in files if "PMX" in s]

allImaget = []
allImage = []
filenameSv = ""

for file in filesPNG:

    filename = os.path.basename(file)
    img = cv2.imread(file, -1)
    ohight, owidth, oColorimg = img.shape
    filenamedPos = filename.rfind(".")
    filenameNPos = filename.rfind("N")

    filenameNM = filename[:filenamedPos]
    imgBK = img

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
    img = image_H
    if len(chunks) > 0:    
            print(filenameNM)
            img = image_H
#            cv2.imwrite(output_dir + filenameNM + "P-N" + ".png", image_H)

    filenameOrg = filename[:filenameNPos]
    if filenameOrg != filenameSv:
        allImage = allImaget
        print(len(allImage))
        allImaget = []
        colsImage = []
        rowImage = []  
        j = 0
        cntW = 0
        for imga in allImage:
            rowImage.append(imga)
            j = j + 1
            if j == Cols:
                cntW = cntW + 1
                image_v = cv2.hconcat(rowImage) 
                rowImage = [] 
                j = 0  
                colsImage.append(image_v)
        image_H = cv2.vconcat(colsImage)
        if len(allImage) > 0:    
            print(filenameSv)
            cv2.imwrite(dir_path + filenameSv + "-s" + ".png", image_H)
        filenameSv = filenameOrg
        allImaget.append(img)
    else:
        filenameColPos = filename.rfind("C")
        filenameRowPos = filename.rfind("R")
        Cols = int(filename[filenameColPos + 1:filenamedPos])
        Rows = int(filename[filenameRowPos + 1:filenameColPos])   
        allImaget.append(img)

allImage = allImaget
print(len(allImage))
colsImage = []
rowImage = []  
j = 0
cntW = 0
for img in allImage:
    rowImage.append(img)
    j = j + 1
    if j == Cols:
        cntW = cntW + 1
        image_v = cv2.hconcat(rowImage) 
        rowImage = [] 
        j = 0  
        colsImage.append(image_v)
image_H = cv2.vconcat(colsImage)

mask = np.all(image_H[:,:,:] == [255, 255, 255],axis=-1)
dst = cv2.cvtColor(image_H, cv2.COLOR_BGR2BGRA)
dst[mask,3] = 0

print(filenameSv)
cv2.imwrite(dir_path + filenameSv + "-s" + ".png", dst)