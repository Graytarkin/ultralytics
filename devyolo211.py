from pathlib import Path
import os
import cv2
import numpy as np
import glob

files = glob.glob("C:/yolo/AI-Pred-Data/step1" + '/*.png', recursive=True)
output_dir = "C:/yolo/AI-Pred-Data/step2/"

for file in files:

    filename = os.path.basename(file)
    print('File:' + filename)
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    filenamedPos = filename.rfind(".")
    filenameNM = filename[:filenamedPos]
 
    size = (500, 500)  # 分割後の大きさ
    rows = int(np.ceil(img.shape[0] / size[0]))  # 行数
    cols = int(np.ceil(img.shape[1] / size[1]))  # 列数
    rowsb1 = int(img.shape[0] / rows)
    rowsb2 = int(rowsb1 / 2)   
    colsb1 = int(img.shape[1] / cols)
    colsb2 = int(colsb1 / 2)   

    print(filenameNM, rows, cols)

    mcimg = img[rowsb2 : img.shape[0] - rowsb2, colsb2 : img.shape[1] - colsb2]
    mrows = int(np.ceil(mcimg.shape[0] / size[0]))  # 行数
    mcols = int(np.ceil(mcimg.shape[1] / size[1]))  # 列数

    filenameNMP  = filenameNM.rfind("-")
    filenameNM = filenameNM[0:filenameNMP] + "MX" + str(colsb2) + "MY" + str(rowsb2) + filenameNM[filenameNMP+2:]
    print("msize", mrows, mcols, filenameNM)

    chunks = []
    csizex = []
    csizey = []
    for mrow_img in np.array_split(mcimg, mrows, axis=0):
        for chunk in np.array_split(mrow_img, mcols, axis=1):
            sy, sx, sc = chunk.shape
            csizex.append(sx)
            csizey.append(sy) 
            chunks.append(chunk)
    print(len(chunks))


    for i, chunk in enumerate(chunks):
        szx = csizex[i]
        szy = csizey[i]
        save_path = output_dir+filenameNM+"-N"+str(f'{i:04}')+"-X"+str(szx)+"Y"+str(szy)+"-R"+str(mrows)+"C"+str(mcols)+".png"
#        print(save_path)
        cv2.imwrite(str(save_path), chunk)