from pathlib import Path
import os
import cv2
import numpy as np
import glob

types = ('jpg','jpeg')
files = []
for t in types:
    files += glob.glob("C:/yolo/AI-Pred-Data/step0/*." + t, recursive=True)
print(len(files))

output_dir = "C:/yolo/AI-Pred-Data/step2/"

for file in files:

    filename = os.path.basename(file)
    print('File:' + filename)
    img = cv2.imread(file)

    ohight, owidth, oColorimg = img.shape
    if ohight < 2000 or owidth < 3000:
        result = cv2.resize(img, (owidth * 2, ohight * 2)) 
        img = result
        ohight, owidth, oColorimg = img.shape 
        
    if ohight < 2000 or owidth < 3000:
        result = cv2.resize(img, (owidth * 1.5, ohight * 1.5)) 
        img = result
        ohight, owidth, oColorimg = img.shape 


    filenamedPos = filename.rfind(".")
    filenameNM = filename[:filenamedPos]
 
    size = (500, 500)  # 分割後の大きさ
    rows = int(np.ceil(img.shape[0] / size[0]))  # 行数
    cols = int(np.ceil(img.shape[1] / size[1]))  # 列数

    print(filenameNM, rows, cols)

    chunks = []
    csizex = []
    csizey = []
    for row_img in np.array_split(img, rows, axis=0):
        for chunk in np.array_split(row_img, cols, axis=1):
            chunks.append(chunk)
            sy, sx, sc = chunk.shape
            csizex.append(sx)
            csizey.append(sy)           
    print(len(chunks))

    for i, chunk in enumerate(chunks):
        szx = csizex[i]
        szy = csizey[i]
        save_path = output_dir+filenameNM+"-N"+str(f'{i:04}')+"-X"+str(szx)+"Y"+str(szy)+"-R"+str(rows)+"C"+str(cols)+".jpeg"
#        print(save_path)
        cv2.imwrite(str(save_path), chunk)