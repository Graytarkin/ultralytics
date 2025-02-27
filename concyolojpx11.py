import cv2
import glob
import os
import numpy as np

files = glob.glob("C:/yolo/AI-Pred-Data/step4/" + "/*.png", recursive=True)
dir_path = "C:/yolo/AI-Pred-Data/step3/"
PathOut = "C:/yolo/AI-Pred-Data/step5/"

PathOutT = "C:/yolo/AI-Pred-Data/step8/"

for file in files:

    filename = os.path.basename(file)
#    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(file)
#    ohight, owidth,color = img.shape    
    ohight, owidth, oColorimg = img.shape
    filenameXPos = filename.rfind("PMX")
    filenameYPos = filename.rfind("MY")
    filenameEPos = filename.rfind("--")
    filenameNM = filename[:filenameXPos]
    adjustMX = int(filename[filenameXPos +3 :filenameYPos])
    adjustMY = int(filename[filenameYPos +2 :filenameEPos])

    print(filenameNM, adjustMX, adjustMY)

#    gray2 = no_lines_img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 閾値の設定
    threshold = 160
    # 二値化(閾値100を超えた画素を255にする。)
    ret, img_thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    rgbam = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)

    # Point 1: 白色部分に対応するマスク画像を生成
    mask = np.all(rgbam[:,:,:] == [255, 255, 255], axis=-1)
 
    # Point 2: 元画像をBGR形式からBGRA形式に変換
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
 
    # Point3: マスク画像をもとに、白色部分を透明化
    rgba[mask,3] = 0
 
    #### png画像として出力
    cv2.imwrite(PathOutT + filenameNM + "-T.png", rgba)

    x1, y1, x2, y2 = adjustMX, adjustMY, rgba.shape[1] + adjustMX, rgba.shape[0] + adjustMY

    dst = cv2.imread(dir_path + filenameNM + "--q.jpeg")

    rgba = rgba[:,:,:3]  # アルファチャンネルは取り出しちゃったのでもういらない。
    dst[y1:y2, x1:x2] = dst[y1:y2, x1:x2] * (1 - rgba / 255)
    dst[y1:y2, x1:x2] = dst[y1:y2, x1:x2] + rgba * (rgba / 255) # 貼り付ける方の画像に透過率をかけて加算。

    cv2.imwrite(PathOut + filenameNM + "-a.jpeg", dst)
