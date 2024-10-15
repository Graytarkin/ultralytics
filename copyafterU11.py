import csv
import os
import glob
import hdbscan
import heapq
import numpy as np
from io import BytesIO
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from PIL import Image as Imgpl
# from openpyxl import load_workbook
from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.drawing.image import Image as Imgxl

dirOfPredx = "C:/yolo/AI-Pred-Data/step6/"

# 各クラスタリングアルゴリズムの設定
algorithms = [3,4,5,6,7,8]

with open("C:/yolo/ultralytics/runs/detect/predict/predictions.csv", encoding="utf-8") as file:
    lst = list(csv.reader(file))

saveF = ""
excelWRl = []
lcnt = 0
for listl in lst:
    lcnt = lcnt + 1
    dotPos = listl[0].rfind(".")
    listltype = listl[0][dotPos + 1:]

    if listl[0][:dotPos] != saveF:
        saveF = listl[0][:dotPos]
        textCnt = 0
        with open("C:/yolo/ultralytics/runs/detect/predict/labels/" + saveF + ".txt", "r", encoding="utf-8") as f:
            listtxt = f.readlines()
            listtxtS = []
            for listelm in listtxt:
                valofelm = listelm.split()
                listtxtS.append(valofelm)
#            print(len(listtxt),listtxtS)
    if listltype == "png":
        posfnameE = listl[0].rfind("PMX")
        fname = listl[0][:posfnameE]
    else:
        posfnameE = listl[0].rfind("-N")
        fname = listl[0][:posfnameE]
                
    elmm = [listl[0],listl[1],listl[2],listtxtS[textCnt][0],listtxtS[textCnt][1],listtxtS[textCnt][2],listtxtS[textCnt][3],listtxtS[textCnt][4],fname]
    textCnt +=1
    excelWRl.append(elmm)
#print(excelWRl)

filesp = glob.glob("C:/PylocalYolo/yolov5/runs/detect/exp" + "/*.png", recursive=True)
for file in filesp:
    filename = os.path.basename(file)
    posfnameE = listl[0].rfind("PMX")
    fname = listl[0][:posfnameE]
    if not(any(filename in row for row in excelWRl)):
        elmm = [filename,'','0','0','0','0','0','0',fname]
        textCnt +=1
        excelWRl.append(elmm)


filesg = glob.glob("C:/PylocalYolo/yolov5/runs/detect/exp" + "/*.Jpeg", recursive=True)
for file in filesg:
    filename = os.path.basename(file)
    posfnameE = listl[0].rfind("-N")
    fname = listl[0][:posfnameE]
    if not (any(filename in row for row in excelWRl)):
        elmm = [filename,'','0','0','0','0','0','0',fname]
        textCnt +=1
        excelWRl.append(elmm)

sortedexcelWRl = sorted(excelWRl, key=lambda x:(x[0], x[4]))

exName = ""
exPosR = 0
exPosC = 0

wb = Workbook()
ws = wb.active

fnamesv = ""
mkexposl = []

for dataelm in sortedexcelWRl:
    exName = dataelm
    exPosR = exPosR + 1
#   
    posDot = exName[0].rfind(".")
    filetype = exName[0][posDot + 1:]
#    print(filetype)
    if filetype == "png":
        posRCC = exName[0].rfind("C")
        posRCR = exName[0].rfind("-R")
        posSX = exName[0].rfind("-X")
        posSY = exName[0].rfind("Y")
        posN = exName[0].rfind("-N")
        posMX = exName[0].rfind("PMX")
        posMY = exName[0].rfind("MY")

        fname = exName[0][:posMX]
        filepathname = exName[0][:posMX + 1]
        fMX = int(exName[0][posMX + 3: posMY])
        fMY = int(exName[0][posMY + 2: posN])
        fNo = int(exName[0][posN + 2: posSX])
        fwx = int(exName[0][posSX + 2: posSY])
        fhy = int(exName[0][posSY + 1: posRCR])
        frr = int(exName[0][posRCR + 2: posRCC])
        fcc = int(exName[0][posRCC + 1: posDot])

    else:
        posRCC = exName[0].rfind("C")
        posRCR = exName[0].rfind("-R")
        posSX = exName[0].rfind("-X")
        posSY = exName[0].rfind("Y")
        posN = exName[0].rfind("-N")

        fname = exName[0][:posN]
        filepathname = exName[0][:posN + 1]
        fMX = 0
        fMY = 0
        fNo = int(exName[0][posN + 2: posSX])
        fwx = int(exName[0][posSX + 2: posSY])
        fhy = int(exName[0][posSY + 1: posRCR])
        frr = int(exName[0][posRCR + 2: posRCC])
        fcc = int(exName[0][posRCC + 1: posDot])

#
    ws.cell(row=exPosR, column=1).value = str(exName[0])
    ws.cell(row=exPosR, column=2).value = filetype    
    ws.cell(row=exPosR, column=3).value = str(exName[1])
    lLetter = str(exName[1])
    ws.cell(row=exPosR, column=4).value = str(exName[2])
    lLpercent = int(float(exName[2]) * 100)
    ws.cell(row=exPosR, column=5).value = str(exName[3]) 
    lLclassno = int(exName[3])   
    ws.cell(row=exPosR, column=6).value = str(exName[4])
    ws.cell(row=exPosR, column=7).value = str(exName[5])
    ws.cell(row=exPosR, column=8).value = str(exName[6])
    ws.cell(row=exPosR, column=9).value = str(exName[7])
    ws.cell(row=exPosR, column=11).value = filepathname
    ws.cell(row=exPosR, column=11).value = fname
    ws.cell(row=exPosR, column=12).value = fMX
    ws.cell(row=exPosR, column=13).value = fMY
    ws.cell(row=exPosR, column=14).value = fNo
    ws.cell(row=exPosR, column=15).value = fwx
    ws.cell(row=exPosR, column=16).value = fhy
    ws.cell(row=exPosR, column=17).value = frr
    ws.cell(row=exPosR, column=18).value = fcc      

    if filepathname != fnamesv:
        fnamesv = filepathname
        addwcol = 0
        addhrow = 0
        fwcolsv = fwx
        fhrowsv = fhy
        rowaddsv = -1
        fnosv = -1
    rowadd = fNo // fcc
    if rowadd != rowaddsv:
        addwcol = 0
        rowaddsv = rowadd
        addhrow = fhy + addhrow      
    if fNo != fnosv:
        fnosv = fNo
        addwcol = fwx + addwcol
    
    fforgY = fMY + addhrow - fhrowsv
    fforgX = fMX + addwcol - fwcolsv
    ws.cell(row=exPosR, column=20).value = fforgY
    ws.cell(row=exPosR, column=21).value = fforgX 

#    lFwx = int(float(exName[4]) * float(fwx) * 1)+ int(float(exName[6]) * float(fwx) * 0.5)
    lFwx = int(float(exName[4]) * float(fwx) * 1) - int(float(exName[6]) * float(fwx) * 0.5)
    ws.cell(row=exPosR, column=22).value = lFwx
    lFhy = int(float(exName[5]) * float(fhy) * 1) 
    ws.cell(row=exPosR, column=23).value = lFhy 
    areaw = int(float(exName[6]) * float(fwx) * float(exName[7]) * float(fhy) * 0.1)
    ws.cell(row=exPosR, column=24).value = areaw
            
    ttlFhy = fforgY + lFhy
    ttlFwx = fforgX + lFwx
#    ttlFhy2 = ttlFhy * 2
#
    ws.cell(row=exPosR, column=25).value = ttlFwx 
    ws.cell(row=exPosR, column=26).value = ttlFhy
    leftend = int((float(exName[4]) - (float(exName[6]) * 0.5)) *  float(fwx)) + fforgX
    rightend = int((float(exName[4]) + (float(exName[6]) * 0.5)) *  float(fwx)) + fforgX
#               fn:0     ft:1   char:2    cls:3      rate:4     tx:5    ty:6    le:7      re:8    aw:9
    mkexpos = [fname, filetype, lLetter, lLclassno, lLpercent, ttlFwx, ttlFhy, leftend, rightend, areaw]
    if lLpercent > 0:
        mkexposl.append(mkexpos)

wb.save(dirOfPredx + "exName" + ".xlsx")     

sdnposl = sorted(mkexposl, key=lambda x:(x[0],x[6])) 

clstsv = []
clstsvl = []
score2 = []
datafilesv = ""
for i in range(0,len(sdnposl)):
    datafile = sdnposl[i][0]
    if datafilesv != datafile:
        if datafilesv != "":
            column = [row[0] for row in score2]
            row = [row[1] for row in score2]
            score3 = np.array(score2)

            X = score3
            cntminsv = 999
            for algo in algorithms:
                if len(X)> 1:
                    cluster = hdbscan.HDBSCAN(min_cluster_size=algo).fit_predict(X)
                    cntmin = np.sum(cluster < 0)
                    if cntmin <= cntminsv:
                        cntminsv = cntmin
                        clstsv = cluster
            clstsvl.append(clstsv)    
            score2 = []           
        datafilesv = datafile         
    cellx = sdnposl[i][5] / 70
    celly = sdnposl[i][6] * 1
    scoreelm = [cellx, celly]
    score2.append(scoreelm)

column = [row[0] for row in score2]
row = [row[1] for row in score2]
score3 = np.array(score2)

X = score3
cntminsv = 999
for algo in algorithms:
    if len(X)> 1:
        cluster = hdbscan.HDBSCAN(min_cluster_size=algo).fit_predict(X)
        cntmin = np.sum(cluster < 0)
        if cntmin <= cntminsv:
            cntminsv = cntmin
            clstsv = cluster
clstsvl.append(clstsv) 

addi = 0
addj = 0
sdnpcls = []
sdnpclsmin = []
for i in range(0,len(sdnposl)):
    if addi >= len(clstsvl[addj]):
        addj = addj + 1
        addi = 0
    lcls = clstsvl[addj][addi]
    addi = addi + 1
    sdnpclsElm = [sdnposl[i][0],sdnposl[i][1],sdnposl[i][2],sdnposl[i][3],sdnposl[i][4],
                  sdnposl[i][5],sdnposl[i][6],sdnposl[i][7],sdnposl[i][8],sdnposl[i][9],0,lcls]
    if lcls < 0:
        sdnpclsmin.append(sdnpclsElm)
    else:
        sdnpcls.append(sdnpclsElm)

sdnclss = sorted(sdnpcls, key=lambda x:(x[0],x[6])) 
sdnclsmins = sorted(sdnpclsmin, key=lambda x:(x[0],x[6])) 


for imin in range(0,len(sdnclsmins)):
    flgfound = 0
    tempappc = []
    for icls in range(0,len(sdnclss)):
        if sdnclsmins[imin][0] == sdnclss[icls][0]:
            appclsElm = [sdnclss[icls][6], sdnclss[icls][11]]
            tempappc.append(appclsElm) 
    tempappci = [row[0] for row in tempappc]
    idx = np.abs(np.asarray(tempappci) - int(sdnclsmins[imin][6])).argmin()
    sdnclsmins[imin][11] = tempappc[idx][1]
    sdnclsmins[imin][10] = -1 
 
sdnclss.extend(sdnclsmins)

serai = sorted(sdnclss, key=lambda x:(x[0],x[11],x[5]))    

derai = []
cherai = []
meanscore = []
datafckey = ""
######
for i in range(0,len(serai)):
    datafc = serai[i][0] + str(serai[i][11])
    if datafckey != datafc:
        if datafckey != "":
            column = [row[0] for row in meanscore]
            row = [row[1] for row in meanscore]
            scorenp = np.array(meanscore)

# 各クラスタリングアルゴリズムの設定
            maxclsl = []
            for ai in range(2,7):
                if len(scorenp)> 1:
                    cluster = hdbscan.HDBSCAN(min_cluster_size=ai).fit_predict(scorenp)
                    cntmin = np.sum(cluster < 0)
                    maxclsElm = [cntmin,max(cluster),cluster,ai]
                    maxclsl.append(maxclsElm)
            maxclsls = sorted(maxclsl, key=lambda x:(x[3]),reverse=True)
            algomin = 99
            for si in range(0,len(maxclsls)):
                if maxclsls[si][0] == 0:
                    if algomin > maxclsls[si][1]:
                        if maxclsls[si][1] > 1:
                            algomin = maxclsls[si][1]
                            selectc = maxclsls[si][2]
            if algomin == 99:
                outmin = 99
                j = 0
                for ix in range(0,len(maxclsls)):
                    if maxclsls[ix][0] < outmin:
                        outmin = maxclsls[ix][0]
                        j = ix
                if outmin != 99:
                    selectc = maxclsls[j][2]

        for di in range(0,len(derai)): 
            ncElm = derai[di]
            if selectc[di] < 0:
                ncElm[10] = 88
            else:
                ncElm[10] = selectc[di]
            cherai.append(ncElm)
        datafckey = datafc
        derai = []
        meanscore = []
    scoreelm = [serai[i][5], serai[i][6]]
    meanscore.append(scoreelm)
    derai.append(serai[i])

column = [row[0] for row in meanscore]
row = [row[1] for row in meanscore]
scorenp = np.array(meanscore)

maxclsl = []
for ai in range(2,7):
    if len(scorenp)> 1:
        cluster = hdbscan.HDBSCAN(min_cluster_size=ai).fit_predict(scorenp)
        cntmin = np.sum(cluster < 0)
        maxclsElm = [cntmin,max(cluster),cluster,ai]
        maxclsl.append(maxclsElm)
    maxclsls = sorted(maxclsl, key=lambda x:(x[3]),reverse=True)
    algomin = 99
    for i in range(0,len(maxclsls)):
        if maxclsls[i][0] == 0:
            if algomin > maxclsls[i][1]:
                if maxclsls[i][1] > 1:
                    algomin = maxclsls[i][1]
                    selectc = maxclsls[i][2]
    if algomin == 99:
        outmin = 99
        j = 0
        for i in range(0,len(maxclsls)):
            if maxclsls[i][0] < outmin:
                outmin = maxclsls[i][0]
                j = i
            if outmin != 99:
                selectc = maxclsls[j][2]

cntMinus = 0
for i in range(0,len(derai)): 
    ncElm = derai[i]
    if selectc[i] < 0:
        ncElm[10] = 88
    else:
        ncElm[10] = selectc[i]
    cherai.append(ncElm)

serai = sorted(cherai, key=lambda x:(x[0],x[11],x[10],x[5])) 


for i in reversed(range(1, len(serai))):
    if abs(serai[i - 1][5] - serai[i][5]) < 5 and abs(serai[i - 1][6] - serai[i][6]) < 5 and \
       abs(serai[i - 1][7] - serai[i][7]) < 5 and abs(serai[i - 1][8] - serai[i][8]) < 5 and \
       (serai[i - 1][0] == serai[i][0]) and (serai[i - 1][3] == serai[i][3]) and \
       (serai[i - 1][11] == serai[i][11]) and (serai[i - 1][10] == serai[i][10]):
        if (serai[i - 1][4] > serai[i][4]):
            perai = serai.pop(i)
        else:
            perai = serai.pop(i -1)
        j = i + 2

temperai = []

Groupnum = 1
terai = []
teraiElm = [serai[0][0],serai[0][1],serai[0][2],serai[0][3],serai[0][4],serai[0][5],
            serai[0][6],serai[0][7],serai[0][8],serai[0][9],serai[0][10],serai[0][11],0,0,0]
terai.append(teraiElm)
for i in range(1, len(serai)):
    if ((serai[i][0]!=serai[i-1][0]) or (serai[i][11]!=serai[i-1][11]) or (serai[i][10]!=serai[i-1][10])) or \
       (abs(serai[i][5]-serai[i-1][5]) > 10 and abs(serai[i][7]-serai[i-1][7]) > 10 and abs(serai[i][8]-serai[i-1][8]) > 10) or \
        abs(serai[i][6]-serai[i-1][6]) > 30                                                                                                                                                                                                  :
        x78dif = 0
    else:
        x78dif = serai[i][7]-serai[i -1][8]
    
    if x78dif <  -10:
        terai[i - 1][13] = Groupnum 
        teraiElm = [serai[i][0],serai[i][1],serai[i][2],serai[i][3],serai[i][4],serai[i][5],
            serai[i][6],serai[i][7],serai[i][8],serai[i][9],serai[i][10],serai[i][11],x78dif,Groupnum,0]
        terai.append(teraiElm)
    else:
        Groupnum =  Groupnum + 1
        teraiElm = [serai[i][0],serai[i][1],serai[i][2],serai[i][3],serai[i][4],serai[i][5],
            serai[i][6],serai[i][7],serai[i][8],serai[i][9],serai[i][10],serai[i][11],x78dif,0,0] 
        terai.append(teraiElm)


svGroup = 0
tterai = []
ttterai = []
erakunai = []
for i in range(0,len(terai)):
    if terai[i][13] > 0:
        if svGroup != terai[i][13]:
            if svGroup != 0:
                if len(tterai) > 0:
                    ttterai.append(tterai)
            svGroup = terai[i][13]        
            tterai = []
            tterai.append(terai[i])
        else:
            tterai.append(terai[i])
    else:
        kunaiElm = [terai[i][0],terai[i][1],terai[i][2],terai[i][3],terai[i][4],terai[i][5],terai[i][6],terai[i][7],
                    terai[i][8],terai[i][9],terai[i][10],terai[i][11],terai[i][12],terai[i][13],terai[i][14]]
        erakunai.append(kunaiElm)
        if len(tterai) > 0:
            ttterai.append(tterai)
            tterai = []
if len(tterai) > 0:
    ttterai.append(tterai)        


def DuplDelete(listx):
    steraid = []
    steraind = []
    sterai = listx
    for j in reversed(range(1,len(sterai))):
        absFour = []
        absFour.append(abs(sterai[j][5] - sterai[0][5]))
        absFour.append(abs(sterai[j][6] - sterai[0][6]))
        absFour.append(abs(sterai[j][7] - sterai[0][7]))
        absFour.append(abs(sterai[j][8] - sterai[0][8]))
        mchcnt = sum(x < 7 for x in absFour)     
        absFourS = sorted(absFour)
        s3sum = sum(heapq.nsmallest(3, absFourS))
        s2sum = sum(heapq.nsmallest(2, absFourS))
        s1min = min(absFourS)
        ratewidth = float(int(sterai[j][9]))/ float(int(sterai[0][9]))
        if ratewidth > 1.22 or ratewidth < 0.82:
            ratelbl = 3
        elif ratewidth > 1.213 or ratewidth < 0.825:
            ratelbl = 2
        elif ratewidth > 1.121 or ratewidth < 0.892:
            ratelbl = 1
        else:       
            ratelbl = 0
        jpegpngF = 0
        if sterai[j][1] != sterai[0][1]:
            jpegpngF = 1

        flgPop = 0
        if sterai[j][3] == sterai[0][3]:
            if mchcnt > 2:
                steraieLm14 = mchcnt * 10
                flgPop = 1
            elif mchcnt > 1:
                if s2sum < 6 or ratelbl > 2:
                    steraieLm14 = mchcnt * 10 + ratelbl
                    flgPop = 1
                else:
                    if s2sum < 8 and jpegpngF > 0:
                        steraieLm14 = 510 + ratelbl
                        flgPop = 1
                    else:
                        steraieLm14 = 910 + ratelbl
                        flgPop = 2
            elif mchcnt > 0:
                if s1min < 5 and ratelbl > 0:
                    steraieLm14 = mchcnt * 10 + ratelbl
                    flgPop = 1
                else:
                    steraieLm14 = 920 + ratelbl
                    flgPop = 2
            else:
                steraieLm14 = 930 + ratelbl
                flgPop = 2                
        else:
            if mchcnt > 3:
                steraieLm14 = mchcnt * 10 + ratelbl
                flgPop = 1
            elif mchcnt > 2:
                if s3sum < 6 or ratelbl > 0:                
                    steraieLm14 = mchcnt * 10 + ratelbl
                    flgPop = 1
                else:
                    if s2sum < 3 and jpegpngF > 0:
                        steraieLm14 = 520 + ratelbl
                        flgPop = 1
                    else:
                        steraieLm14 = 940 + ratelbl
                        flgPop = 2
            elif mchcnt > 1:
                if ratelbl > 2:
                    steraieLm14 = mchcnt * 10 + ratelbl
                    flgPop = 1
                elif s2sum < 5 and jpegpngF > 0:
                    steraieLm14 = 530 + ratelbl
                    flgPop = 1
                else:
                    steraieLm14 = 950 + ratelbl
                    flgPop = 2
            elif mchcnt > 0:
                if s1min < 2 and ratelbl > 2: 
                    steraieLm14 = mchcnt * 10 + ratelbl
                    flgPop = 1                      
                else:
                    steraieLm14 = 960 + ratelbl
                    flgPop = 2
            else:
                steraieLm14 = 970 + ratelbl
                flgPop = 2      
        kElm = [sterai[j][0],sterai[j][1],sterai[j][2],sterai[j][3],sterai[j][4],sterai[j][5],sterai[j][6],
               sterai[j][7],sterai[j][8],sterai[j][9],sterai[j][10],sterai[j][11],sterai[j][12],sterai[j][13],steraieLm14]

        if flgPop == 1 and sterai[j][4] <= sterai[0][4]:
            pop = sterai.pop(j)
            steraid.append(kElm)
        else:
            pop = sterai.pop(j)
            steraind.append(kElm)            
    return sterai,steraid,steraind

tderai = []
for i in range(0,len(ttterai)):
    stterai = sorted(ttterai[i], key=lambda x:(x[4]),reverse=True)
    stterai[0][14] = 999
    l1, l2, l3 = DuplDelete(stterai)
    if len(l3) > 0:
        if len(l2) > 0:
            erakunai.extend(l2)
        l1.extend(l3)
#        print(l1)
        tderai = sorted(l1, key=lambda x:(x[14]))
        l1, l2, l3 = DuplDelete(tderai)
        erakunai.extend(l1)
        erakunai.extend(l2)
        erakunai.extend(l3)
    else:
        erakunai.extend(l2)        
        erakunai.extend(l1)

kunai = sorted(erakunai, key=lambda x:(x[0],x[11],x[10],x[5])) 

print("kunai", len(kunai))

for i in reversed(range(0, len(kunai))):
    if (kunai[i][14] > 0) and (kunai[i][14] < 900): 
            perai = kunai.pop(i)

wbt = Workbook()
wst = wbt.active
j = 2
#derai.append(serai[0]) 
for i in range(0, len(kunai) - 1):
    if abs(kunai[i + 1][5] - kunai[i][5]) < 5 and abs(kunai[i + 1][6] - kunai[i][6]) < 5 and \
       (kunai[i + 1][0] == kunai[i][0]) and \
       (kunai[i + 1][11] == kunai[i][11]) and (kunai[i + 1][10] == kunai[i][10]):
        temperai.append(kunai[i])
        temperai.append(kunai[i + 1])
    else:
        if abs(kunai[i + 1][5] - kunai[i][5]) < 16 and abs(kunai[i + 1][6] - kunai[i][6]) < 4 and \
         (kunai[i + 1][0] == kunai[i][0]) and \
         (kunai[i + 1][11] == kunai[i][11]) and (kunai[i + 1][10] == kunai[i][10]) and \
         ((kunai[i + 1][9] / kunai[i][9] < 0.9) or (kunai[i][9] / kunai[i + 1][9] < 0.9)):
            temperai.append(kunai[i])
            temperai.append(kunai[i + 1])
        else:
            if len(temperai) > 0:
                temperai.append(kunai[i])
                strai = sorted(temperai, key=lambda x:(x[0],x[4]),reverse=True)
                mxrai = sorted(temperai, key=lambda x:(x[0],x[8]),reverse=True)
                mirai = sorted(temperai, key=lambda x:(x[0],x[7]))
#                       fn:0         ft:1        char:2      cls:3       rate:4      tx:5        ty:6     
#                       le:7         re:8        aw:9          CC:10      lc:11
                tElm = [strai[0][0],strai[0][1],strai[0][2],strai[0][3],strai[0][4],strai[0][5],strai[0][6],
                        mirai[0][7],mxrai[0][8],strai[0][9],strai[0][10],strai[0][11],strai[0][12],strai[0][13],strai[0][14]]
                derai.append(tElm)

                for i in range(0,len(temperai)):
                    j = j + i
                    wst.cell(row=j, column=2).value = temperai[i][0]
                    wst.cell(row=j, column=3).value = temperai[i][1]
                    wst.cell(row=j, column=4).value = temperai[i][2]
                    wst.cell(row=j, column=5).value = temperai[i][3]
                    wst.cell(row=j, column=6).value = temperai[i][4]
                    wst.cell(row=j, column=7).value = temperai[i][5]
                    wst.cell(row=j, column=8).value = temperai[i][6]
                    wst.cell(row=j, column=9).value = temperai[i][7]
                    wst.cell(row=j, column=10).value = temperai[i][8]
                    wst.cell(row=j, column=11).value = temperai[i][9]
                    wst.cell(row=j, column=12).value = temperai[i][10]
                    wst.cell(row=j, column=13).value = temperai[i][11]
#  wst.cell(row=j, column=14).value = serai[i][12]
                wst.cell(row=j, column=22).value = tElm[0]
                wst.cell(row=j, column=23).value = tElm[1]
                wst.cell(row=j, column=24).value = tElm[2]
                wst.cell(row=j, column=25).value = tElm[3]
                wst.cell(row=j, column=26).value = tElm[4]
                wst.cell(row=j, column=27).value = tElm[5]
                wst.cell(row=j, column=28).value = tElm[6]
                wst.cell(row=j, column=29).value = tElm[7]
                wst.cell(row=j, column=30).value = tElm[8]
                wst.cell(row=j, column=31).value = tElm[9]
                wst.cell(row=j, column=32).value = tElm[10]
                wst.cell(row=j, column=33).value = tElm[11]
                temperai = []
            else:
                derai.append(kunai[i])

wbt.save(dirOfPredx + "exNamevacktemperai" + ".xlsx") 



sdp = sorted(derai, key=lambda x:(x[0],x[11],x[5]))

print("len-sdp",len(sdp))

eraifilesv = ""
erailcls1sv = 0
erailcls2sv = 0
xerai = []

#          fn:0       ft:1     char:2     cls:3    rate:4     tx:5       ty:6     le:7       re:8     aw:9     CC:10      lc:11   cc2:12 
eraElm = [sdp[0][0],sdp[0][1],sdp[0][2],sdp[0][3],sdp[0][4],sdp[0][5],sdp[0][6],sdp[0][7],sdp[0][8],sdp[0][9],sdp[0][10],sdp[0][11],1]
#print(eraElm)
xerai.append(eraElm)
colc = 1
for i in range(1,len(sdp)):
    eraifile = sdp[i][0]
    erailcls1 = sdp[i][11]
    erailcls2 = sdp[i][10]
    npos = sdp[i-1][8]
    ppos = sdp[i][7]
    nupos = sdp[i-1][6]
    cppos = sdp[i][6]
    if eraifilesv != eraifile or erailcls1sv != erailcls1:
        npos = sdp[i][7]
        nupos = sdp[i][6]
        eraifilesv = eraifile
        erailcls1sv = erailcls1
        colc = 1
    if erailcls2sv != erailcls2:
        if abs(ppos - npos) < 6 and abs(cppos - nupos) < 6:
#            print("connect",eraifile,erailcls1,erailcls2sv,"-",erailcls2)
            colc = colc
            erailcls2sv = erailcls2
        else:
            npos = sdp[i][7]
            nupos = sdp[i][6]
            erailcls2sv = erailcls2
            colc = colc + 1
    diffx = abs(ppos - npos)
    diffy = abs(cppos - nupos)
    if diffy > 30:
            colc = colc + 1
    elif diffx > 14:
            colc = colc + 1
#               fn:0     ft:1     char:2     cls:3    rate:4     tx:5       ty:6     le:7      
#               re:8     aw:9     CC:10      lc:11   cc2:12 
    eraEsp = [sdp[i][0],sdp[i][1],sdp[i][2],sdp[i][3],sdp[i][4],sdp[i][5],sdp[i][6],sdp[i][7],
              sdp[i][8],sdp[i][9],sdp[i][10],sdp[i][11],colc]
    xerai.append(eraEsp)

sspt = sorted(xerai, key=lambda x:(x[0],x[11],x[5])) 

#print(xerai)  

wbt = Workbook()
wst = wbt.active
for i in range(0,len(xerai)):
  j = i + 2
  wst.cell(row=j, column=2).value = xerai[i][0]
  wst.cell(row=j, column=3).value = xerai[i][1]
  wst.cell(row=j, column=4).value = xerai[i][2]
  wst.cell(row=j, column=5).value = xerai[i][3]
  wst.cell(row=j, column=6).value = xerai[i][4]
  wst.cell(row=j, column=7).value = xerai[i][5]
  wst.cell(row=j, column=8).value = xerai[i][6]
  wst.cell(row=j, column=9).value = xerai[i][7]
  wst.cell(row=j, column=10).value = xerai[i][8]
  wst.cell(row=j, column=11).value = xerai[i][9]
  wst.cell(row=j, column=12).value = xerai[i][10]
  wst.cell(row=j, column=13).value = xerai[i][11]
  wst.cell(row=j, column=14).value = xerai[i][12]
wbt.save(dirOfPredx + "exNameerai" + ".xlsx") 

exOutNmsv = ""
rowidysv = 0
colidxsv = 0
charclxsv = 0
charclysv = 0
ratesv = 0
writeCnt = 0
charstr = ""

for clsElm in xerai:  
    exOutNm = clsElm[0]
    if exOutNmsv != exOutNm:
        if exOutNmsv != "":
            wb.save(dirOfPredx + exOutNmsv + ".xlsx")    
        wb = Workbook()
        ws = wb.active          
        exOutNmsv = exOutNm

        column_width = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1,'F': 1, 'G': 1, 'H': 1,
                    'I': 1, 'J': 1, 'K': 1, 'L': 1, 'M': 1,'N': 1, 'O': 1, 'P': 1,
                    'Q': 1, 'R': 1, 'S': 1, 'T': 1, 'U': 1,'V': 1, 'W': 1, 'X': 1,
                    'Y': 1, 'Z': 1, 'AA': 1, 'AB': 1, 'AC': 1,'AD': 1, 'AE': 1, 'AF': 1,
                    'AG': 1, 'AH': 1, 'AI': 1, 'AJ': 1, 'AK': 1,'AL': 1, 'AM': 1, 'AN': 1,
                    'AO': 1, 'AP': 1, 'AQ': 1, 'AR': 1, 'AS': 1,'AT': 1, 'AU': 1, 'AV': 1,
                    'AW': 1, 'AX': 1, 'AY': 1, 'AZ': 1,
                    'BA': 1, 'BB': 1, 'BC': 1, 'BD': 1,'BE': 1, 'BF': 1,
                    'BG': 1, 'BH': 1, 'BI': 1, 'BJ': 1, 'BK': 1,'BL': 1, 'BM': 1, 'BN': 1,
                    'BO': 1, 'BP': 1, 'BQ': 1, 'BR': 1, 'BS': 1,'BT': 1, 'BU': 1, 'BV': 1,
                    'BW': 1, 'BX': 1, 'BY': 1, 'BZ': 1,
                    'CA': 1, 'CB': 1, 'CC': 1, 'CD': 1,'CE': 1, 'CF': 1,
                    'CG': 1, 'CH': 1, 'CI': 1, 'CJ': 1, 'CK': 1,'CL': 1, 'CM': 1, 'CN': 1,
                    'CO': 1, 'CP': 1, 'CQ': 1, 'CR': 1, 'CS': 1,'CT': 1, 'CU': 1, 'CV': 1,
                    'CW': 1, 'CX': 1, 'CY': 1, 'CZ': 1,
                    'DA': 1, 'DB': 1, 'DC': 1, 'DD': 1,'DE': 1, 'DF': 1,
                    'DG': 1, 'DH': 1, 'DI': 1, 'DJ': 1, 'DK': 1,'DL': 1, 'DM': 1, 'DN': 1,
                    'DO': 1, 'DP': 1, 'DQ': 1, 'DR': 1, 'DS': 1,'DT': 1, 'DU': 1, 'DV': 1,
                    'DW': 1, 'DX': 1, 'DY': 1, 'DZ': 1,  
                    'EA': 1, 'EB': 1, 'EC': 1, 'ED': 1,'EE': 1, 'EF': 1,
                    'EG': 1, 'EH': 1, 'EI': 1, 'EJ': 1, 'EK': 1,'EL': 1, 'EM': 1, 'EN': 1,
                    'EO': 1, 'EP': 1, 'EQ': 1, 'ER': 1, 'ES': 1,'ET': 1, 'EU': 1, 'EV': 1,
                    'EW': 1, 'EX': 1, 'EY': 1, 'EZ': 1,
                    'FA': 1, 'FB': 1, 'FC': 1, 'FD': 1,'FE': 1, 'FF': 1,
                    'FG': 1, 'FH': 1, 'FI': 1, 'FJ': 1, 'FK': 1,'FL': 1, 'FM': 1, 'FN': 1,
                    'FO': 1, 'FP': 1, 'FQ': 1, 'FR': 1, 'FS': 1,'FT': 1, 'FU': 1, 'FV': 1,
                    'FW': 1, 'FX': 1, 'FY': 1, 'FZ': 1,
                    'GA': 1, 'GB': 1, 'GC': 1, 'GD': 1,'GE': 1, 'GF': 1,
                    'GG': 1, 'GH': 1, 'GI': 1, 'GJ': 1, 'GK': 1,'GL': 1, 'GM': 1, 'GN': 1,
                    'GO': 1, 'GP': 1, 'GQ': 1, 'GR': 1, 'GS': 1,'GT': 1, 'GU': 1, 'GV': 1,
                    'GW': 1, 'GX': 1, 'GY': 1, 'GZ': 1,
                    'HA': 1, 'HB': 1, 'HC': 1, 'HD': 1,'HE': 1, 'HF': 1,
                    'HG': 1, 'HH': 1, 'HI': 1, 'HJ': 1, 'HK': 1,'HL': 1, 'HM': 1, 'HN': 1,
                    'HO': 1, 'HP': 1, 'HQ': 1, 'HR': 1, 'HS': 1,'HT': 1, 'HU': 1, 'HV': 1,
                    'HW': 1, 'HX': 1, 'HY': 1, 'HZ': 1,
                    'IA': 1, 'IB': 1, 'IC': 1, 'ID': 1,'IE': 1, 'IF': 1,
                    'IG': 1, 'IH': 1, 'II': 1, 'IJ': 1, 'IK': 1,'IL': 1, 'IM': 1, 'IN': 1,
                    'IO': 1, 'IP': 1, 'IQ': 1, 'IR': 1, 'IS': 1,'IT': 1, 'IU': 1, 'IV': 1,
                    'IW': 1, 'IX': 1, 'IY': 1, 'IZ': 1,
                    'JA': 1, 'JB': 1, 'JC': 1, 'JD': 1,'JE': 1, 'JF': 1,
                    'JG': 1, 'JH': 1, 'JI': 1, 'JJ': 1, 'JK': 1,'JL': 1, 'JM': 1, 'JN': 1,
                    'JO': 1, 'JP': 1, 'JQ': 1, 'JR': 1, 'JS': 1,'JT': 1, 'JU': 1, 'JV': 1,
                    'JW': 1, 'JX': 1, 'JY': 1, 'JZ': 1,
                    'KA': 1, 'KB': 1, 'KC': 1, 'KD': 1,'KE': 1, 'KF': 1,
                    'KG': 1, 'KH': 1, 'KI': 1, 'KJ': 1, 'KK': 1,'KL': 1, 'KM': 1, 'KN': 1,
                    'KO': 1, 'KP': 1, 'KQ': 1, 'KR': 1, 'KS': 1,'KT': 1, 'KU': 1, 'KV': 1,
                    'KW': 1, 'KX': 1, 'KY': 1, 'KZ': 1,
                    'LA': 1, 'LB': 1, 'LC': 1, 'LD': 1,'LE': 1, 'LF': 1,
                    'LG': 1, 'LH': 1, 'LI': 1, 'LJ': 1, 'LK': 1,'LL': 1, 'LM': 1, 'LN': 1,
                    'LO': 1, 'LP': 1, 'LQ': 1, 'LR': 1, 'LS': 1,'LT': 1, 'LU': 1, 'LV': 1,
                    'LW': 1, 'LX': 1, 'LY': 1, 'LZ': 1,
                    'MA': 1, 'MB': 1, 'MC': 1, 'MD': 1,'ME': 1, 'MF': 1,
                    'MG': 1, 'MH': 1, 'MI': 1, 'MJ': 1, 'MK': 1,'ML': 1, 'MM': 1, 'MN': 1,
                    'MO': 1, 'MP': 1, 'MQ': 1, 'MR': 1, 'MS': 1,'MT': 1, 'MU': 1, 'MV': 1,
                    'MW': 1, 'MX': 1, 'MY': 1, 'MZ': 1,
                    'NA': 1, 'NB': 1, 'NC': 1, 'ND': 1,'NE': 1, 'NF': 1,
                    'NG': 1, 'NH': 1, 'NI': 1, 'NJ': 1, 'NK': 1,'NL': 1, 'NM': 1, 'NN': 1,
                    'NO': 1, 'NP': 1, 'NQ': 1, 'NR': 1, 'NS': 1,'NT': 1, 'NU': 1, 'NV': 1,
                    'NW': 1, 'NX': 1, 'NY': 1, 'NZ': 1,
                    'OA': 1, 'OB': 1, 'OC': 1, 'OD': 1,'OE': 1, 'OF': 1,
                    'OG': 1, 'OH': 1, 'OI': 1, 'OJ': 1, 'OK': 1,'OL': 1, 'OM': 1, 'ON': 1,
                    'OO': 1, 'OP': 1, 'OQ': 1, 'OR': 1, 'OS': 1,'OT': 1, 'OU': 1, 'OV': 1,
                    'OW': 1, 'OX': 1, 'OY': 1, 'OZ': 1,
                    'PA': 1, 'PB': 1, 'PC': 1, 'PD': 1,'PE': 1, 'PF': 1,
                    'PG': 1, 'PH': 1, 'PI': 1, 'PJ': 1, 'PK': 1,'PL': 1, 'PM': 1, 'PN': 1,
                    'PO': 1, 'PP': 1, 'PQ': 1, 'PR': 1, 'PS': 1,'PT': 1, 'PU': 1, 'PV': 1,
                    'PW': 1, 'PX': 1, 'PY': 1, 'PZ': 1,
                    'QA': 1, 'QB': 1, 'QC': 1, 'QD': 1,'QE': 1, 'QF': 1,
                    'QG': 1, 'QH': 1, 'QI': 1, 'QJ': 1, 'QK': 1,'QL': 1, 'QM': 1, 'QN': 1,
                    'QO': 1, 'QP': 1, 'QQ': 1, 'QR': 1, 'QS': 1,'QT': 1, 'QU': 1, 'QV': 1,
                    'QW': 1, 'QX': 1, 'QY': 1, 'QZ': 1,
                    'RA': 1, 'RB': 1, 'RC': 1, 'RD': 1,'RE': 1, 'RF': 1,
                    'RG': 1, 'RH': 1, 'RI': 1, 'RJ': 1, 'RK': 1,'RL': 1, 'RM': 1, 'RN': 1,
                    'RO': 1, 'RP': 1, 'RQ': 1, 'RR': 1, 'RS': 1,'RT': 1, 'RU': 1, 'RV': 1,
                    'RW': 1, 'RX': 1, 'RY': 1, 'RZ': 1,
                    'SA': 1, 'SB': 1, 'SC': 1, 'SD': 1,'SE': 1, 'SF': 1,
                    'SG': 1, 'SH': 1, 'SI': 1, 'SJ': 1, 'SK': 1,'SL': 1, 'SM': 1, 'SN': 1,
                    'SO': 1, 'SP': 1, 'SQ': 1, 'SR': 1, 'SS': 1,'ST': 1, 'SU': 1, 'SV': 1,
                    'SW': 1, 'SX': 1, 'SY': 1, 'SZ': 1,
                    'TA': 1, 'TB': 1, 'TC': 1, 'TD': 1,'TE': 1, 'TF': 1,
                    'TG': 1, 'TH': 1, 'TI': 1, 'TJ': 1, 'TK': 1,'TL': 1, 'TM': 1, 'TN': 1,
                    'TO': 1, 'TP': 1, 'TQ': 1, 'TR': 1, 'TS': 1,'TT': 1, 'TU': 1, 'TV': 1,
                    'TW': 1, 'TX': 1, 'TY': 1, 'TZ': 1}

        for col, width in column_width.items():
            ws.column_dimensions[col].width = width

    if charclysv != clsElm[11]:
        if len(charstr) > 0: 
            ws.cell(row=rowidy, column=colidx).value = charstr
            ws.cell(row=rowidy, column=colidx).alignment = Alignment(vertical='top')  
        charstr = ""
        charclysv = clsElm[11]
        charclxsv = clsElm[12]
        colidx = int(int(clsElm[5] / 10)) + 1
    else:
        if charclxsv != clsElm[12]:
            if len(charstr) > 0: 
                ws.cell(row=rowidy, column=colidx).value = charstr
                ws.cell(row=rowidy, column=colidx).alignment = Alignment(vertical='top')
            charstr = ""
            charclxsv = clsElm[12]
            colidx = int(int(clsElm[5] / 10)) + 1
    
    #rowidy = int(clsElm[11]) * 2 + 1
    rowidy = int(int(clsElm[6] / 25)) + 1
    ws.row_dimensions[rowidy].height = 20

    if str(clsElm[2]) == "MinusMinus":
        char = "-"
    else:
        if str(clsElm[2]) == "DotDot":
            char = "."
        else:
            char = str(clsElm[2])
    charstr = charstr + char

wb.save(dirOfPredx + exOutNmsv + ".xlsx")      