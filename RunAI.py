from ultralytics import YOLO
import subprocess

runPath = "C:/yolo/AI-Pred-Data/MIpyprep.bat"
proc = subprocess.Popen(runPath)
result = proc.communicate() 
print("Prepared Floders")

command = ["python.exe", "ultralytics/detectes11pngx.py"]
proc = subprocess.Popen(command)  #->コマンドが実行される(処理の終了は待たない)
result = proc.communicate()  #->終了を待つ
print("png-images prepared")

command2 = ["python.exe", "ultralytics/devyolo11.py"]
proc = subprocess.Popen(command2)  
result = proc.communicate() 
print("org-images cropped")

command3 = ["python.exe", "ultralytics/devyolo211.py"]
proc = subprocess.Popen(command3)  
result = proc.communicate() 
print("sub-images cropped")

#model = YOLO("ultralytics/beste200.pt")
model = YOLO("ultralytics/bestse100.pt")
# --save-txt --save-csv --line-thickness 1 --hide-labels --conf 0.20
results = model("C:/yolo/AI-Pred-Data/step2/",save=True,save_txt=True,save_conf=True,show_labels=False,show_conf=False,line_width=1)

command4 = ["python.exe", "ultralytics/concyolyo11.py"]
proc = subprocess.Popen(command4)  
result = proc.communicate() 
print("org-images pred")

command5 = ["python.exe", "ultralytics/concyolyp11.py"]
proc = subprocess.Popen(command5)  
result = proc.communicate() 
print("sub-images pred")

command6 = ["python.exe", "ultralytics/concyolojpx11.py"]
proc = subprocess.Popen(command6)  
result = proc.communicate() 
print("main-sub marged")

command7 = ["python.exe", "ultralytics/txtcsv.py"]
proc = subprocess.Popen(command7)  
result = proc.communicate() 
print("txtcsv Pi Bi")

command8 = ["python.exe", "ultralytics/bigRect.py"]
proc = subprocess.Popen(command8)  
result = proc.communicate() 
print("image Pi Bi")

command9 = ["python.exe", "ultralytics/copyafterV11.py"]
proc = subprocess.Popen(command9)  
result = proc.communicate() 
print("Pi Bi")