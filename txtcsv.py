import csv
import yaml
import glob
import os

csv_path1 = "C:/PylocalYolo/yolov5/data/mmletter/images/casestep5/"
csv_path2 = "predictions.csv"
csv_path = csv_path1 + csv_path2

with open('C:/PylocalYolo/yolov5/data/mmletter.yaml', 'r', encoding="utf-8") as yml:
    config = yaml.safe_load(yml)

dicyaml = config['names']

#filesp = glob.glob("C:/PylocalYolo/yolov5/runs/detect/exp/labels" + "/*.txt", recursive=True)
filesp = glob.glob("C:/PylocalYolo/yolov5/runs/detect/exp15/labels" + "/*.txt", recursive=True)

wa = "w"
for file in filesp:
    filename = os.path.basename(file)
    dotPos = filename.rfind(".")
#    bsPos = filename.rfind("\\")
    filenamenp = filename[:dotPos]
    print(filenamenp)
    if "PXM" in filename:
        filetype = ".png"
    else:
        filetype = ".jpg"

    with open(file, "r", encoding="utf-8") as f:
        for listelm in f:
            valofelm = listelm.split()
            clschr = dicyaml[int(valofelm[0])] 

            data = {"Image Name": filenamenp + filetype, "Prediction": valofelm[0], "Confidence": clschr}
            with open(csv_path, mode=wa, newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if wa == "w":
                    writer.writeheader()
                writer.writerow(data)
                wa = "a"