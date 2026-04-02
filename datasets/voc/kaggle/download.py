
"""
This file deals with raw data from Kaggle Hub, so we will upload it, then transform to YOLO format and save 
"""

import glob
import xml.etree.ElementTree as ET

import yaml
import os 
import kagglehub 
cwd=os.getcwd()
#independently from where you are, project root is this
project_root = cwd.split("FastCrowdVision")[0] + "FastCrowdVision"

#get names of labels
voc_dir=os.path.join(project_root, "datasets", "voc","voc.yaml")

with open(voc_dir) as f:
    cfg = yaml.safe_load(f)
config = cfg["names"] 

#download from kaggle voc data
path = kagglehub.dataset_download("zaraks/pascal-voc-2007")
print("Path to VOC files:", path)

#three main paths = images, annotations (raw format of labels, just to get data and forget), labels (the format we will use for training)
img_dir = os.path.join(path, "VOCtrainval_06-Nov-2007", "VOCdevkit", "VOC2007", "JPEGImages")
annot_dir = os.path.join(path, "VOCtrainval_06-Nov-2007", "VOCdevkit", "VOC2007", "Annotations")
labels_dir = os.path.join(path, "VOCtrainval_06-Nov-2007", "VOCdevkit", "VOC2007", "labels")


def convert_label(annot_path,config : dict,labels_dir ):
        """
        This code converts data to YOLO format 

        code borrowed from https://docs.ultralytics.com/datasets/detect/voc/ 

        """
        names = list(config.values())  # names list
        def convert_box(size, box):
                dw, dh = 1.0 / size[0], 1.0 / size[1]
                x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
                return x * dw, y * dh, w * dw, h * dh

        tree = ET.parse(annot_path)
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        fileidx = root.find("filename").text[:-4]  
        
        lb_path = labels_dir+f"/label{fileidx}.txt"

        with open(lb_path, "w", encoding="utf-8") as out_file:
                for obj in root.iter("object"):
                        cls = obj.find("name").text
                        if cls in names and int(obj.find("difficult").text) != 1:
                                xmlbox = obj.find("bndbox")
                                bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ("xmin", "xmax", "ymin", "ymax")])
                                cls_id = names.index(cls)  # class id
                                out_file.write(" ".join(str(a) for a in (cls_id, *bb)) + "\n")

if not os.path.isdir(labels_dir) or not os.listdir(labels_dir):
    os.makedirs(labels_dir, exist_ok=True)
    annot_files = sorted(glob.glob(os.path.join(annot_dir, "*.xml")))
    for f in annot_files:
        convert_label(f, config,labels_dir)
    print(f"Converted {len(annot_files)} annotations to YOLO format")
else:
    print("Labels already exist, skipping conversion")

img_count = len(glob.glob(img_dir + "/*.jpg"))
lbl_count = len(glob.glob(labels_dir + "/*.txt"))

print(f"Images: {img_count}")
print(f"Labels: {lbl_count}")

if img_count == lbl_count:
    print(f"OK: {img_count} images match {lbl_count} labels")
    print("Images path", img_dir)
    print("Labels path", labels_dir)
else:
    raise ValueError(f"Mismatch: {img_count} images vs {lbl_count} labels")