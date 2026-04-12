
"""
voc2007 train and val data 
This file deals with raw data from Kaggle Hub, so we will upload it, then transform to YOLO format and save 
"""
import json 
import glob
import xml.etree.ElementTree as ET
import re 
import shutil
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



folder_test_data_images=os.path.join(path, "VOCtest_06-Nov-2007", "VOCdevkit", "VOC2007", "JPEGImages")
folder_test_data_annot=os.path.join(path, "VOCtest_06-Nov-2007", "VOCdevkit", "VOC2007", "Annotations")



def verif_counts(func) :
        def wrapper(*args):
            x,y,mode= func(*args)
            yy=sorted(glob.glob(y + "/*.jpg"))
            if mode!="test":
                xx=sorted(glob.glob(x + "/*.txt"))
                if len(xx)!=len(yy):
                    raise ValueError("Number of elements inside /labels and /images does not match, corrupted data")
            print("for",mode,"dataset there is",len(yy),"observations")
            return x,y,mode
        return wrapper

def convert_box(size, box):
                dw, dh = 1.0 / size[0], 1.0 / size[1]
                x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
                return x * dw, y * dh, w * dw, h * dh
def get_file_names(dictionary)->dict:
                res=[]
                alls=dictionary["images"]
                for el in alls:
                        res.append(el["file_name"].split('.')[0])
                return res 
@verif_counts
def load_train_val_data(kaggle_path,project_root,mode,img_dir,annot_dir):
   
        """
        mode : train, val only,since test has different folder 
        This code converts data to YOLO format 

        annotations transformations code borrowed from https://docs.ultralytics.com/datasets/detect/voc/ 

        """
        names = list(config.values())  # names list
        
        storage=os.path.join(project_root, "datasets", "voc",mode)

        if not os.path.isdir(storage) or not os.listdir(storage):

            print(f"downloading images and labels for {mode} dataset")

            os.makedirs(os.path.join(storage,"labels"), exist_ok=True)
            os.makedirs(os.path.join(storage, "images"), exist_ok=True)

            #id of image : image path
            images = {re.sub(r"\D", "", os.path.basename(f)): f for f in glob.glob(os.path.join(img_dir,"*"))}
            annots = {re.sub(r"\D", "", os.path.basename(f)) : f for f in glob.glob(os.path.join(annot_dir,"*"))}
            
            #look at all images/labels of mode_dataset
            
            mode_images_store=os.path.join(kaggle_path, "PASCAL_VOC", "PASCAL_VOC", f"pascal_{mode}2007.json")
            #get all info about mode images, which ones are in mode dataset
            with open(mode_images_store, "r", encoding="utf-8") as f:
                d = json.load(f)
        
            files=get_file_names(d)

           
            for file in files:
                image_file = images[file]
                image_path=os.path.join(storage,"images",f"{file}.jpg")
                annot_path = annots[file]

                tree = ET.parse(annot_path)
                root = tree.getroot()
                size = root.find("size")
                w = int(size.find("width").text)
                h = int(size.find("height").text)

                fileidx = root.find("filename").text[:-4]  
                
                lb_path=os.path.join(storage,"labels",f"label{fileidx}.txt")


                with open(lb_path, "w", encoding="utf-8") as out_file:
                        for obj in root.iter("object"):
                                cls = obj.find("name").text
                                if cls in names and int(obj.find("difficult").text) != 1:
                                        xmlbox = obj.find("bndbox")
                                        bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ("xmin", "xmax", "ymin", "ymax")])
                                        cls_id = names.index(cls)  # class id
                                        out_file.write(" ".join(str(a) for a in (cls_id, *bb)) + "\n")
                shutil.copy(image_file, image_path)
        else:
            print("you already have all required data")
        #return links where labels and images are stored 
        labels_path=os.path.join(storage,"labels")
        images_path=os.path.join(storage,"images")

        return labels_path, images_path,mode

@verif_counts
def load_test_data(kaggle_path,project_root,img_dir,annot_dir):
   
        """
        test dataset only 
        This code converts data to YOLO format 

        annotations transformations code borrowed from https://docs.ultralytics.com/datasets/detect/voc/ 

        """
        names = list(config.values())  # names list
        
        storage=os.path.join(project_root, "datasets", "voc","test")

        if not os.path.isdir(storage) or not os.listdir(storage):

            print(f"downloading images and labels for test dataset")

            os.makedirs(os.path.join(storage,"labels"), exist_ok=True)
            os.makedirs(os.path.join(storage, "images"), exist_ok=True)

            annot_files = sorted(glob.glob(os.path.join(annot_dir, "*.xml")))
            img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))


            for annot_file,img_file in zip(annot_files,img_files):
             
                name_image=re.sub(r"\D", "", os.path.basename(img_file))
                image_path=os.path.join(storage,"images",f"{name_image}.jpg")
                annot_path = annot_file

                tree = ET.parse(annot_path)
                root = tree.getroot()
                size = root.find("size")
                w = int(size.find("width").text)
                h = int(size.find("height").text)

                fileidx = root.find("filename").text[:-4]  
                
                lb_path=os.path.join(storage,"labels",f"label{name_image}.txt")


                with open(lb_path, "w", encoding="utf-8") as out_file:
                        for obj in root.iter("object"):
                                cls = obj.find("name").text
                                if cls in names and int(obj.find("difficult").text) != 1:
                                        xmlbox = obj.find("bndbox")
                                        bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ("xmin", "xmax", "ymin", "ymax")])
                                        cls_id = names.index(cls)  # class id
                                        out_file.write(" ".join(str(a) for a in (cls_id, *bb)) + "\n")
                shutil.copy(img_file, image_path)
        else:
            print("you already have all required data")
        #return links where labels and images are stored 
        labels_path=os.path.join(storage,"labels")
        images_path=os.path.join(storage,"images")

        return labels_path, images_path,"test"



if __name__=="__main__":
    labels_path, images_path, _ = load_train_val_data(path, project_root, "train", img_dir, annot_dir)
    print(f"Train loaded — labels: {labels_path}, images: {images_path}")

    labels_path, images_path, _ = load_train_val_data(path, project_root, "val", img_dir, annot_dir)
    print(f"Val loaded — labels: {labels_path}, images: {images_path}")
    #since test dataset is found in another dataset, we can't directly apply same function, 
    labels_path, images_path, _ = load_test_data(path, project_root, folder_test_data_images, folder_test_data_annot)
    print(f"Test loaded — labels: {labels_path}, images: {images_path}")