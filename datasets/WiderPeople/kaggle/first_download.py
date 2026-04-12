import kagglehub
import glob
import os 
import yaml
from PIL import Image
import shutil
import re
#this file to extract raw data in raw format from kaggle storage, preprocess it to yolo format

cwd=os.getcwd()
#independently from where you are, project root is this
project_root = cwd.split("FastCrowdVision")[0] + "FastCrowdVision"

#get names of labels
wide_dir=os.path.join(project_root, "datasets", "WiderPeople","widerpeople.yaml")

with open(wide_dir) as f:
    cfg = yaml.safe_load(f)
config = cfg["names"] 

path = kagglehub.dataset_download("imneonizer/wider-person")
print("Path to  files:", path)

img_dir = os.path.join(path, "Images")
annot_dir = os.path.join(path, "Annotations")


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

@verif_counts
def load_all_data(kaggle_path,project_root,mode,img_dir,annot_dir):
    """
    mode : train, test, val only 

    """

    def convert_box(size, box):
                    dw, dh = 1.0 / size[0], 1.0 / size[1]
                    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
                    return x * dw, y * dh, w * dw, h * dh

    storage=os.path.join(project_root, "datasets", "WiderPeople",mode)

    if not os.path.isdir(storage) or not os.listdir(storage):

        print(f"downloading images and labels for {mode} dataset")

        os.makedirs(os.path.join(storage,"labels"), exist_ok=True)
        os.makedirs(os.path.join(storage, "images"), exist_ok=True)

        #id of image : image path
        images = {re.sub(r"\D", "", os.path.basename(f)): f for f in glob.glob(os.path.join(img_dir,'*'))}
        for key,val in images.items():
             print(key,val)
             break
        if mode!="test":
            annots = {re.sub(r"\D", "", os.path.basename(f)) : f for f in glob.glob(os.path.join(annot_dir,'*'))}
        
        #look at all images/labels of mode_dataset
        links=os.path.join(kaggle_path, f"{mode}.txt")
        with open(links) as f:
            files = f.read().strip().splitlines()

        
        #fill in with files
        for file in files:
            image_file = images[file]
            image_path=os.path.join(storage,"images",f"image{file}.jpg")
            if mode!="test":
                #we dont have annotations for test 
                annot_file = annots[file]
                #in the dataset there is no info about H,W so need to access image.size 
                img = Image.open(image_file)
                w,h=img.size
                #define where at project root files will be stored 
                lb_path=os.path.join(storage,"labels",f"label{file}.txt")
                

                with open(annot_file) as f:
                        lines = f.read().strip().splitlines()

                #write label<i>.txt file with annotation found 
                with open(lb_path, "w", encoding="utf-8") as out_file:
                    for line in lines[1:]:
                        parts = line.split()
                        cls_id = int(parts[0])
                        #no ignore regions, no crowds 
                        if cls_id in (4, 5):
                            continue
                        xmin, ymin, xmax, ymax = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                        bb = convert_box((w, h), [xmin, xmax, ymin, ymax])
                        #this dataset classes start with 1, i need yolo format tarting at 0 
                        out_file.write(f"{cls_id-1} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n")

                if os.path.getsize(lb_path) > 0:
                    shutil.copy(image_file, image_path)
                else:
                    os.remove(lb_path)
            else:
                shutil.copy(image_file, image_path)
    else:
        print("you already have all required data")
    #return links where labels and images are stored 
    labels_path=os.path.join(storage,"labels")
    images_path=os.path.join(storage,"images")

    return labels_path, images_path,mode

if __name__=="__main__":

        labels_path, images_path, _ = load_all_data(path, project_root, "train", img_dir, annot_dir)
        print(f"Train loaded — labels: {labels_path}, images: {images_path}")

        labels_path, images_path, _ = load_all_data(path, project_root, "val", img_dir, annot_dir)
        print(f"Val loaded — labels: {labels_path}, images: {images_path}")

        labels_path, images_path, _ = load_all_data(path, project_root, "test", img_dir, annot_dir)
        print(f"Test loaded —  images: {images_path}")
