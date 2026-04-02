
"all data from here is already processed and loaded to S3 using kaggle/download.py and configuration in sspcloud.fr"
import os 
import s3fs
fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"},
    anon=True  
)
BUCKET="aayrapetyan"

def load_data():
    cwd = os.getcwd()
    project_root = cwd.split("FastCrowdVision")[0] + "FastCrowdVision"

    img_dir = os.path.join(project_root, "datasets", "voc", "images")
    lbl_dir = os.path.join(project_root, "datasets", "voc", "labels")

    print("Images path", img_dir)
    print("Labels path", lbl_dir)

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)


    images = fs.ls(f"{BUCKET}/FastCrowdVision/datasets/voc/JPEGImages/JPEGImages")
    labels = fs.ls(f"{BUCKET}/FastCrowdVision/datasets/voc/labels/labels")

    fs.get(images, img_dir)
    fs.get(labels, lbl_dir)

if __name__=="__main__":
    load_data()