import os
import s3fs

fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"},
    anon=True
)

BUCKET = "aayrapetyan"

def load_data():
    cwd = os.getcwd()
    project_root = cwd.split("FastCrowdVision")[0] + "FastCrowdVision"

    train_dir = os.path.join(project_root, "datasets", "voc", "train")
    val_dir = os.path.join(project_root, "datasets", "voc", "val")
    test_dir = os.path.join(project_root, "datasets", "voc", "test")

    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    downloads = [
        (f"{BUCKET}/FastCrowdVision/datasets/voc/train/images", train_dir),
        (f"{BUCKET}/FastCrowdVision/datasets/voc/train/labels", train_dir),

        (f"{BUCKET}/FastCrowdVision/datasets/voc/val/images", val_dir),
        (f"{BUCKET}/FastCrowdVision/datasets/voc/val/labels", val_dir),

        (f"{BUCKET}/FastCrowdVision/datasets/voc/test/images", test_dir),
        (f"{BUCKET}/FastCrowdVision/datasets/voc/test/labels", test_dir),
    ]

    for s3_path, local_parent in downloads:
        name = s3_path.split("/")[-1]
        print(f"Downloading {name} to {local_parent}/{name} ...")
        fs.get(s3_path, os.path.join(local_parent, name), recursive=True)

    print("Download completed.")
    print("Train images:", os.path.join(train_dir, "images"))
    print("Train labels:", os.path.join(train_dir, "labels"))
    print("Val images:", os.path.join(val_dir, "images"))
    print("Val labels:", os.path.join(val_dir, "labels"))
    print("Test images:", os.path.join(test_dir, "images"))
    print("Test labels:", os.path.join(test_dir, "labels"))

if __name__ == "__main__":
    load_data()