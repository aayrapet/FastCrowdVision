import kagglehub
import os

# Download dataset
path = kagglehub.dataset_download("michelmaximov/voc2007-yolo")

# Train paths
train_images_dir = os.path.join(path, "train", "images")
train_labels_dir = os.path.join(path, "train", "labels")

# Validation paths
val_images_dir = os.path.join(path, "val", "images")
val_labels_dir = os.path.join(path, "val", "labels")

# Test paths
test_images_dir = os.path.join(path, "test", "images")
test_labels_dir = os.path.join(path, "test", "labels")

# Print dataset info
print("Dataset root:", path)

print("\nTrain dataset:")
print("Images:", train_images_dir)
print("Labels:", train_labels_dir)

print("\nValidation dataset:")
print("Images:", val_images_dir)
print("Labels:", val_labels_dir)

print("\nTest dataset:")
print("Images:", test_images_dir)
print("Labels:", test_labels_dir)