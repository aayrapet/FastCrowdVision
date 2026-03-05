from utils import center_to_corner
from PIL import Image
from torchvision import transforms
import torch
import glob
import torch.nn as nn
from  torch.utils.data.distributed import DistributedSampler


class DataSSD300(torch.utils.data.Dataset):
    """
    Load resized to 300*300 resolution image

    return:
        img_tensor tensor of resized image
        label_list tensor of labels
        gt_box tensor of gt boxes

    rem:
        len(gt_box)=len(label_list)

    """

    def __init__(self, img_dir, lbl_dir, gt_normalised: bool = True):
        self.images = sorted(glob.glob(img_dir + "/*.jpg"))
        self.labels = sorted(glob.glob(lbl_dir + "/*.txt"))
        self.transform = transforms.Compose(
            [transforms.Resize((300, 300)), transforms.ToTensor()]
        )
        self.gt_normalised = gt_normalised

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # images
        img = Image.open(self.images[idx]).convert("RGB")

        if not self.gt_normalised:
            W, H = img.size
        else:
            H, W = 1, 1
        img_tensor = self.transform(img)

        # labels and gt boxes
        with open(self.labels[idx]) as f:
            gt_box = []
            label_list = []
            for line in f:
                label, cx, cy, w, h = map(float, line.split())

                gt_box.append((cx / W, cy / H, w / W, h / H))

                label_list.append(
                    label + 1
                )  # yolo labels start at 0, my ssd start at 1, O is BG
            gt_box = center_to_corner(torch.tensor(gt_box, dtype=torch.float32) )
            #just to be sure we clamp to [0,1]
            gt_box = gt_box.clamp(min=0, max=1)

            label_list = torch.tensor(label_list, dtype=torch.int64)

        return img_tensor, label_list, gt_box


class DataSplitter(nn.Module):
    """
    Split into training, validation, testing dataloaders
    """

    def __init__(self, batch_size: int, test_size: float, val_size: float,multigpu =False):
        super().__init__()
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.multigpu=multigpu

    @staticmethod
    def collate_ssd(batch):
        images, labels, boxes = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, list(labels), list(boxes)

    def forward(self, dataset: DataSSD300):

        # https://stackoverflow.com/questions/65138643/examples-or-explanations-of-pytorch-dataloaders

        test_amount, val_amount = int(len(dataset) * self.test_size), int(
            len(dataset) * self.val_size
        )

        # this function will automatically randomly split your dataset but you could also implement the split yourself
        train_set, test_set,val_set = torch.utils.data.random_split(
            dataset,
            [(len(dataset) - (test_amount + val_amount)), test_amount, val_amount],
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=False if self.multigpu else True,
            collate_fn=self.collate_ssd,
            sampler=DistributedSampler(train_set) if self.multigpu else None
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False if self.multigpu else True,
            collate_fn=self.collate_ssd,
            sampler=DistributedSampler(val_set) if self.multigpu else None
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False if self.multigpu else True,
            collate_fn=self.collate_ssd,
            sampler=DistributedSampler(test_set) if self.multigpu else None
        )

        return train_dataloader, val_dataloader, test_dataloader
