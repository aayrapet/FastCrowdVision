import torch
import torch.nn as nn
from multiloss import MultiLoss
from eval import evaluation
from ssd import SSD
from torchmetrics.detection import MeanAveragePrecision
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from dotenv import load_dotenv

load_dotenv()
import os

wandb.login(key=os.getenv("WANDB_API_KEY"))
entity = os.getenv("ENTITY")
project = os.getenv("PROJECT")


def logwandb(
    epoch,multigpu, model_device, run, train_loss, val_loss, map_score, early_stop
):
    """
    saving logs with multiple gpus

    """
    if multigpu:
        if model_device.index == 0:
            run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "map_score": map_score,
                    "early_stop": early_stop,
                }
            )
    else:
        run.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "map_score": map_score,
                "early_stop": early_stop,
            }
        )


def adjust_lr_rate(optimizer, nb_times, gamma):
    # https://stackoverflow.com/questions/48324152/how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no-lr-sched
    lr = optimizer.param_groups[0]["lr"]
    lr = lr * (gamma)#**nb_times
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    modelname: str,
    gamma,
    lr_schedule_epochs: list,
    epoch_verbose: int = 5,
    early_stop_iter=30,
    start_epoch=0,
):
    
 
    model_device = next(model.parameters()).device
    metric = MeanAveragePrecision().to(model_device)
    multigpu = False
    model_attributes = model
    if model_device != torch.device("cpu"):
        if torch.cuda.device_count() > 1:
            device_id = model_device.index
            multigpu = True
            model = DDP(model, device_ids=[device_id])
            model_attributes = model.module
    if multigpu : 
        if model_device.index == 0:
            run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity=entity,
                # Set the wandb project where this run will be logged.
                project=project,
                # Track hyperparameters and run metadata.
                config={
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "modelname": modelname,
                    "lr_schedule_epochs": lr_schedule_epochs,
                    "epochs": model_attributes.N_epochs,
                },
            )
        else:
            run = None
    else:
        run = wandb.init(
                # Set the wandb entity where your project will be logged (generally your team name).
                entity=entity,
                # Set the wandb project where this run will be logged.
                project=project,
                # Track hyperparameters and run metadata.
                config={
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "modelname": modelname,
                    "lr_schedule_epochs": lr_schedule_epochs,
                    "epochs": model_attributes.N_epochs,
                },
            )

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    criterion = MultiLoss(anchors=model_attributes.anchors)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)
    N_epochs = model_attributes.N_epochs

    counter_early_stop = 0
    nb_lr_update = 1

    for epoch in range(start_epoch, N_epochs):

        if epoch in lr_schedule_epochs:
            adjust_lr_rate(optimizer, nb_lr_update, gamma)
            nb_lr_update = nb_lr_update + 1

        if multigpu:
            # https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            # note that shuffling is needed only for training , val we dont care
            train_dataloader.sampler.set_epoch(epoch)

        model_attributes.phase = "train"
        model.train()
        train_loss = 0
        train_samples = 0

        for image, labels_list, gt_box_list in train_dataloader:
            N_images = image.shape[0]
            image = image.to(model_device)
            labels_list = [lbl.to(model_device) for lbl in labels_list]
            gt_box_list = [gt.to(model_device) for gt in gt_box_list]
            regressions, classifications = model(image)
            Loss_loc, Loss_conf, no_pos = criterion(
                gt_box_list, labels_list, regressions, classifications
            )
            if no_pos:
                total_loss = torch.tensor(0.0, device=model_device, requires_grad=True)
            else:

                total_loss = Loss_conf + model_attributes.alpha * Loss_loc
                train_loss = train_loss + total_loss.item() * N_images
                train_samples = train_samples + N_images
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if train_samples > 0:
            train_loss = train_loss / train_samples
        else:
            train_loss = float("inf")

        # train_losses.append(train_loss)

        if epoch % epoch_verbose == 0:
            print(f"Epoch [{epoch:03d}] | Train Loss: {train_loss:.6f}")

        model.eval()
        eps = 10 ** (-2)
        val_loss = 0
        val_samples = 0

        model_attributes.phase = "test"
        with torch.no_grad():
            for image, labels_list, gt_box_list in val_dataloader:
                N_images = image.shape[0]
                image = image.to(model_device)
                labels_list = [lbl.to(model_device) for lbl in labels_list]
                gt_box_list = [gt.to(model_device) for gt in gt_box_list]

                regressions, classifications, output = model(image)
                Loss_loc, Loss_conf, no_pos = criterion(
                    gt_box_list, labels_list, regressions, classifications
                )
                if no_pos:
                    # no pisitives identified in a batch , skip this iteration
                    continue

                evaluation(output, labels_list, gt_box_list, metric)

                total_loss = Loss_conf + model_attributes.alpha * Loss_loc
                val_loss = val_loss + total_loss.item() * N_images
                val_samples = val_samples + N_images

        if val_samples > 0:
            val_loss = val_loss / val_samples
            map_score = metric.compute()["map"].item()
            metric.reset()
        else:
            val_loss = float("inf")
            map_score = 0

        # val_losses.append(val_loss)

        if best_val_loss - val_loss > eps:
            counter_early_stop = 0
            best_val_loss = val_loss
            results = (
                {
                    "model_state": model_attributes.state_dict(),
                    "hyperparameters": {
                        "alpha": model_attributes.alpha,
                        "N_epochs": model_attributes.N_epochs,
                    },
                    "val_loss": best_val_loss,
                    "optimizer": optimizer.state_dict(),
                    "map_score": map_score,
                    "epoch": epoch,
                }
            )
            if multigpu:

                if model_device.index == 0:

                    torch.save(
                        results,
                        f"{modelname}.pth",
                    )
            else:

                torch.save(
                    results,
                    f"{modelname}.pth"
                )

        else:
            counter_early_stop = counter_early_stop + 1

        if epoch % epoch_verbose == 0:
            print(
                f"Epoch [{epoch:03d}] | Val Loss: {val_loss:.6f} | Val MAP: {map_score:.6f}"
            )

        if counter_early_stop >= early_stop_iter:

            logwandb(
                epoch,
                multigpu,
                model_device,
                run,
                train_loss,
                val_loss,
                map_score,
                True,#early stop 
            )
            break

        else:
            logwandb(
                epoch,
                multigpu,
                model_device,
                run,
                train_loss,
                val_loss,
                map_score,
                False,#early stop 
            )
    if run is not None:
        run.finish()


def load_model(link, device, model):
    loaded_state = torch.load(link, map_location=device)
    model.load_state_dict(loaded_state["model_state"])
    model.to(device)
    model.eval()
    return model


def predict(model, image):
    with torch.no_grad():
        output = model(image)
    return output
