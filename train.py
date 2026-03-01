import torch
import torch.nn as nn
from multiloss import MultiLoss

from ssd import SSD


def train(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    modelname: str,
    device="cpu",
    epoch_verbose: int = 10,
    early_stop_iter=10,
):

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    criterion = MultiLoss(anchors=model.anchors)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)
    N_epochs = model.N_epochs

    counter_early_stop = 0

    for epoch in range(N_epochs):

        model.train()
        train_loss = 0
        train_samples = 0

        for image, labels_list, gt_box_list in train_dataloader:
            N_images = image.shape[0]
            image = image.to(device)
            labels_list = [lbl.to(device) for lbl in labels_list]
            gt_box_list = [gt.to(device) for gt in gt_box_list]
            regressions, classifications = model(image)
            Loss_loc, Loss_conf, no_pos = criterion(
                gt_box_list, labels_list, regressions, classifications
            )
            if no_pos:
                # no pisitives identified in a batch , skip this iteration
                continue
            total_loss = Loss_conf + model.alpha * Loss_loc
            train_loss = train_loss + total_loss.item() * N_images
            train_samples = train_samples + N_images
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if train_samples > 0:
            train_loss = train_loss / train_samples
        else:
            train_loss = float("inf")

        train_losses.append(train_loss)

        if epoch % epoch_verbose == 0:
            print(f"Epoch [{epoch:03d}] | Train Loss: {train_loss:.6f}")

        model.eval()
        eps = 10 ** (-2)
        val_loss = 0
        val_samples = 0

        with torch.no_grad():
            for image, labels_list, gt_box_list in val_dataloader:
                N_images = image.shape[0]
                image = image.to(device)
                labels_list = [lbl.to(device) for lbl in labels_list]
                gt_box_list = [gt.to(device) for gt in gt_box_list]

                regressions, classifications = model(image)
                Loss_loc, Loss_conf, no_pos = criterion(
                    gt_box_list, labels_list, regressions, classifications
                )
                if no_pos:
                    # no pisitives identified in a batch , skip this iteration
                    continue
                total_loss = Loss_conf + model.alpha * Loss_loc
                val_loss = val_loss + total_loss.item() * N_images
                val_samples = val_samples + N_images
                # TO DO : calcumate val metric

        if val_samples > 0:
            val_loss = val_loss / val_samples
        else:
            val_loss = float("inf")

        val_losses.append(val_loss)

        if best_val_loss - val_loss > eps:
            counter_early_stop = 0
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "hyperparameters": {
                        "alpha": model.alpha,
                        "N_epochs": model.N_epochs
                    },
                    "val_loss": best_val_loss,
                    "optimizer": optimizer.state_dict()
                       
                },
                f"{modelname}.pth",
            )

        else:
            counter_early_stop = counter_early_stop + 1

        if counter_early_stop >= early_stop_iter:
            break

        if epoch % epoch_verbose == 0:
            print(f"Epoch [{epoch:03d}] | Val Loss: {val_loss:.6f}")

    return "model trained "


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
