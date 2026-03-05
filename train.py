import torch
import torch.nn as nn
from multiloss import MultiLoss
from eval import evaluation
from ssd import SSD
from torchmetrics.detection import MeanAveragePrecision

def train(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    modelname: str,
   
    epoch_verbose: int = 5,
    early_stop_iter=10,
    start_epoch=0
):
    model_device=model.device
    metric = MeanAveragePrecision().to(model_device)
    multigpu=False
    model_attributes=model
    if model.device!=torch.device("cpu"):
        if torch.cuda.device_count()>0:
                device_id=f"{model.device.type}:{model.device.index}"
                multigpu=True
                model=DDP(model,device_ids=[device_id])
                model_attributes=model.module



    
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    criterion = MultiLoss(anchors=model_attributes.anchors)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)
    N_epochs = model_attributes.N_epochs

    counter_early_stop = 0

    for epoch in range(start_epoch,N_epochs):
        model_attributes.phase="train"
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
                # no pisitives identified in a batch , skip this iteration
                continue
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

        train_losses.append(train_loss)

        if epoch % epoch_verbose == 0:
            print(f"Epoch [{epoch:03d}] | Train Loss: {train_loss:.6f}")

        model.eval()
        eps = 10 ** (-2)
        val_loss = 0
        val_samples = 0

        model_attributes.phase="test"
        with torch.no_grad():
            for image, labels_list, gt_box_list in val_dataloader:
                N_images = image.shape[0]
                image = image.to(model_device)
                labels_list = [lbl.to(model_device) for lbl in labels_list]
                gt_box_list = [gt.to(model_device) for gt in gt_box_list]

                regressions, classifications,output = model(image)
                Loss_loc, Loss_conf, no_pos = criterion(
                    gt_box_list, labels_list, regressions, classifications
                )
                if no_pos:
                    # no pisitives identified in a batch , skip this iteration
                    continue

                evaluation(output,labels_list,gt_box_list,metric)
                

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

        val_losses.append(val_loss)

        if best_val_loss - val_loss > eps:
            counter_early_stop = 0
            best_val_loss = val_loss
            if multigpu:

                if model.device.index==0:

                    torch.save(
                        {
                            "model_state": model_attributes.state_dict(),
                            "hyperparameters": {
                                "alpha": model_attributes.state_dict().alpha,
                                "N_epochs": model_attributes.state_dict().N_epochs
                            },
                            "val_loss": best_val_loss,
                            "optimizer": optimizer.state_dict(),
                            "map_score" : map_score,
                            "epoch" : epoch
                            
                        },
                        f"{modelname}.pth",
                    )
            else:

                torch.save(
                        {
                            "model_state": model_attributes.state_dict(),
                            "hyperparameters": {
                                "alpha": model_attributes.state_dict().alpha,
                                "N_epochs": model_attributes.state_dict().N_epochs
                            },
                            "val_loss": best_val_loss,
                            "optimizer": optimizer.state_dict(),
                            "map_score" : map_score,
                            "epoch" : epoch
                            
                        },
                        f"{modelname}.pth",
                    )


        else:
            counter_early_stop = counter_early_stop + 1

        if epoch % epoch_verbose == 0:
            print(f"Epoch [{epoch:03d}] | Val Loss: {val_loss:.6f} | Val MAP: {map_score:.6f}")

        if counter_early_stop >= early_stop_iter:
            return {
                "train_losses" : train_losses,
                "val_losses" : val_losses,
                "map_score" : map_score,
                "early_stop" : True
            }

    return {
                "train_losses" : train_losses,
                "val_losses" : val_losses,
                "map_score" : map_score,
                "early_stop" : False
            }



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
