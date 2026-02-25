
import torch
import torch.nn as nn
from multiloss import MultiLoss

from ssd import SSD 



def train(model):

    criterion=MultiLoss(anchors=model.anchors)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)


    N_epochs=model.N_epochs


    for epoch in range(N_epochs):

        model.train()


        train_loss=0
        train_samples=0
        for data in train_loader:

            image,label=data
            image=image.to(device)
            label=label.to(device)



            regressions,classifications=model(image)
            
            Loss_loc,Loss_conf,no_pos=criterion(start_index=0,end_index=2,gt_list,labels_list,regressions,classifications)

            if no_pos:
                # no pisitives identified in a batch , skip this iteration 
                continue
            total_loss=Loss_conf+model.alpha*Loss_loc



            loss=loss_fn(y_pred,label)
            train_loss=train_loss=loss.item()*image.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss=train_loss/train_samples
        train_losses.append(train_loss)

        if epoch%5==0:
            print("loss is",train_loss,epoch)
        model.eval()

        val_loss=0
        val_samples=0
        with torch.no_grad():
            for data in val_loader:
                image,label=data
                image=image.to(device)
                label=label.to(device)

                y_pred=model(image)
                loss=loss_fn(y_pred,label)
                val_loss=val_loss+loss.item()*image.shape[0]
        val_loss=val_loss/val_samples
        val_losses.append(val_loss)

        if epoch%5==0:
            print("val loss is",val_loss,epoch)

        if valmetric>maxmetric:
            maxmetric=valmetric
            torch.save({
                "model_state" : model.state_dict(),
                "hyperparameters": {
                    "alpha" : model.alpha
                }
            }, "bestmodel.pth")
               

def load_model(link,device,Model):
    loaded_state=torch.load(link,map_location=device)
    hyperparameters=loaded_state["hyperparameters"]
    model=Model(**hyperparameters)
    model.load_state_dict(loaded_state["hyperparameters"])
    model.to(device)
    model.eval()
    return model

def predict(model,image):
    with torch.no_grad():
        output=model(image)
    return output


