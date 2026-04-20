import torch 



def evaluation(test_output,labels_list,gt_box_list,metric):
    """
    input : 
        test_output -> output from ssd forward in test mode, shape [nb images, top-k-generated-boxes, 6]
        among 6 there are 4 coords of boxes, class probability column and class column itself (from 1 to N) O (background) is excluded

        gt_box_list  : list[tensor] coords of gt boxes corresponding to each image i of test_output
        labels_list : list(tensor)

    output : 

        MAP metric 

    """
    # metric = MeanAveragePrecision()
    #do over all images in the batch of eval dataloder
    preds=[]
    targets=[]
    for i in range(test_output.shape[0]):
        #remember: since for each image we generate top k bboxes , if geenrated bboxes<200, then the rest is 0, so need just to filter them out 
        actual_preds=test_output[i][~(test_output[i] == 0).all(dim=1)]#filter out where alll row values are zero 
        preds.append({
            "boxes" : actual_preds[:,:4],
            "scores" : actual_preds[:,4],
            "labels" : actual_preds[:,5].to(torch.int64)-1#map fct requires 0 indexing
        })
        targets.append({
            "boxes" : gt_box_list[i],
            "labels" : labels_list[i]-1
        })

    metric.update(preds, targets)

def load_model(link, device, model,optimizer):
    loaded_state = torch.load(link, map_location=device)
    model.load_state_dict(loaded_state["model_state"])
    model.to(device)
    old_lr=optimizer.param_groups[0]["lr"]
    optimizer.load_state_dict(loaded_state["optimizer"])
    wandbid=loaded_state.get("wandb_run_id",None)
    #we update the lr as i did schedule lr and it did not work well 
    for param_group in optimizer.param_groups:
            param_group["lr"] = old_lr

    return model , loaded_state["epoch"], optimizer,loaded_state.get("max_map",None),wandbid


def predict(model, image):
    model.eval()
    model.phase = "test"
    with torch.no_grad():
        output = model(image)
    return output

 