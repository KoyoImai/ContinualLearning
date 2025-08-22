
import torch
import torch.nn as nn



def evaluation_cclis(model, criterion, dataloader, args):

    # model を evalモードに変更
    model.eval()


    correct = 0
    total_loss = 0
    total = 0 # number of samples
    num_batch = len(dataloader)

    if torch.cuda.is_available():
        model.cuda()
    
    with torch.no_grad():

        for idx, (images, labels, importance_weight, index) in enumerate(dataloader):

            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]


            total += bsz


            w = model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.prototypes.weight.copy_(w)

            features, output = model(images)


            target_labels = list(range(10))

            # ISSupCon
            loss = criterion(output,
                            features, 
                            labels, 
                            importance_weight, 
                            index, 
                            target_labels=target_labels, 
                            sample_num=None, 
                            score_mask=None,
                            reduction='mean',
                            )

            total_loss += loss.item() * bsz

    
    return total_loss/total, None








def evaluation_co2l():



    assert False
