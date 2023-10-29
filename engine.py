import torch
import torch.nn as nn

device = torch.device("cuda")

def test(model, dataloader=None):
    model.eval()
    cnt, ACC, correct_top5 = 0.0, 0, 0,0
    with torch.no_grad():
        for i,(img, iden) in enumerate(dataloader):
            img, iden = img.to(device), iden.to(device)

            bs = img.size(0)
            iden = iden.view(-1)
            _,out_prob = model(img)
            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()


            _, top5 = torch.topk(out_prob,5, dim = 1)  
            for ind,top5pred in enumerate(top5):
                if iden[ind] in top5pred:
                    correct_top5 += 1
        
            cnt += bs

    return ACC * 100.0 / cnt,correct_top5* 100.0 / cnt

def test_acc_loss(model, dataloader=None):
    
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss, cnt, ACC, correct_top5 = 0.0, 0, 0,0
    with torch.no_grad():
        for i,(img, iden) in enumerate(dataloader):
            img, iden = img.to(device), iden.to(device)

            bs = img.size(0)
            iden = iden.view(-1)
            _,out_prob = model(img)
            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            loss += criterion(out_prob,iden)

            _, top5 = torch.topk(out_prob,5, dim = 1)  
            for ind,top5pred in enumerate(top5):
                if iden[ind] in top5pred:
                    correct_top5 += 1
        
            cnt += bs

    return ACC * 100.0 / cnt,correct_top5* 100.0 / cnt, loss/len(dataloader)

