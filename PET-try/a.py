import torch
if __name__ == '__main__':
    model_pth = r'resume/SHA_model.pth'
    net = torch.load(model_pth, map_location=torch.device('cpu'))
    for key, value in net["state_dict"].items():
        print(key,value.size(),sep="  ")