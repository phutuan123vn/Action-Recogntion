from Model.model import ModelSTGCN
import torch

def main():
    model = ModelSTGCN(in_channels=3, num_class=5)
    x = torch.rand(5, 3, 50, 17)
    model.to('cuda')
    model.eval()
    out = model(x.to('cuda'))
    print(out.shape)

if __name__=="__main__":
    main()
    