import torch
import torchvision
from runx import SumX

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

model = torchvision.models.resnet18()
inp = torch.randn(1, 3, 224, 224)
sumx = SumX()
sumx.summarize(model, inp)
