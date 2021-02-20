
from r import *
from newvgg import *
from C3D_model import C3D

m = Net()
print(m)
print(sum(p.numel() for p in m.parameters() if p.requires_grad))
