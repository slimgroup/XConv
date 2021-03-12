from pyxconv import *
from pyxconv.modules import BReLU

r1 = torch.nn.ReLU()
r2 = BReLU()

xt = torch.randn(1, 1, 20, 20, requires_grad=True)
xp = (xt + 0).requires_grad_()

y1 = r1(xt)
y2 = r2(xp)

print(y1 - y2)
print(y1.grad_fn, y2.grad_fn)

out = torch.randn(1, 1, 20, 20)

g1 = y1.grad_fn(out)
g2 = torch.mul(out, y2.grad_fn.saved_tensors[0])

print(g1 - g2)
