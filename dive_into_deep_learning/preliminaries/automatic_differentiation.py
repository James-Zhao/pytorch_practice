import torch

# 2.5.1 例子
x = torch.arange(4.0)
print(x)

x.requires_grad_(True)
print(x.grad)

y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)
print(x.grad == 4 * x)

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 2.5.2 非标量变量的反向传播
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。 # 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x))) y.sum().backward()
print(x.grad)

# 2.5.3 分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z= u * x
z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)

# 2.5.4 Python控制流的梯度计算
def f(a):
    b=a*2
    while b.norm() < 1000:
        b=b*2
    if b.sum() > 0:
        c=b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)