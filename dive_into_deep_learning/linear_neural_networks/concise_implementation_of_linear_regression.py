import torch
import lr_util

# 3.3.1 生成数据集
import numpy as np
from torch.utils import data

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = lr_util.synthetic_data(true_w, true_b, 1000)

# 3.3.2 读取数据集
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 3.3.3 定义模型
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))

# 3.3.4 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 3.3.5 定义损失函数
loss = nn.MSELoss()

# 3.3.6 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 3.3.7 训练
# 在每个迭代周期里，我们将完整遍历一次数据集(train_data)，不停地从中获取一个小批量的输 入和相应的标签。对于每一个小批量，我们会进行以下步骤:
# • 通过调用net(X)生成预测并计算损失l(前向传播)。
# • 通过进行反向传播来计算梯度。
# • 通过调用优化器来更新模型参数。

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差:', true_b - b)
