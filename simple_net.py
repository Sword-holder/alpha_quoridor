import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np

SIZE = 7
# 输入的状态空间大小
STATE_SIZE = 2 * SIZE * SIZE + 2 * (SIZE - 1) * (SIZE - 1)
WALLS_SIZE = (SIZE - 1) * (SIZE - 1)
ACTION_SIZE = 16 + 2 * WALLS_SIZE

class SimplePolicyNet(nn.Module):
    '''
        简单的策略网络设计，用了两个线性回归
    '''
    def __init__(self): 
        super(simple_policy_net, self).__init__()
        self.hidden_size = 128
        self.linear1 = nn.Linear(STATE_SIZE, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, ACTION_SIZE)
        self._option()
  
    def forward(self, x): 
        out = self.linear1(x)
        out = self.linear2(out)
        return out
    
    def _option(self):
        self.criterion = nn.MSELoss(size_average = False)
        self.optimizer = optim.SGD(model.parameters(), lr = 0.01)
        self.training_step = 10

    def _train_step(self, batch_x, batch_y):
        # 创建模型
        model = SimplePolicyNet()

        for epoch in range(self.training_step):
            # 正向传播获得输出
            pred_y = model(batch_x)
            # 交叉熵函数 
            loss = self.criterion(pred_y, batch_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss.data[0]))

        test_x = Variable(torch.Tensor([[4.0]]))
        pred_y = model(test_x)
        print("predict (after training)", 4, model(test_x).data[0][0])

    def train(self):
        batch_x = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
        batch_y = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
        self._train_step(batch_x, batch_y)

    def save_model(self, model_file):
        """ 保存模型"""
        torch.save(self.policy_value_net.state_dict(), 'ckpt/%s.pth'%(model_file))