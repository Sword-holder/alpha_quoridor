import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

SIZE = 7
# 棋盘长度
BOARD_WIDTH = 2 * SIZE - 1
# 输入的状态空间大小
BOARD_SIZE = BOARD_WIDTH * BOARD_WIDTH
# 挡板动作空间大小
WALLS_SIZE = (SIZE - 1) * (SIZE - 1)
# 动作空间大小
ACTION_SIZE = 16 + 2 * WALLS_SIZE

class Net(nn.Module):
    '''
        策略价值网络的基本结构
        以SIZE=7的情况为例
        用了4个13*13的矩阵作为输入
        第一个矩阵表示我方位置，也就是说只有一个位置是1，其他全为0
        第二个矩阵表示对方位置，同样只有一个位置是1，其他全为0
        第三个矩阵表示横向挡板的放置情况
        第四个矩阵表示纵向挡板的放置情况
    '''
    def __init__(self): 
        super(Net, self).__init__()
        # 公共部分
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 策略网络部分
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * BOARD_SIZE, ACTION_SIZE)
        # 价值网络部分
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * BOARD_SIZE, 128)
        self.val_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        '''
            输入为17 * 17 * 4的状态空间
        '''
        # 公共部分
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 策略网络部分
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * BOARD_SIZE)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # 价值网络部分
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * BOARD_SIZE)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


class PolicyValueNet():
    '''
        完整的策略价值网络
    '''
    def __init__(self, model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.l2_const = 1e-4  # coef of l2 penalty
        # 定义策略价值网络对象
        if self.use_gpu:
            self.policy_value_net = Net().cuda()
        else:
            self.policy_value_net = Net()
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        '''
            输入：一组状态输入
            输出：一组（动作概率，状态价值）的输出
        '''
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        '''
            输入：棋盘本身
            输出：一个（动作编码，动作概率）列表和一个状态价值
        '''
        legal_positions = board.valid_actions()
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 4, BOARD_WIDTH, BOARD_WIDTH))

        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        '''
            执行一轮训练
        '''
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # 设置学习率
        self.set_learning_rate(lr)

        # 正向传播
        log_act_probs, value = self.policy_value_net(state_batch)
        # 定义loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # 注意: L2 penalty被加入到优化器
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # 反向传播更新参数
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        # for pytorch version < 0.5 please use the following line instead.
        # return loss.data[0], entropy.data[0]
        # for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        '''
            将模型保存到文件中
        '''
        net_params = self.get_policy_param()  # 获取模型参数
        torch.save(net_params, model_file)

    def set_learning_rate(self, lr):
        '''
            设置训练的学习率
        '''
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr