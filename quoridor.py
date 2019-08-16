from queue import Queue
import numpy as np
import time

def record_time(func):
    def dec(*args, **kw):
        start = time.time()
        ret = func(*args, **kw)
        end = time.time()
        print('call %s():' % func.__name__, end=' ')
        print('spend time:', end - start, 's')
        return ret
    return dec

class Quoridor(object):
    '''
        为提高效率，这个类中对动作进行了编码
        0~3表示分别表示朝上、右、下、左方向移动
        4~15分别表示跳跃动作，其中
            4、5、6表示上上、上右、上左
            7、8、9表示右上、右右、右下
            10、11、12表示下右、下下、下左
            13、14、15表示左上、左下、左左
        其后的(SIZE - 1)*(SIZE - 1)位置表示放置横向挡板
        再后面的(SIZE - 1)*(SIZE - 1)位置表示放置纵向挡板
            即：
            16 ~ 16+(SIZE - 1)*(SIZE - 1) - 1 表示在对应位置放置横向挡板
            16+(SIZE - 1)*(SIZE - 1) ~ 2 * (16+(SIZE - 1)*(SIZE - 1)) - 1 表示在对应位置放纵向挡板
    '''

    # 棋盘宽度
    SIZE = 7
    # 棋盘中的格子数
    BOARD_SIZE = SIZE * SIZE
    WALLS_SIZE = (SIZE - 1) * (SIZE - 1)
    # 动作空间大小
    ACTION_SIZE = 16 + 2 * WALLS_SIZE
    # move_step用于记录各个方向移动的距离，列表的索引即为方向，值为移动距离     
    MOVE_STEP = [SIZE, 1, -SIZE, -1]
    # 完整状态下，棋盘的大小
    STATE_BOARD_SIZE = 2 * SIZE - 1

    # 初始挡板数量
    WALL_NUMBER = 8

    def __init__(self):
        # _self_loc和_oppo_loc表示棋子的位置，均为二维棋盘的线性展开
        # 初始状态下，我方位于第一排正中间，对方位于最后一排正中间
        self._p1_loc = self.SIZE // 2
        self._p2_loc = self.SIZE * (self.SIZE - 1) + self.SIZE // 2
        # walls数组表示所有的挡板空间
        self._walls = np.zeros(self.WALLS_SIZE, dtype=np.int8)
        # player为1表示player1，-1表示player2，一开始player默认为player1
        self.player = 1
        # 每个玩家的挡板数量
        self.wall_remaining = {1:self.WALL_NUMBER, -1:self.WALL_NUMBER}

    def get_current_player(self):
        '''
            获取当前玩家
        '''
        return self.player

    def check_end(self):
        '''
            检查游戏是否结束，若结束，返回player编号，否则返回None
        '''
        if self._p1_loc // self.SIZE == self.SIZE - 1:
            return 1, 
        if self._p2_loc // self.SIZE == 0:
            return -1
        return None

    def valid_actions(self):
        '''
            获取所有的合法动作，返回一个定长的合法动作数组，索引为动作编号，值为1时，动作有效，否则动作无效
            每个动作实际上是一个编号
            为了提高效率，返回的是numpy数组
        '''
        valids = self._valid_moves()
        valids.extend(self._valid_jumps())
        valids.extend(self._valid_walls())
        return valids

    def take_action(self, action):
        '''
            执行一个动作，输入：action，动作编码
            不返回任何值
        '''
        if 0 <= action <= 3:
            self._move(action)
        elif 4 <= action <= 15:
            d1 = (action - 4) // 3
            d2 = (action - 4) % 3
            jump_step = [self.SIZE, 1, -self.SIZE, -1]
            jump_step.pop((d1 + 2) % 4)
            d2 = self.MOVE_STEP.index(jump_step[d2])
            self._jump(d1, d2)
        elif 16 <= action <= 16 + self.WALLS_SIZE - 1:
            self._place_wall(action - 16, d=1)
        elif 16 + self.WALLS_SIZE <= action <= 16 + 2 * self.WALLS_SIZE:
            self._place_wall(action - 16 - self.WALLS_SIZE, d=2)
        else:
            print('Error: invalid action!')

    def _move(self, forward):
        '''
            移动棋子，player为1时表示我方移动，为-1时表示对手移动
            forward取值为0、1、2、3，分别表示上、右、下、左
        '''
        loc = self._current_loc()
        loc += self.MOVE_STEP[forward]
        self._set_current_loc(loc)

    def _jump(self, d1, d2):
        '''
            跳跃动作，d1和d2是连续跳跃的两个方向
        '''
        self._move(d1)
        self._move(d2)

    def _place_wall(self, loc, d):
        '''
            用于在loc位置放置一个挡板，d为1时表示横向，为2时表示纵向
        '''
        self._walls[loc] = d
        self.wall_remaining[self.get_current_player()] -= 1

    def _set_current_loc(self, loc):
        # 更新当前位置
        if self.player == 1:
            self._p1_loc = loc
        else:
            self._p2_loc = loc

    def _current_loc(self):
        '''
            获得当前玩家的位置
        '''
        return self._p1_loc if self.player == 1 else self._p2_loc

    def _oppo_loc(self):
        '''
            获得对方玩家的位置
        '''
        return self._p1_loc if self.player == 2 else self._p2_loc

    def _player_loc(self):
        '''
            获得双方玩家的位置
            返回一个二元组，第一个元素是当前玩家位置，第二个元素是对方玩家的位置
        '''
        loc = [self._p1_loc, self._p2_loc]
        return loc[::self.player]

    def alter(self):
        '''
            轮换当前下棋的玩家
        '''
        self.player = -self.player

    def _valid_moves(self):
        '''
            获取所有合法移动列表
        '''
        valid_moves = []
        for i in range(4):
            if self._legal_mvoe(i):
                valid_moves.append(i)
        return valid_moves

    def _valid_jumps(self):
        '''
            获取所有合法跳跃列表
        '''
        jumps = []
        locs = self._player_loc()
        try:
            ind = self.MOVE_STEP.index(locs[1] - locs[0])
            # 如果自身与对手相邻，假装走到对手的位置，获取所有合法移动
            self._set_current_loc(locs[1])
            sim_moves = self._valid_moves()
            jump_step = [self.SIZE, 1, -self.SIZE, -1]
            jump_step.pop((ind + 2) % 4)
            for m in sim_moves:
                try:
                    j = jump_step.index(self.MOVE_STEP[m])
                    jumps.append(4 + 3 * ind + j)
                except ValueError:
                    continue
        except ValueError:
            pass
            
        return jumps
    
    def _valid_walls(self):
        '''
            获取所有合法放置挡板的动作列表
        '''
        # 先判断是否还有挡板可以放
        if self.wall_remaining[self.get_current_player()] == 0: return []
        valid_walls = []
        for i in range(self.WALLS_SIZE):
            if self._walls[i] == 0:
                if not self._will_block(i, d=1) and not self._wall_overlap(i, d=1):
                    valid_walls.append(16 + i)
                if not self._will_block(i, d=2) and not self._wall_overlap(i, d=2):
                    valid_walls.append(16 + self.WALLS_SIZE + i)
        return valid_walls

    def _wall_overlap(self, wall, d):
        '''
            判断放置的这个挡板会不会和周围挡板重叠
        '''
        row = wall // (self.SIZE - 1)
        col = wall % (self.SIZE - 1)
        # 如果为横向，则检查左右两个交叉点，否则检查上下两个交叉点
        check_dim = col if d == 1 else row
        step = 1 if d == 1 else self.SIZE - 1
        if check_dim != 0 and self._walls[wall - step] == d: return True
        if check_dim != self.SIZE - 2 and self._walls[wall + step] == d: return True
        return False

    def _will_block(self, wall, d):
        '''
            判断这个挡板是否会将棋子封锁
        '''
        self._walls[wall] = d
        blocked = self._blocked()
        self._walls[wall] = 0
        return blocked

    def _blocked(self):
        '''
            判断当前棋局是否堵塞了游戏中的两个棋子
        '''
        return self._block_place(self._p1_loc, dest=self.SIZE - 1) or self._block_place(self._p2_loc, dest=0)

    def _block_place(self, place, dest):
        '''
            判断当前棋局是否堵塞了棋子，棋子的位置由place给出，dest表示目标
            返回一个布尔值，表示是否堵塞
            算法使用广度优先搜索实现
        '''
        row = place // self.SIZE
        col = place % self.SIZE
        visited = np.zeros(shape=(self.SIZE, self.SIZE), dtype=np.int8)
        return not self._dfs(row, col, dest, visited)
    
    def _dfs(self, row, col, dest, visited):
        '''
            从row行col列开始进行深度优先搜索，搜到dest行后，返回True，否则返回False
        '''
        if row == dest:
            return True
        if row < 0 or row >= self.SIZE:
            return False
        if col < 0 or col >= self.SIZE:
            return False
        if visited[row][col] == 1:
            return False
        
        visited[row][col] = 1
        connected = False

        place = row * self.SIZE + col
        left = place - 1
        right = place + 1
        up = place + self.SIZE
        down = place - self.SIZE

        forward = col < dest
        priority = 1 if forward else -1

        if row != (0, self.SIZE - 1)[forward] and not self._has_wall_crossed(place, (down, up)[forward]):
            connected = connected or self._dfs(row + priority, col, dest, visited)
        if col != 0 and not self._has_wall_crossed(place, left):
            connected = connected or self._dfs(row, col - 1, dest, visited)
        if col != self.SIZE - 1 and not self._has_wall_crossed(place, right):
            connected = connected or self._dfs(row, col + 1, dest, visited)
        if row != (0, self.SIZE - 1)[1 - forward] and not self._has_wall_crossed(place, (down, up)[1 - forward]):
            connected = connected or self._dfs(row - priority, col, dest, visited)

        return connected

    def _legal_mvoe(self, d):
        '''
            判断当前玩家向d方向的移动是否合法
            d取值为0、1、2、3，分别表示上、右、下、左
            返回一个布尔值
        '''
        locs = self._player_loc()
        # 计算移动后的位置
        move_loc = locs[0] + self.MOVE_STEP[d]
        # 走到了框外，则非法
        if move_loc < 0 or move_loc >= self.BOARD_SIZE:
            return False
        # 走到了左右两边的格子外，则非法
        if locs[0] % self.SIZE == 0 and move_loc % self.SIZE == self.SIZE - 1:
            return False
        if locs[0] % self.SIZE == self.SIZE - 1 and move_loc % self.SIZE == 0:
            return False
        # 如果移动后会与对手的位置重合，则非法
        if move_loc == locs[1]:
            return False
        # 如果挡板挡住了移动的方向，则非法
        if self._has_wall_crossed(locs[0], move_loc):
            return False
        
        return True

    def _has_wall_crossed(self, pos1, pos2):
        '''
            判断两个相邻位置间是否有挡板
        '''
        # 保证pos1更小
        if pos1 > pos2:
            pos1, pos2 = pos2, pos1

        pos1_row = pos1 // self.SIZE
        pos1_col = pos1 % self.SIZE

        cross1 = pos1_row * (self.SIZE - 1) + pos1_col
        # 移动的位置在右边
        if pos2 - pos1 == 1:
            check_wall = 2
            cross2 = cross1 - (self.SIZE - 1)
            if pos1_row == self.SIZE - 1: cross1 = None
            if pos1_row == 0: cross2 = None
        # 移动的位置在上边
        if pos2 - pos1 == self.SIZE:
            check_wall = 1
            cross2 = cross1 - 1
            if pos1_col == self.SIZE - 1: cross1 = None
            if pos1_col == 0: cross2 = None

        # 检查两个交叉点是否把路挡住了
        if cross1 is not None and self._walls[cross1] == check_wall:
            return True
        if cross2 is not None and self._walls[cross2] == check_wall:
            return True

        return False

    def print_board(self):
        '''
            打印棋盘
        '''
        board = self._integerate_state()
        board = board[::-1]
        char_dic = ['.', '1', '2', '3', '4']
        for i in range(self.STATE_BOARD_SIZE):
            for j in range(self.STATE_BOARD_SIZE):
                print(char_dic[board[i][j]], end='')
            print()
        print('p1 walls:', self.wall_remaining[1], ', p2 walls:', self.wall_remaining[-1], end='. ')
        print('It\'s p', 1 if self.player == 1 else 2, '\'s turn')

    def current_state(self):
        '''
            返回适用于神经网络寻来你的当前状态
        '''
        # 首先生成单个的棋盘
        board = self._integerate_state()
        state = np.zeros(shape=(4, self.SIZE, self.SIZE), dtype=np.int8)
        state[0] = board
        for i in range(len(state)):
            for j in range(len(state[0])):
                state[board[i][j] - 1] = 1
        return state

    def _integerate_state(self):
        '''
            生成单个的棋盘，0表示没有东西，1表示竖向挡板，2表示横向挡板，3表示我方位置，4表示对方位置
        '''
        board = np.zeros(shape=(self.STATE_BOARD_SIZE, self.STATE_BOARD_SIZE), dtype=np.int8)

        board[self._p1_loc // self.SIZE * 2][(self._p1_loc % self.SIZE) * 2] = 3
        board[self._p2_loc // self.SIZE * 2][(self._p2_loc % self.SIZE) * 2] = 4
        for i, w in enumerate(self._walls):
            if w == 1:
                row = i // (self.SIZE - 1) * 2 + 1
                col = (i % (self.SIZE - 1)) * 2
                board[row][col] = 1
                board[row][col + 2] = 1
            elif w == 2:
                row = i // (self.SIZE - 1) * 2
                col = i % (self.SIZE - 1) * 2 + 1
                board[row][col] = 2
                board[row + 2][col] = 2

        return board



def start_game():
    q = Quoridor()
    while True:
        q.print_board()
        print(q.valid_actions())
        action = int(input('please input action code:'))
        q.take_action(action)
        if q.check_end():
            break
        q.alter()

def test():
    q = Quoridor()
    q.take_action(1)
    q.take_action(1)
    q.take_action(21)
    q.take_action(52)
    # q.take_action(58)
    # q.take_action(56)
    q.print_board()
    print(sorted(q.valid_actions()))

if __name__ == '__main__':
    test()
