from quoridor import Quoridor
import copy
import sys, time, random
import numpy as np

def record_time(func):
    def dec(*args, **kw):
        start = time.time()
        ret = func(*args, **kw)
        end = time.time()
        print('call %s():' % func.__name__, end=' ')
        print('spend time:', end - start, 's')
        return ret
    return dec

INF = 100000000

def evaluate_fn(board):
    oppo_dis =  board.oppo_distance()
    self_dis = board.self_distance()
    # self_walls = board.wall_remaining[1]
    # oppo_walls = board.wall_remaining[-1]
    return 100 * (oppo_dis - self_dis) # + int(random.random() * 100)
    # return 0

def _minimax(board, depth, alpha, beta, max_layer):
    if depth == 0:
        return evaluate_fn(board), None
    end, _ = board.check_end()
    if end:
        return evaluate_fn(board), None

    if max_layer:
        max_eva = -INF
        valids = board.valid_actions()
        m_action = None
        for action in valids:
            board_copy = copy.deepcopy(board)
            board_copy.take_action(action)
            eva = _minimax(board_copy, depth-1, alpha, beta, False)[0]
            if eva > max_eva:
                max_eva = eva
                m_action = action
            alpha= max(alpha, max_eva)
            if beta <= alpha:
                break
        return max_eva, m_action
    else:
        min_eva = INF
        board.alter()
        valids = board.valid_actions()
        m_action = None
        for action in valids:
            board_copy = copy.deepcopy(board)
            board_copy.take_action(action)
            board_copy.alter()
            eva = _minimax(board_copy, depth-1, alpha, beta, True)[0]
            if eva < min_eva:
                min_eva = eva
                m_action = action  
            beta = min(beta, eva)
            if beta <= alpha:
                break
        return min_eva, m_action

def minimax(board, depth):
    '''
        输入一个Quoridor棋盘对象
        输出当前玩家的决策
    '''
    return _minimax(board, depth=depth, alpha=-INF, beta=INF, max_layer=True)[1]

def create_board(self_loc, oppo_loc, self_walls, oppo_walls, walls):
    q = Quoridor()
    q._self_loc = self_loc
    q._oppo_loc = oppo_loc
    q.wall_remaining[1] = self_walls
    q.wall_remaining[-1] = oppo_walls
    q._walls[:] = walls[:]
    return q

def main():
    board = create_board(3, 45, 8, 8, np.zeros(36))
    action = minimax(board, 2)
    print('action = ', action)
    board.print_board()
    board.take_action(action)
    board.print_board()
    print(action)

def interpret(action):
    if action[0] == 'h':
        row = int(action[1])
        col = int(action[2])
        return 16 + row * 6 + col
    elif action[0] == 'v':
        row = int(action[1])
        col = int(action[2])
        return 16 + 36 + row * 6 + col
    else:
        return int(action)

def game():
    counter = 0
    depth = 1
    q = Quoridor()
    while True:
        q.print_board()
        print(q.valid_actions())
        action = input('please input action code:')
        action = interpret(action)
        q.take_action(action)
        end, winner = q.check_end()
        if end:
            print('Game over, winner is ', winner)
            break

        q.alter()
        counter += 1
        if counter > 12:
            depth = 2
        action = minimax(q, depth)
        q.take_action(action)
        end, winner = q.check_end()
        if end:
            print('Game over, winner is ', winner)
            break
        q.alter()


if __name__ == '__main__':
    game()