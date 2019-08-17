import numpy as np
from quoridor import Quoridor
from policy_value_net import PolicyValueNet
from mcts_player import MCTSPlayer

class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board):
        board.print_board()

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (-1, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board()
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=1, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        print('=================================')
        print(self.board._p1_loc)
        print(self.board._p2_loc)
        print('=================================')
        states, mcts_probs, current_players = [], [], []
        max_step = 30
        step = 0
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.get_current_player())
            # perform a move
            self.board.take_action(move)
            if is_shown:
                self.graphic(self.board)
            end, winner = self.board.check_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    print("Game end. Winner is player:", winner)
                    return winner, zip(states, mcts_probs, winners_z)
            step += 1
            if step == max_step:
                if is_shown:
                    print("Game end. Tie")
                return winner, zip(states, mcts_probs, np.zeros(len(current_players)))



    def start_human_ai_play(self):
        model_file = 'best_policy.model'
        best_policy = PolicyValueNet(model_file=model_file)
        cts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)

        while True:
            self.graphic(self.board)
            action = int(input('Your action:'))
            self.board.take_action(action)
            self.graphic(self.board)
            print(self.board.valid_actions())
            move = cts_player.get_action(self.board)
            self.board.take_action(move)
            self.graphic(self.board)
            print(self.board.valid_actions())
            end, winner = self.board.check_end()
            if end:
                print("Game end. Winner is", winner)
                break
        


if __name__ == '__main__':
    q = Quoridor()
    g = Game(q)
    g.start_human_ai_play()


def test():
    from quoridor import Quoridor
    from pure_mcts import MCTSPlayer as MCTS_Pure
    from mcts_player import MCTSPlayer
    from policy_value_net import PolicyValueNet
    policy_value_net = PolicyValueNet(model_file=None)
    c_puct = 5
    n_playout = 800
    temp = 1.0
    board = Quoridor()
    game = Game(board)
    mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                      c_puct=c_puct,
                                      n_playout=n_playout,
                                      is_selfplay=1)
    winner, play_data = game.start_self_play(mcts_player,
                                            is_shown=1,
                                            temp=temp)
    print(winner)
    print(play_data)

    state_batch = [data[0] for data in play_data]
    mcts_probs_batch = [data[1] for data in play_data]
    winner_batch = [data[2] for data in play_data]

    learn_rate = 2e-3
    lr_multiplier = 1.0
    kl_targ = 0.02

    old_probs, old_v = policy_value_net.policy_value(state_batch)
    for i in range(5):
        loss, entropy = policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    learn_rate*lr_multiplier)
        new_probs, new_v = policy_value_net.policy_value(state_batch)
        kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
        )
        if kl > kl_targ * 4:  # early stopping if D_KL diverges badly
            break
    # adaptively adjust the learning rate
    if kl > kl_targ * 2 and lr_multiplier > 0.1:
        lr_multiplier /= 1.5
    elif kl < kl_targ / 2 and lr_multiplier < 10:
        lr_multiplier *= 1.5

    explained_var_old = (1 -
                            np.var(np.array(winner_batch) - old_v.flatten()) /
                            np.var(np.array(winner_batch)))
    explained_var_new = (1 -
                            np.var(np.array(winner_batch) - new_v.flatten()) /
                            np.var(np.array(winner_batch)))
    print(("kl:{:.5f},"
            "lr_multiplier:{:.3f},"
            "loss:{},"
            "entropy:{},"
            "explained_var_old:{:.3f},"
            "explained_var_new:{:.3f}"
            ).format(kl,
                    lr_multiplier,
                    loss,
                    entropy,
                    explained_var_old,
                    explained_var_new))
    policy_value_net.save_model('./current_policy.model')
