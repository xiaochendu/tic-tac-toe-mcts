import matplotlib.pyplot as plt
import numpy as np
import random

X = 1
O = -1

BOARD_SIZE = 3


class MCTS:
    def __init__(self, num_sim=10) -> None:
        self.tic_tac_toe = self._set_tic_tac_toe()
        self.num_simulations = num_sim

    def _set_tic_tac_toe(self):
        starting_state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        return Node(starting_state)
    
    def simulate(self):
        for i in range(self.num_simulations):
            print(f"Simulating iter {i+1}")
            self.tic_tac_toe.simulate(X)
    
    def run(self):
        player = X
        move_num = 1
        max_moves = 10

        fig = plt.figure()
        ax = fig.add_subplot(1, max_moves, move_num)

        while not self.tic_tac_toe.is_terminal():
            self.simulate()
            # get best move
            best_child = self.tic_tac_toe.greedy_policy(player)
            print("best child")
            print(best_child)
            # doesn't display before exiting
            ax.imshow(best_child.state)
            self.tic_tac_toe = best_child
            player = Node.get_next_player(player)

            move_num += 1
        
        plt.show()
        plt.close()

class Node:
    def __init__(self, state, parent=None) -> None:
        self.state = state
        self.parent = parent
        self.children = []
        self.value = 0.0
        self.rewards = []
        self.visit_count = 0

    
    def to_string(self):
        string = ("Current state:\n"
        f"{self.state}\n"
        f"Visit count: {self.visit_count}\n"        
        f"Value: {self.value}\n"
        )
        return string

    def __str__(self) -> str:
        return self.to_string()
    
    def __repr__(self):
        return self.__str__()

    def add_children(self, child):
        self.children.append(child)

    def get_winner(self, curr_state=None) -> int:
        if not curr_state:
            curr_state = self.state

         # if draw
        if np.all(curr_state != 0):
            return 0

        # TODO: use own method to get winner and run tests

        # taken from: https://github.com/int8/monte-carlo-tree-search/blob/58771d1e61c5b0024c23c2c7a4cdb88ffe2efd0a/mctspy/tree/nodes.py#L57
        # refactor this
        rowsum = np.sum(curr_state, 0)
        colsum = np.sum(curr_state, 1)
        diag_sum_tl = curr_state.trace()
        diag_sum_tr = curr_state[::-1].trace()

        player_one_wins = any(rowsum == BOARD_SIZE)
        player_one_wins += any(colsum == BOARD_SIZE)
        player_one_wins += (diag_sum_tl == BOARD_SIZE)
        player_one_wins += (diag_sum_tr == BOARD_SIZE)

        if player_one_wins:
            return X

        player_two_wins = any(rowsum == -BOARD_SIZE)
        player_two_wins += any(colsum == -BOARD_SIZE)
        player_two_wins += (diag_sum_tl == -BOARD_SIZE)
        player_two_wins += (diag_sum_tr == -BOARD_SIZE)

        if player_two_wins:
            return O

        return None


    def is_terminal(self) -> bool:
       return self.get_winner() is not None


    def get_possible_moves(self):
        # print("curr state indexes", list(np.ndindex(self.state.shape)))
        return[(i, j) for i, j in np.ndindex(self.state.shape) if self.state[i, j] == 0]


    def backup(self, reward):
        # back propagate rewards to parent nodes
        self.visit_count += 1
        self.rewards.append(reward)
        # print("Rewards", self.rewards)
        # print("Rewards sum", sum(self.rewards))
        # print("Visits", self.visit_count)
        if self.value > 0.0:
            # TODO: set learning rate
            # self.value = (self.value * (self.visit_count - 1) + reward) / self.visit_count
            self.value = sum(self.rewards) / self.visit_count
        else:
            self.value = reward
        
        # print("state value", self.value)
        if self.parent:
            self.parent.backup(reward)

    @classmethod
    def get_next_player(cls, curr_player):
        return -curr_player


    def rollout(self, player):
        state_backup = self.state.copy()
        while not self.is_terminal():
            possible_moves = self.get_possible_moves()
            next_move = self.rollout_policy(possible_moves)
            self.state[next_move] = player

        winner = self.get_winner()
        # print("the winner is")
        # print(winner)
        reward = float(winner)

        self.state = state_backup
        self.backup(reward)

    def expand(self, player):
        # expand all states
        # TODO still doesn't solve the problem of same afterstate from different prev states
        possible_moves = self.get_possible_moves()
        for next_move in possible_moves:
            new_state = self.state.copy()
            new_state[next_move] = player
            new_child = Node(new_state, self)
            assert new_child.parent is self, "Child's parent is not current Node!"
            self.children.append(new_child)
        
        # print("Created children", self.children)

    def rollout_policy(self, actions):
        return random.choice(actions)

    def select_child(self, curr_player):
        if np.random.binomial(1, 0.05):
            # print("using random policy")
            # use random policy
            return np.random.choice(self.children)
        else:
            # print("greedy policy")
            return self.greedy_policy(curr_player)


    def greedy_policy(self, curr_player):
        child_vals = [x.value*curr_player for x in self.children]
        best_child_idx = np.random.choice([index for index, val in enumerate(child_vals) if val == np.max(child_vals)])
        return self.children[best_child_idx]


    def simulate(self, player):
        if not self.is_terminal():                
            next_player = self.get_next_player(player)

            if self.children:
                # for child in self.children:
                child = self.select_child(player)
                child.simulate(next_player)
            else:
                self.expand(player)
                child = self.select_child(player)
                child.rollout(next_player)

        else:
            # terminal state
            winner = self.get_winner()
            # print("the winner is")
            # print(winner)
            reward = float(winner)
            self.backup(reward)


        # TODO printing out some graphics of the generated tree and maps

# step 1: write averaging returns with random rollout policy

# write some tests


def selection():
    pass

def expansion():
    pass

def simulation_policy():
    # can have a tree policy followed by a rollout policy
    pass


def evaluate_action():
    # could be as simple as averaging returns
    pass


if __name__ == "__main__":
    mcts = MCTS(100)
    mcts.run()
