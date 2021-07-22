import numpy as np
import random

X = 1
O = -1

BOARD_SIZE = 3


class MCTS:
    def __init__(self, num_iter=10) -> None:
        self.tic_tac_toe = self._set_tic_tac_toe()
        self.simulate_iter = num_iter

    def _set_tic_tac_toe(self):
        starting_state = np.zeros((BOARD_SIZE, BOARD_SIZE))
        return Node(starting_state)
    
    def simulate(self):
        for i in range(self.simulate_iter):
            print(f"Simulating iter {i+1}")
            self.tic_tac_toe.simulate(self.random_policy, X)

    def random_policy(self, actions):
        print("actions")
        print(actions)
        return random.choice(actions)


class Node:
    def __init__(self, state, parent=None) -> None:
        self.state = state
        self.parent = parent
        self.children = []
        self.value = None
        self.visit_count = 0

    
    def to_string(self):
        string = ("Current state:"
        f"{self.state}"
        f"Visit count: {self.visit_count}"        
        f"Value: {self.value}"
        )
        return string

    def __str__(self) -> str:
        return self.to_string()
    
    def __repr__(self):
        return self.__str__

    def add_children(self, child):
        self.children.append(child)


    def get_winner(self) -> int:
         # if draw
        if np.all(self.state != 0):
            return 0

        # taken from: https://github.com/int8/monte-carlo-tree-search/blob/58771d1e61c5b0024c23c2c7a4cdb88ffe2efd0a/mctspy/tree/nodes.py#L57
        # refactor this
        rowsum = np.sum(self.state, 0)
        colsum = np.sum(self.state, 1)
        diag_sum_tl = self.state.trace()
        diag_sum_tr = self.state[::-1].trace()

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
        print("curr state indexes", list(np.ndindex(self.state.shape)))
        return[(i, j) for i, j in np.ndindex(self.state.shape) if self.state[i, j] == 0]


    def backup(self, reward):
        # back propagate rewards to parent nodes
        self.visit_count += 1
        if self.value:
            self.value = self.value / (self.visit_count + 1) + reward / self.visit_count
        else:
            self.value = reward
        
        print("state value", self.value)
        if self.parent:
            self.parent.backup(reward)


    def get_next_player(self, curr_player):
        return -curr_player


    def rollout(self):
        pass

    def expand(self):
        # expand all states
        # TODO still doesn't solve the problem of same afterstate from different prev states
        if not self.children:
            possible_moves = self.get_possible_moves()
            for move in possible_moves:
                new_state = self.state.copy()
                possible_moves = self.get_possible_moves()


    def simulate(self, policy, player):
        print("Current state: ")
        print(self.state)
        if not self.is_terminal():
            print("Getting new moves")
            possible_moves = self.get_possible_moves()

            print(possible_moves)

            next_move = policy(possible_moves)
            new_state = self.state.copy()
            new_state[next_move] = player

            # if new_state not in self.children:
            #     # write logic to check
            #     child = Node(new_state, self)
            #     self.add_children(child)
            # else:
            #     # get child
            #     child = Node(new_state, self)

            child = Node(new_state, self)

            next_player = self.get_next_player(player)
            child.simulate(policy, next_player)
        
        else:
            # terminal state

            winner = self.get_winner()
            print("the winner is")
            print(winner)
            reward = winner
            self.backup(reward)

        # TODO figure out same afterstate reached from different parents


# step 1: write averaging returns with random rollout policy


# write some tests


def selection():
    pass

def expansion():
    pass

def simulate():
    pass

def simulation_policy():
    # can have a tree policy followed by a rollout policy
    pass


def evaluate_action():
    # could be as simple as averaging returns
    pass


if __name__ == "__main__":
    mcts = MCTS()
    mcts.simulate()
