import collections
import matplotlib.pyplot as plt
import numpy as np
import random


X = 1
O = -1

BOARD_SIZE = 3


class MCTS:
    def __init__(self, num_sim=10) -> None:
        self.nodes = collections.defaultdict(object)
        self.num_simulations = num_sim
        self.tic_tac_toe = self._set_tic_tac_toe()

    def _set_tic_tac_toe(self):
        starting_state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        starting_node = Node(starting_state, mcts=self)
        # add starting node to all nodes dict
        node_hash = self.get_hash(starting_node)
        self.nodes[node_hash] = starting_node

        # print("hash of numpy array")
        # print(self.get_hash(starting_node.state))

        # print("hash of node")
        # print(self.get_hash(starting_node))

        print(self.nodes)
        return starting_node
    
    def simulate(self, player):
        for i in range(self.num_simulations):
            # print(f"Simulating iter {i+1}")
            self.tic_tac_toe.simulate(player)
    
    @classmethod
    def get_hash(self, node):
        if isinstance(node, Node):
            data = node.state
        elif isinstance(node, np.ndarray):
            data = node
        return "".join([str(x) for x in np.ravel(data)])
        
    def run(self):
        player = X
        move_num = 1
        max_moves = 10

        # fig = plt.figure()
        # ax = fig.add_subplot(1, max_moves, move_num)

        while not self.tic_tac_toe.is_terminal() and move_num < max_moves:
            self.simulate(player)
            # get best move
            best_child = self.tic_tac_toe.greedy_policy(player)
            print("best child")
            print(best_child)

            for child in self.tic_tac_toe.children:
                assert np.abs(np.sum(child.state)) <= 1.0, "number of X and O differ by more than 1"
                # print(child)

            # doesn't display before exiting
            # ax.imshow(best_child.state)
            self.tic_tac_toe = best_child
            player = Node.get_next_player(player)
            print("next player", player)

            move_num += 1

class Node:
    def __init__(self, state, parent=None, mcts=None, exploration_constant=5) -> None:
        self.state = state
        self.parents = []
        if parent:
            self.add_parent(parent)
        self.current_parent = None # parent that chose this (child) node during selection phase
        self.children = []
        self.value = None
        self.rewards = []
        self.visit_count = 0
        self.exploration_constant = exploration_constant
        self.MCTS = mcts

    def to_string(self):
        string = ("Current state:\n"
        f"{self.state}\n"
        f"Visit count: {self.visit_count}\n"
        # f"Rewards: {self.rewards}\n"
        f"Value: {self.value}\n"
        )
        return string

    def __str__(self) -> str:
        return self.to_string()
    
    def __repr__(self):
        return self.__str__()

    def update_curr_parent(self, parent):
        self.current_parent = parent

    def add_parent(self, parent):
        self.parents.append(parent)

    def add_children(self, child):
        self.children.append(child)

    def get_winner(self, curr_state=None) -> int:
        if not curr_state:
            curr_state = self.state

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

         # if draw
        if np.all(curr_state != 0):
            return 0
        
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
        if self.value is not None:
            # TODO: set learning rate
            # self.value = (self.value * (self.visit_count - 1) + reward) / self.visit_count
            self.value = sum(self.rewards) / self.visit_count
        else:
            self.value = reward
        
        # print("state value", self.value)
        # if self.parents:
        #     for parent in self.parents:
        #         parent.backup(reward)
        if self.current_parent:
            self.current_parent.backup(reward)
            self.current_parent = None

    @classmethod
    def get_next_player(cls, curr_player):
        return -curr_player

    def rollout(self, player):
        state_backup = self.state.copy()
        while not self.is_terminal():
            # self.state = state_backup.copy()
            possible_moves = self.get_possible_moves()
            next_move = self.rollout_policy(possible_moves)
            assert self.state[next_move] == 0, "next move is already taken"
            self.state[next_move] = player

        winner = self.get_winner()
        # print("the winner is")
        # print(winner)
        reward = float(winner)

        self.state = state_backup.copy()
        self.backup(reward)

    def get_symmetric_states(self, state):
        pass

    def expand(self, player):
        # expand all states
        possible_moves = self.get_possible_moves()
        # print("current node")
        # print(self)

        # print("player", player)
        for next_move in possible_moves:
            new_state = self.state.copy()
            assert new_state[next_move] == 0, "next move is already taken"
            new_state[next_move] = player

            # find symmetrical states
            symmetric_states = self.get_symmetric_states(new_state)

            # solve afterstates
            # check if node already contained in MCTS
            mcts = self.MCTS
            hash_code = mcts.get_hash(new_state)
            # print("afterstate is", hash_code)



            if hash_code in mcts.nodes:
                # print("afterstate already created, getting existing node and adding to parents list")
                new_child = mcts.nodes[hash_code]
                new_child.add_parent(self)
                # TODO append parent node
            else:
                # print("creating new afterstate and adding to mcts")
                new_child = Node(new_state, parent=self, mcts=self.MCTS)
                mcts.nodes[hash_code] = new_child
            
            assert self in new_child.parents, "Current node not in child's parents!"

            # print(new_child)
            assert np.abs(np.sum(new_child.state)) <= 1.0, "number of X and O differ by more than 1"

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
        
    def select_child_ucb(self, curr_player):
        max_ucb_value = -5
        best_child_index = None
        # ensure there's at least 1 visit
        total_visits = max(1, np.sum([child.visit_count if child.visit_count else 1e-5 for child in self.children]))
        for index, child in enumerate(self.children):
            num_visits = child.visit_count if child.visit_count else 1e-5
            value = child.value if child.value else 0
            ucb_value = value*curr_player + child.exploration_constant * np.sqrt(np.log(total_visits) / num_visits)
            # import pdb; pdb.set_trace()
            # print("ucb value", ucb_value)
            if ucb_value > max_ucb_value:
                max_ucb_value = ucb_value
                best_child_index = index
        
        selected_child = self.children[best_child_index]
        selected_child.update_curr_parent(self)
        return selected_child

    def greedy_policy(self, curr_player):
        child_vals = [x.value*curr_player if x.value else 0 for x in self.children]
        best_child_idx = np.random.choice([index for index, val in enumerate(child_vals) if val == np.max(child_vals)])
        return self.children[best_child_idx]

    def simulate(self, player):
        if not self.is_terminal():                
            next_player = self.get_next_player(player)

            if self.children:
                # for child in self.children:
                #     print(child)
                # child = self.select_child(player)
                child = self.select_child_ucb(player)
                child.simulate(next_player)
            else:
                self.expand(player)
                # child = self.select_child(player)
                child = self.select_child_ucb(player)
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

if __name__ == "__main__":
    mcts = MCTS(100)
    mcts.run()
