import copy
import random
import numpy as np


class MinimaxNode:
    def __init__(self, state):
        self.state = state
        self.move = None
        self.score = None
        self.score_avg = None
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0


class MinimaxTree:
    def __init__(self, game):
        self.game = game
        self.root = MinimaxNode(game.get_state())

    def build_tree(self, node, depth, maximizing_player):
        game_current = copy.deepcopy(self.game)
        game_current.set_state(node.state)

        if game_current.is_terminal() or depth == 0:
            node.score = game_current.evaluate()
            node.score_avg = node.score
            return

        legal_moves = game_current.get_legal_moves()
        best_score = float('-inf') if maximizing_player else float('inf')
        for move in legal_moves:
            game_current.apply_move(move)
            child_node = MinimaxNode(game_current.get_state().copy())
            child_node.move = move
            node.children.append(child_node)
            self.build_tree(child_node, depth - 1, not maximizing_player)
            game_current.delete_move(move)

            if maximizing_player:
                if child_node.score > best_score:
                    best_score = child_node.score
            else:
                if child_node.score < best_score:
                    best_score = child_node.score

        node.score = best_score
        node.score_avg = round(np.array([child.score for child in node.children]).mean(), 3)

    def get_best_move(self, node: MinimaxNode, is_maximizing_player):
        node = self.root if node is None else node

        if node.children == '':
            return -1

        if is_maximizing_player:
            best_score = max(child.score for child in node.children)
        else:
            best_score = min(child.score for child in node.children)

        good_moves = []
        for i, child in enumerate(node.children):
            if best_score == child.score:
                good_moves.append([child.move, child.score, child.score_avg])

        print(good_moves)
        for child in node.children:
            print(child.move, child.score, child.score_avg, end=', ')
        print()

        return random.choice(good_moves)

    def print_tree(self, node=None, depth=1):
        if node is None:
            print(f"{self.root.state}: {self.root.move}, {self.root.score}")
        node = self.root if node is None else node

        for child in node.children:
            print(f"{"--" * depth}{child.state}: {child.move}, {child.score}")
            self.print_tree(child, depth + 1)
