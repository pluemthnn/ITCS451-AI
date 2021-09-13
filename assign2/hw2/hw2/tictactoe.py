"""This module contains classes and functions for Tic-Tac-Toe.

Members:
1. Name:  ID:
2. Name:  ID:
3. Name:  ID:
4. Name: nongpluemnaruk  ID: 

"""
from __future__ import annotations
from os import stat
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union
from numpy.core import numeric

from tabulate import tabulate
import numpy as np
import math

symbol_map = ['_', 'X', 'O']


class Player(Enum):
    X = 1
    O = 2


@dataclass(frozen=True)
class TicTacToeState:
    # TODO 1: Add state information that you need.
    board: np.ndarray
    # The board position is numbered as follow:
    # 0 | 1 | 2
    # ----------
    # 3 | 4 | 5
    # ----------
    # 6 | 7 | 8
    #
    # If you need anything more than `board`, please provide them here.
    curPlayer: Player  # keep track of the current player

    def find_pos(pos) -> Tuple[int, int]:
        ''' Return x, y position that match input value '''
        x = None
        y = None
        pos_board = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ])
        for i in range(len(pos_board)):
            for j in range(len(pos_board[i])):
                if pos_board[i][j] == pos:
                    x = i
                    y = j
                    return x, y
            # end loop
        # end loop
        return x, y

    def isGoal(state: TicTacToeState) -> Union[int, None]:
        """ check if any player win or tie
            return player that win ( 1 -> X, 2 -> O, 3 -> Tie) or None """

        pattern = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6],
                   [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]  # All possibility that player can win
        Is_full = 0

        for i in range(len(pattern)):
            isAgent = []

            x, y = TicTacToeState.find_pos(pattern[i][0])

            for j in range(len(pattern[i])):
                # isAgent collect value inside board of that position
                x, y = TicTacToeState.find_pos(pattern[i][j])
                isAgent.append(state.board[x][y])
            # end loop

            Check = isAgent[0] + isAgent[1] + isAgent[2]
            if ((isAgent[0] == isAgent[1] == isAgent[2]) and (Check == 3 or Check == 6)):
                # if all value in isAgent is equal and check if [1,1,1] [2,2,2]
                # print("Agent", isAgent)
                return isAgent[0]
            isAgent.clear()
        # end loop

        for i in range(len(state.board)):
            for j in range(len(state.board[i])):
                # Count of all board is full with X or O
                if state.board[i][j] == 1 or state.board[i][j] == 2:
                    Is_full += 1

        if(Is_full == 9):
            ''' if Tie '''
            return 3

        return None

    def bot_util(state: TicTacToeState, player: Player) -> Union[int, None]:
        """ check if any player win or tie and
            return a player's score by calculation ... ( 1 -> X, 2 -> O, 3 -> Tie) or None """

        pattern = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6],
                   [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]  # All possibility that player can win

        status = 0  # win(1) lose(-1) tie(0)
        Is_full = 0  # check tie

        valid_act = TicTacToeState.actions(state)  # find lefted valid action
        valid_actions = len(valid_act) + 1

        for i in range(len(pattern)):
            isAgent = []

            x, y = TicTacToeState.find_pos(pattern[i][0])

            for j in range(len(pattern[i])):
                # isAgent collect value inside board of that position
                x, y = TicTacToeState.find_pos(pattern[i][j])
                isAgent.append(state.board[x][y])
            # end loop

            Check = isAgent[0] + isAgent[1] + isAgent[2]
            if ((isAgent[0] == isAgent[1] == isAgent[2]) and (Check == 3 or Check == 6)):
                # if all value in isAgent is equal and check if [1,1,1] [2,2,2]
                # print(f"Agent : {isAgent} player : {player.value}")
                if isAgent[0] == player.value:
                    status = 1
                else:
                    status = -1
                return status*valid_actions
            isAgent.clear()
        # end loop

        # check is tie
        for i in range(len(state.board)):
            for j in range(len(state.board[i])):
                # Count of all board is full with X or O
                if state.board[i][j] == 1 or state.board[i][j] == 2:
                    Is_full += 1

        if(Is_full == 9):
            status = 0
            return status*valid_actions

        return None

    # TODO 2: Create actions function

    @classmethod
    def actions(cls, state: TicTacToeState) -> List[int]:
        """Return a list of valid position (from 0 to 8) for the current player.

        In Tic-Tac-Toe, a player can always make a move as long as there is
        an empty spot. If the board is full, however, return an empty list.
        """
        pos_board = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ])
        Act_List = []
        Count = 0

        for i in range(len(state.board)):
            for j in range(len(state.board[i])):
                if state.board[i][j] == 0:
                    Act_List.append(pos_board[i][j])
                if state.board[i][j] == 1 or state.board[i][j] == 2:
                    Count += 1
            # end loop
        # end loop
            if Count == 9:
                return []
            # end condition
        return Act_List

    # TODO 3: Create a transtion function
    @classmethod
    def transition(cls, state: TicTacToeState, action: Union[int, None]) -> TicTacToeState:
        """Return a new state after a player plays `action`.

        If `action` is None, skip the player turn.

        The current player is in the `state.curPlayer`. If the action is not
        valid, skip the player turn.

        Note
        --------------------------
        Keep in mind that you should not modify the previous state
        If you need to clone a numpy's array, you can do so like this:
        >>> y = np.array(x)
        >>> # make change
        >>> y.flags.writeable = False
        This will create an array y as a copy of array x and then make
        array y immutable (cannot be changed, for safty).
        """
        New_state = np.array(state.board)
        x, y = TicTacToeState.find_pos(action)

        New_state[x][y] = state.curPlayer.value
        New_state.flags.writeable = False

        if state.curPlayer.value == 2:
            return TicTacToeState(New_state, Player.X)
        # Skip the player turn
        return TicTacToeState(New_state, Player.O)

    # TODO 4: Create a terminal test function

    @classmethod
    def isTerminal(cls, state: TicTacToeState) -> bool:
        """Return `True` is the `state` is terminal (end of the game)."""
        is_terminal = TicTacToeState.isGoal(state)

        if is_terminal != None:
            if is_terminal < 4 and is_terminal > -1:
                return True

        # end loop
        return False

    # TODO 5: Create a utility function
    @classmethod
    def utility(cls, state: TicTacToeState, player: Player) -> Union[float, None]:
        """Return the utility of `player` for the `state`.

        If the state is non-terminal, return None.

        The `player` can be different than the `state`.`curPlayer`.
        """

        # Normal
        is_terminal = TicTacToeState.isGoal(state)
        if player.value == is_terminal:
            return 1.0
        return 0.0

        # For AI
        # return TicTacToeState.bot_util(state, player)

    def __repr__(self) -> str:
        a = [[symbol_map[c] for c in row] for row in self.board]
        if TicTacToeState.isTerminal(self):
            return tabulate(a)
        else:
            return tabulate(a) + '\n' + 'Turn: ' + self.curPlayer.name


class StupidBot:

    def __init__(self, player: Player) -> None:
        self.player = player

    def play(self, state: TicTacToeState) -> Union[int, None]:
        """Return an action to play or None to skip."""
        # pretend to be thinking
        time.sleep(1)

        # return random action
        valid_actions = TicTacToeState.actions(state)
        if len(valid_actions) == 0:
            return None
        else:
            return valid_actions[np.random.randint(0, len(valid_actions))]


class HumanPlayer(StupidBot):

    def __init__(self, player: Player) -> None:
        super().__init__(player)

    def play(self, state: TicTacToeState) -> Union[int, None]:
        valid_actions = TicTacToeState.actions(state)
        if len(valid_actions) == 0:
            return None
        else:
            action = int(input(f'Valid: {valid_actions} Your move: '))
            while action not in valid_actions:
                print('Your move is invalid. Try again:')
                action = int(input(f'Valid: {valid_actions} Your move: '))
            return action


class MinimaxBot(StupidBot):

    count = 0

    def __init__(self, player: Player) -> None:
        super().__init__(player)

    # TODO 6: Implement Minimax Decision algorithm
    def play(self, state: TicTacToeState) -> Union[int, None]:
        """Return an action to play or None to skip."""
        player = state.curPlayer
        valid_actions = TicTacToeState.actions(state)
        if len(valid_actions) == 10:
            return valid_actions[np.random.randint(0, len(valid_actions))]
        else:
            best_score = -(math.inf)
            best_move = 0

            for i in valid_actions:
                x, y = TicTacToeState.find_pos(i)
                if state.board[x][y] == 0:
                    tic_state = TicTacToeState.transition(state, i)
                    score = MinimaxBot.minimax(tic_state, False, player)
                    state.board[x][y] == 0
                    if (score > best_score):
                        best_score = score
                        best_move = i
                print(f"Minima {MinimaxBot.count}")
            MinimaxBot.count = 0
            return best_move

    def minimax(state: TicTacToeState, isMaximizing, player: Player):
        valid_actions = TicTacToeState.actions(state)
        # print(f"Is Go--{TicTacToeState.isGoal(state)} {valid_actions}")
        if TicTacToeState.isGoal(state) == 1:
            score = TicTacToeState.bot_util(state, player)  # TODO Important !
            # print(f"in 1 score {score}")
            MinimaxBot.count += 1
            return score
        elif TicTacToeState.isGoal(state) == 2:
            score = TicTacToeState.bot_util(state, player)  # TODO Important !
            # print(f"in -1 score {score}")
            MinimaxBot.count += 1
            return score
        elif TicTacToeState.isGoal(state) == 3:
            score = TicTacToeState.bot_util(state, player)  # TODO Important !
            # print(f"in tie score {score}")
            MinimaxBot.count += 1
            return score

        if (isMaximizing):
            best_score = -(math.inf)
            for i in valid_actions:
                x, y = TicTacToeState.find_pos(i)
                if state.board[x][y] == 0:
                    tic_state = TicTacToeState.transition(state, i)
                    # print(f" try {i} at {x}{y} isMaximizing {isMaximizing}")
                    score = MinimaxBot.minimax(tic_state, False, player)
                    state.board[x][y] == 0
                    if (score > best_score):
                        best_score = score
            return best_score
        else:
            best_score = math.inf
            for i in valid_actions:
                x, y = TicTacToeState.find_pos(i)
                if state.board[x][y] == 0:
                    tic_state = TicTacToeState.transition(state, i)
                    # print(f" try {i} at {x}{y} isMaximizing {isMaximizing}")
                    score = MinimaxBot.minimax(tic_state, True, player)
                    state.board[x][y] == 0
                    if (score < best_score):
                        best_score = score
            return best_score


class AlphaBetaBot(StupidBot):

    count = 0

    def __init__(self, player: Player) -> None:
        super().__init__(player)

    # TODO 7: Implement Alpha-Beta Decision algorithm
    def play(self, state: TicTacToeState) -> Union[int, None]:
        """Return an action to play or None to skip."""
        player = state.curPlayer
        valid_actions = TicTacToeState.actions(state)
        if len(valid_actions) == 10:
            return valid_actions[np.random.randint(0, len(valid_actions))]
        else:
            best_score = -(math.inf)
            best_move = 0

            for i in valid_actions:
                x, y = TicTacToeState.find_pos(i)
                if state.board[x][y] == 0:
                    tic_state = TicTacToeState.transition(state, i)
                    score = AlphaBetaBot.AlphaBeta(
                        tic_state, -(math.inf), math.inf, False, player)
                    state.board[x][y] == 0
                    if (score > best_score):
                        best_score = score
                        best_move = i
                print(f"Apl {AlphaBetaBot.count}")
            AlphaBetaBot.count = 0
            return best_move

    def AlphaBeta(state: TicTacToeState, alpha, beta, isMaximizing, player: Player):
        valid_actions = TicTacToeState.actions(state)
        # print(f"Is Go--{TicTacToeState.isGoal(state)} {valid_actions}")
        if TicTacToeState.isGoal(state) == 1:
            score = TicTacToeState.bot_util(state, player)  # TODO Important !
            AlphaBetaBot.count += 1
            # print(f"in 1 score {score}")
            return score
        elif TicTacToeState.isGoal(state) == 2:
            score = TicTacToeState.bot_util(state, player)  # TODO Important !
            AlphaBetaBot.count += 1
            # print(f"in -1 score {score}")
            return score
        elif TicTacToeState.isGoal(state) == 3:
            score = TicTacToeState.bot_util(state, player)  # TODO Important !
            AlphaBetaBot.count += 1
            # print(f"in tie score {score}")
            return score

        if (isMaximizing):
            best_score = -(math.inf)
            for i in valid_actions:
                x, y = TicTacToeState.find_pos(i)
                if state.board[x][y] == 0:
                    tic_state = TicTacToeState.transition(state, i)
                    # print(f" try {i} at {x}{y} isMaximizing {isMaximizing}")
                    score = AlphaBetaBot.AlphaBeta(
                        tic_state, alpha, beta, False, player)
                    state.board[x][y] == 0
                    if (score > best_score):
                        best_score = score
                    if (best_score >= beta):
                        return best_score
                    alpha = max(alpha, best_score)
            return best_score
        else:
            best_score = math.inf
            for i in valid_actions:
                x, y = TicTacToeState.find_pos(i)
                if state.board[x][y] == 0:
                    tic_state = TicTacToeState.transition(state, i)
                    # print(f" try {i} at {x}{y} isMaximizing {isMaximizing}")
                    score = AlphaBetaBot.AlphaBeta(
                        tic_state, alpha, beta, True, player)
                    state.board[x][y] == 0
                    if (score < best_score):
                        best_score = score
                    if (best_score <= alpha):
                        return best_score
                    beta = min(beta, best_score)
            return best_score
