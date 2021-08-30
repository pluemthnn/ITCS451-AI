"""This module contains classes and function to solve a pathfinding problem.

Author:
    -
    -
Student ID:
    -
    -
"""

# %%

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Callable, Union
from hw1.envutil import find_agent, render_maze

import numpy as np


@dataclass(frozen=True, unsafe_hash=False)
class MazeState:

    # TODO 1: Add other state information here.
    grid: np.ndarray
    # If you need anything more than `grid`, please add here

    # TODO 2 Create a list of all possible actions.
    # Please replace it with your own actions
    # Note that an agent can only rotate and move forward.
    actions: Tuple[str] = ('right', 'move', 'down', 'left', 'up' )

    def __eq__(self, o: object) -> bool:
        if isinstance(o, MazeState):
            return np.all(self.grid == o.grid)
        return False

    def __hash__(self) -> int:
        return render_maze(self.grid).__hash__()

    # TODO 3: Create a transition function
    @classmethod
    def transition(cls, state: MazeState, action: str) -> MazeState:
        """Return a new state after performing `action`.
        If the action is not possible, it should return None.
        The mud disappears as soon as the agent walk onto it.
        Note
        ---------------------
        Keep in mind that you should not modify the previous state
        If you need to clone a numpy's array, you can do so like this:
        >>> y = np.array(x)
        >>> y.flags.writeable = False
        This will create an array y as a copy of array x and then make
        array y immutable (cannot be changed, for safty).
        """
        new_grid = np.array(state.grid)
        # print(new_grid)
        y, x = find_agent(state.grid)  # transform agent position to [x, y]
        agent_pos = state.grid[x, y]
        # print(state)

        if(action == "up" and state.grid[x-1, y] != 1 and agent_pos != 2):
            new_grid[x, y] = 2
        elif(action == "right" and state.grid[x, y+1] != 1 and agent_pos != 3):
            new_grid[x, y] = 3
        elif(action == "down" and state.grid[x+1, y] != 1 and agent_pos != 4):
            new_grid[x, y] = 4
        elif(action == "left" and state.grid[x, y-1] != 1 and agent_pos != 5):
            new_grid[x, y] = 5
        elif(action == "move"):
            if (agent_pos == 2 and state.grid[x-1, y] != 1):  # If agent face up
                new_grid[x, y] = 0
                new_grid[x-1, y] = 2
            # If agent face right
            elif (agent_pos == 3 and state.grid[x, y+1] != 1):
                new_grid[x, y] = 0
                new_grid[x, y+1] = 3
            # If agent face down
            elif (agent_pos == 4 and state.grid[x+1, y] != 1):
                new_grid[x, y] = 0
                new_grid[x+1, y] = 4
            # If agent face left
            elif (agent_pos == 5 and state.grid[x, y-1] != 1):
                new_grid[x, y] = 0
                new_grid[x, y-1] = 5

        # print(new_grid)
        new_grid.flags.writeable = False
        new_state = MazeState(new_grid)
        return new_state

    # TODO 4: Create a cost function
    @classmethod
    def cost(cls, state: MazeState, action: str) -> float: # ERROR WTF
        """Return the cost of `action` for a given `state`.

        If the action is not possible, the cost should be infinite.

        Note
        ------------------
        You may come up with your own cost for each action, but keep in mind
        that the cost must be positive and any walking into
        a mod position should cost more than walking into an empty position.
        """
        y, x = find_agent(state.grid)  # transform agent position to [x, y]
        cost = 0  # initial cost value if passable
        agent_pos = state.grid[x, y]
        next_pos = 0

        
        if (agent_pos != 2 and action == "up"):  # If agent face up -> check wall, mud
            next_pos = state.grid[x-1, y]
            cost += 1

        elif (agent_pos != 3 and action == "right"):  # If agent face right -> check wall, mud
            next_pos = state.grid[x, y+1]
            cost += 1
            
        elif (agent_pos != 4 and action == "down"):  # If agent face down -> check wall, mud
            next_pos = state.grid[x+1, y]
            cost += 1
    
        elif (agent_pos != 5 and action == "left"):  # If agent face left -> check wall, mud
            next_pos = state.grid[x, y-1]
            cost += 1

        elif (action == "move"):
            cost += 1
            if(agent_pos == 2): next_pos = state.grid[x-1, y]
            elif(agent_pos == 3): next_pos = state.grid[x, y+1]
            elif(agent_pos == 4): next_pos = state.grid[x+1, y]
            elif(agent_pos == 5): next_pos = state.grid[x, y-1]

            if (next_pos == 7): 
                cost += 1

        if (next_pos == 1):
            cost = float('inf')

        return cost

    # TODO 5: Create a goal test function
    @classmethod
    def is_goal(cls, state: MazeState) -> bool:
        """Return True if `state` is the goal."""
        i, j = state.grid.shape  # check grid shape
        if(state.grid[i-2, j-2] < 6 and state.grid[i-2, j-2] > 1):  # check if goal position have agent
            return True
        return False

    # TODO 6: Create a heuristic function
    @classmethod
    def heuristic(cls, state: MazeState) -> float:
        """Return a heuristic value for the state.
        Note
        ---------------
        You may come up with your own heuristic function.
        """
        y, x = find_agent(state.grid)  # transform agent position to [x, y]
        i, j = state.grid.shape  # get grid shape
        heu = abs(x-(i-2)) + abs(y-(j-2))
        return heu
# %%


@dataclass
class TreeNode:
    path_cost: float
    state: MazeState
    action: str
    depth: int
    parent: TreeNode = None


def dfs_priority(node: TreeNode) -> float:
    return -1.0 * node.depth


def bfs_priority(node: TreeNode) -> float:
    return 1.0 * node.depth


# TODO: 7 Create a priority function for the greedy search
def greedy_priority(node: TreeNode) -> float:
    return 0.0


# TODO: 8 Create a priority function for the A* search
def a_star_priority(node: TreeNode) -> float:
    return 0.0


# TODO: 9 Implement the graph search algorithm.
def graph_search(
        init_state: MazeState,
        priority_func: Callable[[TreeNode], float]) -> Tuple[List[str], float]:
    """Perform graph search on the initial state and return a list of actions.

    If the solution cannot be found, return None and infinite cost.
    """
    return None, float('inf')
