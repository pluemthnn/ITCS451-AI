# ITCS451-AI
## This Repository is a part of ITCS451 Assignment @muict
 
### Background
We are going to create a pathfinding agent -- an agent that finds a path with the least cost. The environment is a grid world like this (10x10):

```
# # # # # # # # # #
# >   ~       ~   #
#   #       # # ~ #
#   # # #       ~ #
#   # # # # # ~   #
#   ~ ~ ~ # # # ~ #
#     #   # # #   #
#   ~ #   # # # ~ #
#       ~       G #
# # # # # # # # # #
```

In this example, the agent ('>') is at the top-left corner and facing east. You can find the meaning of all symbols here:

    Symbol mapping:
    -  0: ' ', empty (passable)
    -  1: '#', wall (not passable)
    -  2: '^', agent is facing up (north)
    -  3: '>', agent is facing right (east)
    -  4: 'v', agent is facing down (south)
    -  5: '^', agent is facing left (west)
    -  6: 'G', goal
    -  7: '~', mud (passable, but cost more)
For each step, the agent can change its facing direction or move forward (no action is not an option here). The agent has to pay for each action. You can design the cost of the actions. Whenever the agent gets into the mud, it must pay an additional cost and the mud will disappear.

Checkpoint 1 [x]
Implement TODO 1 to 6

  1. 1pt: Your code run (checkpoint1.py) without any error.
  2. 0.5pt: For each correct implementation of the TODO 1 to 6

Checkpoint 2 [ ]tbc.
Implement TODO 7 to 9

  1. 1pt: Your code run (checkpoint2.py) without any error.
  2. 0.5pt: TODO 7
  3. 0.5pt: TODO 8
  4. 1pt: TODO 9, termination condition is correct
  5. 1pt: TODO 9, new nodes are created correctly
  6. 1pt: TODO 9, nodes are correctly pushed into the queue (open-set)
  7. 1pt: TODO 9, a correct plan and a correct cost is returned
