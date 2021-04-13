# Tetris agent should refer to the tetris driver, get the board, compute the best neighbor, and compute a path of actions to get there.
# the computed actions should be added to a queue that the tetris game will pull from at every frame.

class TetrisAgent:
  def __init__(self, driver):
    self.driver = driver
  def make_plan(self):
    pass