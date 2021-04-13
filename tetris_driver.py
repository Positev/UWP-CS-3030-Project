# Tetris driver should be an interface between the tetris game and the tetris agent. 
# 

class TetrisDriver:
  def __init__(self, key_actions, get_board_callback, get_stone_callback):
    self.moves = []
    self.actions = key_actions
    self.get_board_callback = get_board_callback
    self.get_stone_callback = get_stone_callback
  

  #Will return a copy of the board. Current stone is not included on this board
  def get_board(self):
    board = self.get_board_callback()
    
    copy_board = [[val for val in row] for row in board]
    return copy_board
  
  #Will return current stone to be moved.
  def get_stone(self):
    stone, x, y = self.get_stone_callback()
    copy_stone = [[val for val in row] for row in stone]
    return copy_stone

  #Pick from {"LEFT", "RIGHT", "DOWN", "UP", 'RETURN'}
  def enqueue_action(self, action_label):

    if action_label in ["LEFT", "RIGHT", "DOWN", "UP", 'RETURN']:
      self.moves.insert( 0, action_label)

  def next_move_ready(self):
    print(self.get_board())
    print(self.get_stone())
    return len(self.moves) > 0
  def get_next_move(self):
    return self.moves.pop()