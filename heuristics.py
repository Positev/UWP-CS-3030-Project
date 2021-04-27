from operator import add
def clean_board(board):
    board = board.copy()
    board.pop()
    board = list(reversed(list(board)))
    return board

def transpose(board):
    transposed = [[] for _ in range(len(board[0]))]
    for row in board:
        for index, val in enumerate(row):
            transposed[index].append(val)
    return transposed

#good returns(col, height)
def get_column_height(board):
    heights = { index: 0 for index in range(len(board[0]))}
    for height, row in enumerate(board):
        for col, val in enumerate(row):
            if val != 0 and heights[col] < height:
                heights[col] = height

    return list(heights.items())

def aggregate_height(heights):
    return sum([height for index,height in heights])

def categorize_hole(board, hole):
    val = 0
    def in_board(h):

        if not (0 <= h[1] < len(board)):
            return False

        if not (0 <= h[0] < len(board[0])):
            return False

        return True
    
    def check_neighbor(hole, delta):
    
        loc = tuple(map(add, hole, delta))

        if in_board(loc):
            if board[loc[1]][loc[0]] != 0:
                return 1
        
        return 0
    
    val += check_neighbor( hole, (1,0))
    val +=  check_neighbor( hole, (-1,0))
    val += check_neighbor( hole, (0,1))
    val += check_neighbor( hole, (0,-1))

    if val > 1:
        return val

    return 0
    

def hole_count(board):
    pre_boar = board.copy()
    board = clean_board(board)
    holes = 0

    for row_index in range(len(board)):
        row = board[row_index]
        for col_index in range(len(row)):
            val = row[col_index]

            if val == 0:
                holes += categorize_hole(board, (col_index, row_index))
                

    return holes

def columns_with_atleast_one_hole(board):
    board = clean_board(board)
    pass

def bumpiness(heights):
    heights = heights.copy()
    heights.pop()
    last_height = heights[0][1]
    diffs = []
    for index, height in heights:
        diffs.append(abs(height - last_height))
        last_height = height
    return sum(diffs)

def row_transitions(board):
    board = clean_board(board)
    pass

def column_transitions(board):
    board = clean_board(board)
    pass

def pit_count(heights):
    return len([height for index, height in heights if height == 0])
    

def compute_midgame_fitness(board):
    board = clean_board(board)
    transposed_board = transpose(board)
    heights = get_column_height(board)

    max_height = max([height for index, height in heights])

    agheight = aggregate_height(heights)
    bumps = bumpiness(heights)
    pits = pit_count(heights)

    holes = hole_count(board)
    print(f"Bumpiness: {bumps}, Holes: {holes}, Pits: {pits}")
    return bumps * 2 + pits * 6+ max_height + holes * 3

def compute_endgame_fitness(board):
    board = clean_board(board)
    transposed_board = transpose(board)
    heights = get_column_height(board)

    max_height = max([height for index, height in heights])

    agheight = aggregate_height(heights)
    bumps = bumpiness(heights)
    pits = pit_count(heights)
    holes = hole_count(board)

    return agheight * 2 + bumps * 3 + pits * 8 + max_height + holes *1