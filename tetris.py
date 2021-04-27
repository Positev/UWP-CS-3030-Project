#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# Very simple tetris implementation
#
# Control keys:
#       Down - Drop stone faster
# Left/Right - Move stone
#         Up - Rotate Stone clockwise
#     Escape - Quit game
#          P - Pause game
#     Return - Instant drop
#
# Have fun!

# NOTE: If you're looking for the old python2 version, see
#       <https://gist.github.com/silvasur/565419/45a3ded61b993d1dd195a8a8688e7dc196b08de8>

# Copyright (c) 2010 "Laria Carolin Chabowski"<me@laria.me>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

HEADLESS = True
from random import randrange as rand
if not HEADLESS:
    import pygame
import sys, neat, os, numpy,time
import pickle
from heuristics import *
OUTPUT_FILE_PATH = "winner.pkl"
INPUT_FILE_PATH = "winner.pkl"
# The configuration
cell_size = 18
cols =      10
rows =      22
maxfps =    30

colors = [
(0,   0,   0  ),
(255, 85,  85),
(100, 200, 115),
(120, 108, 245),
(255, 140, 50 ),
(50,  120, 52 ),
(146, 202, 73 ),
(150, 161, 218 ),
(35,  35,  35) # Helper color for background grid
]

# Define the shapes of the single parts


tetris_shapes = [

    [[1, 1, 1],
     [0, 1, 0]],

    [[0, 2, 2],
     [2, 2, 0]],

    [[3, 3, 0],
     [0, 3, 3]],

    [[4, 0, 0],
     [4, 4, 4]],

    [[0, 0, 5],
     [5, 5, 5]],
    [
        [6, 6, 6, 6]
    ],

    [[7, 7],
     [7, 7]]
]

def rotate_clockwise(shape):
    return [
        [ shape[y][x] for y in range(len(shape)) ]
        for x in range(len(shape[0]) - 1, -1, -1)
    ]

def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                board_cell = board[ cy + off_y ][ cx + off_x ]
                if cell and board_cell:
                    return True
            except IndexError:
                return True
    return False

def remove_row(board, row):
    del board[row]
    return [[0 for i in range(cols)]] + board

def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy+off_y-1 ][cx+off_x] += val
    return mat1

def new_board():
    board = [
        [ 0 for x in range(cols) ]
        for y in range(rows)
    ]
    board += [[ 1 for x in range(cols)]]
    return board

def find_longest_streak(row):
    cnt, max_val = 0, 0 # running count, and max count
    for e in row: 
        cnt = cnt + 1 if e != 0 else 0  # add to or reset running count
        max_val = max(cnt, max_val) # update max count
    return max_val

def calculate_fitness(score, board):
    fitness = score
    
    board.pop()
    rboard = list(reversed(board))
    for row in rboard:
        score += len([x for x in row if x != 0])
        streak = find_longest_streak(row)
        if streak > 6:
            score += streak ** 2
    return score


class TetrisApp(object):
    def __init__(self, genome):
        self.genome = genome
        if not HEADLESS:
            pygame.init()
            pygame.key.set_repeat(250,25)
        self.width = cell_size*(cols+6)
        self.height = cell_size*rows
        self.rlim = cell_size*cols
        self.bground_grid = [[ 8 if x%2==y%2 else 0 for x in range(cols)] for y in range(rows)]

        if not HEADLESS:
            self.default_font =  pygame.font.Font(
                pygame.font.get_default_font(), 12)

            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.event.set_blocked(pygame.MOUSEMOTION) # We do not need
                                                     # mouse movement
                                                     # events, so we
                                                     # block them.
        self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
        self.init_game()

    def new_stone(self, first = False):
        if not first:
            self.genome.fitness -= compute_midgame_fitness(self.board)
        self.stone = self.next_stone[:]
        self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
        self.stone_x = int(cols / 2 - len(self.stone[0])/2)
        self.stone_y = 0

        if check_collision(self.board,
                           self.stone,
                           (self.stone_x, self.stone_y)):
            self.gameover = True

    def init_game(self):
        self.board = new_board()
        self.new_stone(first=True)
        self.level = 1
        self.score = 0
        self.lines = 0
        if not HEADLESS:
            pygame.time.set_timer(pygame.USEREVENT+1, 100)

    def disp_msg(self, msg, topleft):
        x,y = topleft
        for line in msg.splitlines():
            self.screen.blit(
                self.default_font.render(
                    line,
                    False,
                    (255,255,255),
                    (0,0,0)),
                (x,y))
            y+=14

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image =  self.default_font.render(line, False,
                (255,255,255), (0,0,0))

            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(msg_image, (
              self.width // 2-msgim_center_x,
              self.height // 2-msgim_center_y+i*22))

    def draw_matrix(self, matrix, offset):
        off_x, off_y  = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        colors[val],
                        pygame.Rect(
                            (off_x+x) *
                              cell_size,
                            (off_y+y) *
                              cell_size,
                            cell_size,
                            cell_size),0)

    def add_cl_lines(self, n):
        linescores = [0,1000, 3000, 12000,24000]
        self.lines += n
        if n > 0:
            self.genome.fitness += 10000
        self.score += linescores[n] * self.level
        if self.lines >= self.level*6:
            self.level += 1
            newdelay = 1000-50*(self.level-1)
            newdelay = 100 if newdelay < 100 else newdelay
            
            if not HEADLESS:
                pygame.time.set_timer(pygame.USEREVENT+1, newdelay  / 5)

    def move(self, delta_x):
        if not self.gameover and not self.paused:
            new_x = self.stone_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > cols - len(self.stone[0]):
                new_x = cols - len(self.stone[0])
            if not check_collision(self.board,
                                   self.stone,
                                   (new_x, self.stone_y)):
                self.stone_x = new_x
    def quit(self):
        self.center_msg("Exiting...")
        if not HEADLESS:
            pygame.display.update()
        sys.exit()

    def drop(self, manual):
        if not self.gameover and not self.paused:
            self.score += .1 if manual else 0
            self.stone_y += 1
            if check_collision(self.board,
                               self.stone,
                               (self.stone_x, self.stone_y)):
                self.board = join_matrixes(
                  self.board,
                  self.stone,
                  (self.stone_x, self.stone_y))
                self.new_stone()
                cleared_rows = 0
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = remove_row(
                              self.board, i)
                            cleared_rows += 1
                            break
                    else:
                        break
                self.add_cl_lines(cleared_rows)
                return True
        return False

    def insta_drop(self):
        if not self.gameover and not self.paused:
            while(not self.drop(True)):
                pass

    def rotate_stone(self):
        if not self.gameover and not self.paused:
            new_stone = rotate_clockwise(self.stone)
            if not check_collision(self.board,
                                   new_stone,
                                   (self.stone_x, self.stone_y)):
                self.stone = new_stone

    def toggle_pause(self):
        self.paused = not self.paused

    def start_game(self):
        if self.gameover:
            self.init_game()
            self.gameover = False

    def run(self, net, agent):
        key_actions = {
            'ESCAPE':   self.quit,
            'LEFT':     lambda:self.move(-1),
            'RIGHT':    lambda:self.move(+1),
            'DOWN':     lambda:self.drop(True),
            'UP':       self.rotate_stone,
            'p':        self.toggle_pause,
            'SPACE':    self.start_game,
            'RETURN':   self.insta_drop
        }

        self.gameover = False
        self.paused = False

        if not HEADLESS:
            dont_burn_my_cpu = pygame.time.Clock()
        while 1:
            if not HEADLESS:
                self.screen.fill((0,0,0))
            if self.gameover:
                self.genome.fitness -= compute_endgame_fitness(self.board)
                #print(fitness)
                return self.genome.fitness
            elif not HEADLESS:
                pygame.draw.line(self.screen,(255,255,255),(self.rlim+1, 0),(self.rlim+1, self.height-1))
                self.disp_msg("Next:", (self.rlim+cell_size, 2))
                self.disp_msg("Score: %d\n\nLevel: %d\nLines: %d" % (self.score, self.level, self.lines), (self.rlim+cell_size, cell_size*5))
                self.draw_matrix(self.bground_grid, (0,0))
                self.draw_matrix(self.board, (0,0))
                self.draw_matrix(self.stone, (self.stone_x, self.stone_y))
                self.draw_matrix(self.next_stone, (cols+1,2))
            
            if not HEADLESS:
                pygame.display.update()

            stone = numpy.concatenate(self.stone)
            next_stone = numpy.concatenate(self.stone)

            stone = numpy.append(stone, [0] * (12 - len(stone)))
            next_stone = numpy.append(next_stone, [0] * (12 - len(next_stone)))
                
            decision = net.activate((*numpy.concatenate(self.board), *stone, self.stone_x, self.stone_y, *next_stone))

            clean_board(self.board)

            options = ['LEFT','RIGHT','DOWN','UP', 'RETURN']
            for i in range(len(options)):
                if decision[i] > .5:
                    key_actions[options[i]]()
                    

            

            self.drop(False)
            if not HEADLESS:
                dont_burn_my_cpu.tick(maxfps)

GEN = 0

def eval_genomes(genomes, config):
    global GEN
    nets = []
    ge = []
    
    GEN+=1
    if GEN == 0:
        try:
            with open(INPUT_FILE_PATH, 'rb') as previous_best:
                genome = pickle.load(previous_best)
                genomes.append(len(genomes), genome)
        except:
            print("Cannot load previous best")

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        ge.append(genome)
        nets.append(net)
        App = TetrisApp(genome)
        App.run(net, genome)
        print(genome.fitness)
        del App
if __name__ == '__main__':
    config_path = os.path.join(os.getcwd(), "NEATconfig.txt")
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
    neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes , 40)

    with open(OUTPUT_FILE_PATH, "wb") as f:
        pickle.dump(winner, f)
        f.close()

