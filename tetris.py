#!/usr/bin/env python3
#-*- coding: utf-8 -*-


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
import threading, datetime
import pickle, shutil
from heuristics import *

from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from stat import S_IREAD, S_IRGRP, S_IROTH

RUN_ID = str(datetime.datetime.now()).replace(":","_").replace("-","_").replace(" ","_")
RUNS_PATH = os.path.join(os.getcwd(), 'Runs')
if not os.path.exists(RUNS_PATH):
    os.mkdir(RUNS_PATH)
    
RUN_PATH = os.path.join(RUNS_PATH, RUN_ID)
if not os.path.exists(RUN_PATH):
    os.mkdir(RUN_PATH)
LOG_FILE = os.path.join(RUN_PATH, 'LOG.TXT')
SCRIPT_NAME = 'tetris.py'
CONFIG_NAME = 'NEATconfig.txt'
HEADLESS = True

shutil.copyfile(os.path.join(os.getcwd(), SCRIPT_NAME), os.path.join(RUN_PATH, SCRIPT_NAME))
shutil.copyfile(os.path.join(os.getcwd(), CONFIG_NAME), os.path.join(RUN_PATH, CONFIG_NAME))

os.chmod(os.path.join(RUN_PATH, SCRIPT_NAME), S_IREAD|S_IRGRP|S_IROTH)
os.chmod(os.path.join(RUN_PATH, CONFIG_NAME), S_IREAD|S_IRGRP|S_IROTH)

OUTPUT_FILE_PATH = "winner.pkl"
INPUT_FILE_PATH = "winner.pkl"
# The configuration
cell_size = 18
cols =      10
rows =      22
maxfps =    20

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
        self.gameover = False
        self.paused = False
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

    def drop_timer(self, ):
        self.drop(False)
        threading.Timer(self.drop_time, self.drop_timer).start()

    def init_game(self):
        self.board = new_board()
        self.new_stone(first=True)
        self.level = 1
        self.score = 0
        self.lines = 0
        self.drop_time = .01
        if not HEADLESS:
            pygame.time.set_timer(pygame.USEREVENT+1, self.drop_time)
        else:
            self.drop_timer()

    def disp_msg(self, msg, topleft):
        if HEADLESS:
            return
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
        if HEADLESS:
            return
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
        if HEADLESS:
            return
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
            self.drop_time = newdelay  / 5
            if not HEADLESS:
                pygame.time.set_timer(pygame.USEREVENT+1, self.drop_time)

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
        if not HEADLESS:
            self.center_msg("Exiting...")
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


            if not HEADLESS:
                dont_burn_my_cpu.tick(maxfps)
                for event in pygame.event.get():
                    if event.type == pygame.USEREVENT+1:
                        self.drop(False)
                    elif event.type == pygame.QUIT:
                        self.quit()

class FileReporter(neat.reporting.BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""

    def print(self, msg):
        with open(self.file_name, 'a+') as log_file:
            log_file.write(str(msg) + '\n')
    
    def __init__(self, show_species_detail, file_name):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.file_name = file_name

    def start_generation(self, generation):
        self.generation = generation
        self.print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            self.print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            sids = list(iterkeys(species_set.species))
            sids.sort()
            self.print("   ID   age  size  fitness  adj fit  stag")
            self.print("  ====  ===  ====  =======  =======  ====")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                self.print(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}".format(sid, a, n, f, af, st))
        else:
            self.print('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        self.print('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            self.print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            self.print("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        self.print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        self.print(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def complete_extinction(self):
        self.num_extinctions += 1
        self.print('All species extinct.')

    def found_solution(self, config, generation, best):
        self.print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            self.print("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        self.print(msg)

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
        
    ids = []
    fits = []
    for i in range(len(genomes)):
        ids.append(genomes[i][0])
        fits.append(genomes[i][1].fitness)
    
    plt.scatter(ids, fits)
    plt.savefig(os.path.join(RUN_PATH, f"Generation_{GEN}_scatter.png"))
    plt.clf()
if __name__ == '__main__':
    config_path = os.path.join(os.getcwd(), "NEATconfig.txt")
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, 
    neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    population = neat.Population(config)

    population.add_reporter(FileReporter(True, LOG_FILE))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes , 75)

    with open(OUTPUT_FILE_PATH, "wb") as f:
        pickle.dump(winner, f)
        f.close()

