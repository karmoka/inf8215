import numpy as np
import math
import copy
from collections import deque
from itertools import chain

class State:
    
    """
    Contructeur d'un état initial
    """
    def __init__(self, pos):
        """
        pos donne la position de la voiture i dans sa ligne ou colonne (première case occupée par la voiture);
        """
        self.pos = np.array(pos)
        
        """
        c,d et prev premettent de retracer l'état précédent et le dernier mouvement effectué
        """
        self.c = self.d = self.prev = None
        
        self.nb_moves = 0
        self.score = 0
        self.rock = None
        self.cycle = set()

    """
    Constructeur d'un état à partir du mouvement (c,d)
    """
    def move(self, c, d):
        s = State(self.pos)
        s.prev = self
        s.pos[c] += d
        s.c = c
        s.d = d
        s.nb_moves = self.nb_moves + 1
        s.rock = self.rock
        return s

    def put_rock(self, rock_pos):
        s = State(self.pos)
        s.prev = self
        s.c = self.c
        s.d = self.d
        s.nb_moves = self.nb_moves + 1
        s.rock = rock_pos
        return s
    
    def calc_cost(self, rh, car, range_to_check):
        if range_to_check is None:
            return 1000
        # cost = len(range_to_check)
        cost = len(range_to_check)
        for i in range_to_check:
            x, y = (rh.move_on[car], i)
            if not rh.horiz[car]:
                x,y = y,x

            next_car = self.find_car(rh, x, y)
            if next_car is None:
                cost += rh.length[car]
                continue
            if next_car in self.cycle:
                cost += 10 # Eviter les cycles recursifs
                continue
            cost += self.cost_to_free(rh, x, y)
        return cost
        
    
    def cost_to_free(self, rh, row, column):
        car = self.find_car(rh, row, column)
        self.cycle.add(car)
            
        x,y = row, column
        if rh.horiz[car]:
            x,y = y,x
            
        check_forward = (5 - x) >= rh.length[car]
        check_backward = x >= rh.length[car]
            
        backward = range(x-rh.length[car], self.pos[car]) if check_backward else None
        forward = range(self.pos[car] + rh.length[car], x + rh.length[car] + 1) if check_forward else None
        return min(self.calc_cost(rh, car, backward), self.calc_cost(rh, car, forward))
        
    def find_car(self, rh, row, column):
        for car in range(0, len(self.pos)):
            for length in range(0, rh.length[car]):
                x,y = (rh.move_on[car], self.pos[car] + length)
                if not rh.horiz[car]:
                    x,y = y,x
                if x == row and y == column:
                    return car
        return None

    def score_state(self, rh):
        self.cycle = set()
        car_weight = 0
        for i in range(0, len(self.pos)):
            # Seulement les voitures verticales
            if not rh.horiz[i]:
                # Seulement si elles sont dans un colonnes devant l'auto (donc dans son chemin)
                if rh.move_on[i] > self.pos[0] + 1:
                    # Seulement si l'auto s'etend sur la rangee de l'auto (ligne 2)
                    if self.pos[i] <= 2 and (self.pos[i] + rh.length[i] - 1 ) >= 2:
                        car_weight += self.cost_to_free(rh, 2, rh.move_on[i])
                        self.cycle.add(i)
        self.score = (6 - self.pos[0] - 2) + car_weight + 10*len(self.cycle)
        return self.score
    
    def success(self):
        return self.pos[0] == 4
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        if len(self.pos) != len(other.pos):
            print("les états n'ont pas le même nombre de voitures")
        
        return np.array_equal(self.pos, other.pos)
    
    def __hash__(self):
        h = 0
        for i in range(len(self.pos)):
            h = 37*h + self.pos[i]
        return int(h)
    
    def __lt__(self, other):
        return (self.score) < (other.score)


class Rushhour:
    
    def __init__(self, horiz, length, move_on, color=None):
        self.nbcars = len(horiz)
        self.horiz = horiz
        self.length = length
        self.move_on = move_on
        self.color = color
        
        self.free_pos = None
    
    def init_positions(self, state):
        self.free_pos = np.ones((6, 6), dtype=bool)
        for i in range(self.nbcars):
            if self.horiz[i]:
                self.free_pos[self.move_on[i], state.pos[i]:state.pos[i]+self.length[i]] = False
            else:
                self.free_pos[state.pos[i]:state.pos[i]+self.length[i], self.move_on[i]] = False
        if state.rock is not None:
            self.free_pos[state.rock] = False
    
    def possible_moves(self, state):
        self.init_positions(state)
        new_states = []
        for i in range(self.nbcars):
            if self.horiz[i]:
                if state.pos[i]+self.length[i]-1 < 5 and self.free_pos[self.move_on[i], state.pos[i]+self.length[i]]:
                    new_states.append(state.move(i, +1))
                if state.pos[i] > 0 and self.free_pos[self.move_on[i], state.pos[i] - 1]:
                    new_states.append(state.move(i, -1))
            else:
                if state.pos[i]+self.length[i]-1 < 5 and self.free_pos[state.pos[i]+self.length[i], self.move_on[i]]:
                    new_states.append(state.move(i, +1))
                if state.pos[i] > 0 and self.free_pos[state.pos[i] - 1, self.move_on[i]]:
                    new_states.append(state.move(i, -1))
        return new_states
    
    def possible_rock_moves(self, state):
        self.init_positions(state)
        new_states =[]
        for row in range(len(self.free_pos)):
            if row == 2:
                continue
            for column in range(len(self.free_pos[0])):
                if state.rock is not None:
                    if row == state.rock[0] or column == state.rock[1]:
                        continue
                if self.free_pos[row][column]:
                    new_states.append((row, column))
        return new_states
    

    def print_pretty_grid(self, state):
        self.init_positions(state)
        grid= np.chararray((6, 6),unicode=True)
        grid[:]='-'
        for car in range(self.nbcars):
            for pos in range(state.pos[car], state.pos[car] +self.length[car]):
                if self.horiz[car]:
                    grid[self.move_on[car]][pos] = self.color[car][0]
                else:
                    grid[pos][self.move_on[car]] = self.color[car][0]
        if state.rock is not None:
            grid[state.rock] = 'x'
        # for row in grid:
        #     print(''.join(row))
        print(grid)

class MiniMaxSearch:
    def __init__(self, rushHour, initial_state, search_depth):
        self.rushhour = rushHour
        self.state = initial_state
        self.search_depth = search_depth

    def minimax_1(self, current_depth, current_state):
        moves = self.rushhour.possible_moves(current_state)
        if self.state in moves:
            moves.remove(self.state)
        for move in moves:
            move.score_state(self.rushhour)
        best = min(moves, key=lambda x: x.score)
        
        if current_depth == 0:
            print("check (c={})(d={:2d})(score={})".format(best.c, best.d, best.score))
            print(best.cycle)
        if current_depth < self.search_depth:
            for move in moves:
                if move.c == self.state.c and move.d == self.state.d * -1:
                    continue
                # Associe la valeur du meilleur enfant a son parent
                best_kid = self.minimax_1(current_depth + 1, move)

                # Si le futur de ce mouvement est avantageux, on l'adopte
                # On ne veut pas bouger des voitures qui ne bloquent rien et ne sont pas dams le chemin de la voiture rouge (pas dans un cycle)
                if move.c in best_kid.cycle and best_kid.score <= best.score: 
                    best = move
                    best.score = best_kid.score
                if current_depth == 0:
                    print("check (c={})(d={:2d})(score={})".format(move.c, move.d, best.score))
                    print(move.cycle)
        return best
    
    def minimax_2(self, current_depth, current_state, is_max): 
        #TODO
        return best_move

    def minimax_pruning(self, current_depth, current_state, is_max, alpha, beta):
        #TODO
        return best_move

    def expectimax(self, current_depth, current_state, is_max):
        #TODO
        return best_move

    def decide_best_move_1(self):
        self.state = self.minimax_1(0, self.state)
        self.print_move(True, self.state)
        self.rushhour.print_pretty_grid(self.state)

    def decide_best_move_2(self, is_max):
        #TODO
        pass

    def decide_best_move_pruning(self, is_max):
        # TODO
        pass

    def decide_best_move_expectimax(self, is_max):
        # TODO
        pass

    def solve(self, state, is_singleplayer):
         if is_singleplayer:
#             visited = set()
#             fifo = deque([state])
#             visited.add(state)
#             visit_count = 0

#             while fifo:
#                 visit_count += 1
#                 current_node = fifo.popleft()            
#                 if current_node.success():
#                     print("[Solve] {} Etats visites".format(visit_count))
#                     return current_node
#                 for elem in self.possible_moves(current_node):
#                     if elem not in visited:
#                         fifo.append(elem)
#                         visited.add(elem)
            self.state = state
            self.rushhour.print_pretty_grid(self.state)
            while not self.state.success():
                self.decide_best_move_1()

    def print_move(self, is_max, state):
        if state.c is None:
            print("Begin")
        elif is_max:
            color = self.rushhour.color[state.c]
            if state.d == 1:
                direction = "la droite" if self.rushhour.horiz[state.c] else "le bas" 
            elif state.d == -1:
                direction = "la gauche" if self.rushhour.horiz[state.c] else "le haut"

            print("Voiture {} vers {}".format(color, direction))
        else:
            print("Roche dans la case {}-{}".format(state.rock[0], state.rock[1]))

    def print_history(self):
        s = self.state
        history = []
        while s.prev != None:
            history.append(s.prev)
            s = s.prev
        history.reverse()
        for s in history:
            self.print_move(True, s)


# Solution optimale: 9 moves
# rh = Rushhour([True, False, False, False, True],
#                  [2, 3, 2, 3, 3],
#                  [2, 4, 5, 1, 5],
#                  ["rouge", "vert", "bleu", "orange", "jaune"])
# s = State([1, 0, 1, 3, 2])
# rh.print_pretty_grid(s)
# algo = MiniMaxSearch(rh, s,1) 
# algo.rushhour.init_positions(s)
# print(algo.rushhour.free_pos)
# algo.solve(s, True)


# rh = Rushhour([True, True, False, False, True, True, False, False],
#                  [2, 2, 3, 2, 3, 2, 3, 3],
#                  [2, 0, 0, 0, 5, 4, 5, 3],
#                  ["rouge", "vert", "mauve", "orange", "emeraude", "lime", "jaune", "bleu"])
# s = State([1, 0, 1, 4, 2, 4, 0, 1])
# algo = MiniMaxSearch(rh, s, 3) 
# algo.rushhour.init_positions(s)
# print(algo.rushhour.free_pos)
# algo.solve(s, True)

# solution optimale: 14 moves
rh = Rushhour([True, False, True, False, False, False, True, True, False, True, True],
                 [2, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2],
                 [2, 0, 0, 3, 4, 5, 3, 5, 2, 5, 4],
                 ["rouge", "vert", "mauve", "orange", "emeraude", "lime", "jaune", "bleu", "x", "y", "z"])
s = State([0, 0, 3, 1, 2, 1, 0, 0, 4, 3, 4])
algo = MiniMaxSearch(rh, s,3)
algo.rushhour.init_positions(s)
print(algo.rushhour.free_pos)
algo.solve(s, True)


algo.print_history()