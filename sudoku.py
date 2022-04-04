import copy
from random import choice, random, randint, sample
import time

class Puzzle:
    """
    Puzzle class to represent individual sudoku puzzles
    Contains the puzzle and its respective weight
    """

    def __init__(self, base_puzzle : list, puzzle : list, fitness : float):
        self.base_puzzle = base_puzzle
        self.puzzle = puzzle
        self.fitness = fitness
        self.won = 0

    def set_puzzle(self, puzzle):
        self.puzzle = puzzle

    def set_fitness(self):
        self.fitness = fitness(self.puzzle)

    def get_fitness(self):
        return self.fitness

    def get_puzzle(self):
        return self.puzzle

    def print_puzzle(self):
        for i in self.puzzle:
            print(i)

    def check_won(self):
        if self.fitness == 1.0:
            self.won = 1

    def set_col(self, col_index, vals):
        self.puzzle[col_index] = vals

    def set_row(self, row_index, vals):
        for i in range(9):
            self.puzzle[i][row_index] = vals[i]

    def get_col(self, col_index):
        vals = []
        for i in range(9):
            vals.append(self.puzzle[i][col_index])
        return vals

    def get_row(self, row_index):
        vals = self.puzzle[row_index]
        return vals

    def get_val_at_pos(self, row, col):
        return self.puzzle[row][col]

    def update_position(self, val, xcoord, ycoord):
        self.puzzle[xcoord][ycoord] = val

    def count_vals(self):
        total = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(self.puzzle)):
            for j in range(len(self.puzzle)):
                val = self.puzzle[i][j]
                total[val - 1] += 1

        spread = 0
        for i in total:
            spread += abs(i-9)

        return spread

    def mutate(self, mutation_chance):

        # chance of mutation
        if mutation_chance <= .25:
            spread = self.count_vals()
            mutation_type = random()
            if spread <= 25 and mutation_type < .5:
                # random switching of two values
                switching = randint(1, 2)
                number_of_switches = 0
                while number_of_switches <= switching:
                    # loops through swaps
                    can_be_switched_one = False
                    can_be_switched_two = False
                    # makes sure that the values that are going to be swapped are not in the main puzzle
                    while not can_be_switched_one:
                        one_row_pos = randint(0, 8)
                        one_col_pos = randint(0, 8)
                        if self.base_puzzle[one_row_pos][one_col_pos] == 0:
                            can_be_switched_one = True

                    while not can_be_switched_two:
                        two_row_pos = randint(0, 8)
                        two_col_pos = randint(0, 8)
                        if self.base_puzzle[two_row_pos][two_col_pos] == 0:
                            can_be_switched_two = True

                    # gets the values
                    value1 = self.get_val_at_pos(one_row_pos, one_col_pos)
                    value2 = self.get_val_at_pos(two_row_pos, two_col_pos)
                    # updates the values
                    self.update_position(value1, two_row_pos, two_col_pos)
                    self.update_position(value2, one_row_pos, one_col_pos)
                    number_of_switches += 1

            else:
                # random values are changed to other random values
                number_of_changes = randint(1, 2)
                # decides how many values to change
                changed = 0
                while changed <= number_of_changes:
                    # loops through changing values until the number of values changed is what is desired
                    can_be_swapped = False

                    while not can_be_swapped:
                        # makes sure the change is allowed
                        row_position = randint(0, 8)
                        col_position = randint(0, 8)
                        if self.base_puzzle[row_position][col_position] == 0:
                            can_be_swapped = True

                    new_value = randint(1, 9)
                    self.update_position(new_value, row_position, col_position)
                    changed += 1


def evolve(pop_size, base_puzzle):
    """
    function to run the evolution
    :param pop_size: size of the population
    :param base_puzzle: the starting sudoku puzzle
    :return: returns the population
    """
    start_time = time.time()
    population = create_pop(pop_size, base_puzzle)
    solved = False
    for gen in range(100):
    #while not solved:
        # selects parents and crosses them for the child population
        offspring_population = select_and_cross(population)
        population.extend(offspring_population)
        # does not mutate the top 5% of the population
        population = mutate_population(population, pop_size, .05)
        update_fitness(population)
        population.sort(key=lambda x: x.fitness, reverse=True)
        # trims the population
        population = population[:pop_size]
        print(population[0].get_fitness())
        print(population[0].get_puzzle())
        if population[0].get_fitness() >= 1.0:
            now_time = time.time()
            print("Time taken: ", end='')
            print(round(float(now_time - start_time), 6))
            print("Final fitness: %f \nFinal Puzzle:" % population[0].get_fitness())
            print(population[0].get_puzzle())
            return population
            # solved = True
    now_time = time.time()
    print("Time taken: ", end='')
    print(round(float(now_time - start_time), 6))
    print("Final fitness: %f \nFinal Puzzle:" % population[0].get_fitness())
    print(population[0].get_puzzle())
    return population


# POPULATION-LEVEL OPERATORS


def create_pop(pop_size, base_puzzle):
    """
    Function to create an initial population of randomly populated sudoku puzzles
    :param pop_size: the size of the population created
    :return: the new population
    """
    pop = []
    for i in range(pop_size):
        created = create_ind(base_puzzle)
        weight = fitness(created)
        puzzle1 = Puzzle(base_puzzle, created, weight)
        pop.append(puzzle1)

    return pop


def update_fitness(population):
    """
    evaluates the population and updates the weight in the
    :param population: the whole population
    :return:
    """
    for individual in population:
        individual.set_fitness()

    return population


def select_and_cross(pop):
    """
    selects a population and crosses them returning a new population
    :param pop: the current population
    :return: returns a list of the child population
    """
    population = copy.deepcopy(pop)
    # not sure if the population should be copied or if individuals should be copied in the while loop
    children = []

    new_pop_size = len(population)*.5
    # new_pop_size = len(population)

    while len(children) < new_pop_size:
        # parents are selected using binary tournament selection
        parent1 = tournament(population)
        parent2 = tournament(population)

        child1, child2 = crossover(parent1, parent2)
        children.append(child1)
        children.append(child2)

    return children


def tournament(pop):
    """
    runs a binary tournament to pick new parents
    :param pop: the current population
    :return: an individual
    """
    individual = 0
    for i in range(2):
        new_individual = choice(pop)
        if individual == 0 or new_individual.get_fitness() > individual.get_fitness():
            individual = new_individual
    return individual


def crossover(individual1, individual2):
    """
    performs a crossover on two individuals
    :param individual1: one member of the population
    :param individual2: another member of the population
    :return: returns crossed individuals
    """

    # gets a random number between 2 and 7 to determine how many rows are crossed over
    switching = randint(2, 5)
    # gets the random amount of indexes to be crossed over
    indexes = sample([0, 1, 2, 3, 4, 5, 6, 7, 8], switching)
    for j in indexes:
        # gets row then performs crossover operations
        first = individual1.get_row(j)
        second = individual2.get_row(j)
        individual1.set_row(j, second)
        individual2.set_row(j, first)

    return individual1, individual2


def mutate_population(population, pop_size, keeps):
    """
    mutates the population by looping through the population and calling mutate
    :param population: the full population
    :param locked: the locked indexes which cannot be changed
    :return: the new population
    """
    iteration = 0
    for i in population:
        if iteration < pop_size * keeps:
            iteration += 1
        else:
            mutation_chance = random()
            i.mutate(mutation_chance)

    return population


# INDIVIDUAL-LEVEL OPERATORS: REPRESENTATION & PROBLEM SPECIFIC
# could move them to puzzle class


def create_ind(puzzle):
    """
    Creates an individual random puzzle
    :param puzzle: the starting puzzle
    :return: returns the populated puzzle
    """
    new_puzzle = copy.deepcopy(puzzle)
    for i in range(9):
        for j in range(9):
            if new_puzzle[i][j] == 0:
                new_puzzle[i][j] = randint(1, 9)

    return new_puzzle


def fitness(individual):
    """
    gets the fitness of an individual by counting the amount of duplicates in each row, column, and box
    :param individual: the individual in the population
    :return: returns the fitness of the individual as a float between 1 and 0 where 1 is the best
    """
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # fitness of rows and columns

    row_count = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    col_count = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    box_count = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    row_sum = 0
    col_sum = 0
    box_sum = 0

    for i in range(9):
        for j in range(9):
            row_val = individual[i][j]  # gets the value of the square at that index
            row_count[row_val - 1] += 1  # increases the count by 1
            col_val = individual[j][i]  # gets the value of the square at that index
            col_count[col_val - 1] += 1  # increases the count by 1

        col_sum += (1 / len(set(row_count))) / 9
        col_count = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        row_sum += (1 / len(set(row_count))) / 9
        row_count = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box_val1 = individual[i][j]  # top
            box_count[box_val1 - 1] += 1
            box_val2 = individual[i][j + 1]  # top middle
            box_count[box_val2 - 1] += 1
            box_val3 = individual[i][j + 2]  # top right
            box_count[box_val3 - 1] += 1

            box_val4 = individual[i + 1][j]  # middle left
            box_count[box_val4 - 1] += 1
            box_val5 = individual[i + 1][j + 1]  # middle middle
            box_count[box_val5 - 1] += 1
            box_val6 = individual[i + 1][j + 2]  # middle right
            box_count[box_val6 - 1] += 1

            box_val7 = individual[i + 2][j]  # bottom left
            box_count[box_val7 - 1] += 1
            box_val8 = individual[i + 2][j + 1]  # bottom middle
            box_count[box_val8 - 1] += 1
            box_val9 = individual[i + 2][j + 2]  # bottom right
            box_count[box_val9 - 1] += 1

            box_sum += (1 / len(set(box_count))) / 9
            box_count = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    total_fitness = (box_sum + row_sum + col_sum)/3
    return total_fitness


NUMBER_GENERATION = 100


if __name__ == '__main__':
    puzzle1 = \
        [
            [3, 0, 0, 0, 0, 5, 0, 4, 7],
            [0, 0, 6, 0, 4, 2, 0, 0, 1],
            [0, 0, 0, 0, 0, 7, 8, 9, 0],
            [0, 5, 0, 0, 1, 6, 0, 0, 2],
            [0, 0, 3, 0, 0, 0, 0, 0, 4],
            [8, 1, 0, 0, 0, 0, 7, 0, 0],
            [0, 0, 2, 0, 0, 0, 4, 0, 0],
            [5, 6, 0, 8, 7, 0, 1, 0, 0],
            [0, 0, 0, 3, 0, 0, 6, 0, 0],
        ]

    puzzle2 = \
        [
            [0, 0, 2, 0, 0, 0, 6, 3, 4],
            [1, 0, 6, 0, 0, 0, 5, 8, 0],
            [0, 0, 7, 3, 0, 0, 2, 9, 0],
            [0, 8, 5, 0, 0, 1, 0, 0, 6],
            [0, 0, 0, 7, 5, 0, 0, 2, 3],
            [0, 0, 3, 0, 0, 0, 0, 5, 0],
            [3, 1, 4, 0, 0, 2, 0, 0, 0],
            [0, 0, 9, 0, 8, 0, 4, 0, 0],
            [7, 2, 0, 0, 4, 0, 0, 0, 9]
        ]

    puzzle3 = \
        [
            [0, 0, 4, 0, 1, 0, 0, 6, 0],
            [9, 0, 0, 0, 0, 0, 0, 3, 0],
            [0, 5, 0, 7, 9, 6, 0, 0, 0],
            [0, 0, 2, 5, 0, 4, 9, 0, 0],
            [0, 8, 3, 0, 6, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 6, 0, 7],
            [0, 0, 0, 9, 0, 3, 0, 7, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 6, 0, 0, 0, 0, 1, 0]
        ]

    solved = \
        [
            [6, 9, 4, 1, 3, 8, 2, 5, 7],
            [8, 7, 2, 9, 4, 5, 6, 1, 3],
            [5, 1, 3, 2, 7, 6, 9, 8, 4],
            [9, 2, 6, 5, 1, 4, 3, 7, 8],
            [1, 5, 8, 3, 6, 7, 4, 2, 9],
            [3, 4, 7, 8, 2, 9, 5, 6, 1],
            [4, 6, 9, 7, 5, 1, 8, 3, 2],
            [2, 8, 1, 6, 9, 3, 7, 4, 5],
            [7, 3, 5, 4, 8, 2, 1, 9, 6],
        ]

    #evolved = evolve(10000, puzzle3)

    print(fitness(puzzle2))

