import random
import sys

Cities = ['Houston', 'Dallas', 'Austin', 'Abilene', 'Waco']
Houston = [0, 241, 162, 351, 183]
Dallas = [241, 0, 202, 186, 97]
Austin = [162, 202, 0, 216, 106]
Abilene = [351, 186, 216, 0, 186]
Waco = [183, 97, 106, 186, 0]
lookup = [Houston, Dallas, Austin, Abilene, Waco]


def generate_population(size):
  choices = [1, 2, 3, 4]
  population = []
  for _ in range(size):
    new_individual = [0]
    new_individual.extend(random.sample(choices, len(choices)))
    new_individual.append(0)
    population.append(new_individual)
  return population


def fitness(individual):
  distance = 0
  for city_idx in range(len(individual) - 1):
    from_city = individual[city_idx]
    to_city = individual[city_idx + 1]
    distance = distance + lookup[from_city][to_city]
  return distance


def population_fitness(population):
  population_fitness_array = []
  for individual in population:
    population_fitness_array.append(fitness(individual))
  return population_fitness_array


# Not actually crossover since it is only taking genetic information from one parent at a time
'''
def crossover(parent_1, parent_2, crossover_rate):
  if random.uniform(0, 1) < crossover_rate:
    parent_1 = list(filter(lambda x: x != 0, parent_1))
    parent_2 = list(filter(lambda x: x != 0, parent_2))
    child_1 = parent_1.copy()
    child_2 = parent_2.copy()
    
    child_1 = create_child(parent_1, child_1)
    child_2 = create_child(parent_2, child_2)
    return child_1, child_2
  return parent_1, parent_2
'''


def crossover(parent_1, parent_2, crossover_rate):
  if random.uniform(0, 1) < crossover_rate:
    child_1 = create_child(parent_1, parent_2)
    child_2 = create_child(parent_2, parent_1)
    return child_1, child_2
  return parent_1, parent_2


def create_child(parent_1, parent_2):
  parent_1 = list(filter(lambda x: x != 0, parent_1))
  parent_2 = list(filter(lambda x: x != 0, parent_2))
  child_1 = [-1] * len(parent_1)
  city_1 = random.randint(0, len(parent_1))
  city_2 = random.randint(0, len(parent_1))
  
  if city_1 > city_2:
    temp = city_1
    city_1 = city_2
    city_2 = temp
    
  parent_1_slice = parent_1[city_1:city_2]
  child_1[city_1:city_2] = parent_1_slice
  
  child_idx = city_2
  parent_idx = city_2
  while -1 in child_1:
    if child_idx >= len(child_1):
      child_idx = 0
    if parent_idx >= len(parent_2):
      parent_idx = 0
    if parent_2[parent_idx] not in child_1:
      child_1[child_idx] = parent_2[parent_idx]
      child_idx = child_idx + 1
    parent_idx = parent_idx + 1
  child_1.insert(0, 0)
  child_1.append(0)
  return child_1


'''
def create_child(parent, child):
  random_idx = random.randint(0, len(parent) - 2)
  crossoverFromCity = parent[random_idx]
  child.remove(crossoverFromCity)
  crossoverToCity = parent[random_idx + 1]
  child.remove(crossoverToCity)
  
  random.shuffle(child)
  random_idx = random.randint(0, len(parent))
  child.insert(random_idx, crossoverFromCity)
  child.insert(random_idx + 1, crossoverToCity)
  
  child.insert(0, 0)
  child.append(0)
  return child
'''


def mutate(individual, mutate_rate):
  #for city_idx in range(1, len(individual) - 1):
  if random.uniform(0, 1) < mutate_rate:
    city_idx = random.randint(1, len(individual) - 2)
    city_to_swap_idx = random.randint(1, len(individual) - 2)
    temp = individual[city_to_swap_idx]
    individual[city_to_swap_idx] = individual[city_idx]
    individual[city_idx] = temp
  return individual
  

def tournament(population, population_fitness_array, n=2):
  fit_max = sys.maxsize
  for rep in range(n):
    index = random.randrange(0, len(population))
    if population_fitness_array[index] < fit_max:
      fit_max = population_fitness_array[index]
      best_individual = population[index]
  return best_individual


def pair_parents(population):
  return [[population[i], population[i + 1]] for i in range(0, len(population), 2)]


def unfold_children(childPairs):
  children = []
  for childPair in childPairs:
    for child in childPair:
      children.append(child)
  return children


def best_individual(population):
  best_score = sys.maxsize
  best = []
  for individual in population:
    individual_score = fitness(individual)
    if individual_score < best_score:
      best_score = individual_score
      best = individual
  return best
  

def main(population_size, max_generations, n, crossover_rate, mutate_rate):
  population = generate_population(population_size)
  population_fitness_array = population_fitness(population)
  
  for generation in range(max_generations):
    tournament_population = [tournament(population, population_fitness_array, n=n) for _ in population]
    parent_pairs = pair_parents(tournament_population)
    child_pairs = [crossover(pair[0], pair[1], crossover_rate) for pair in parent_pairs]
    children = unfold_children(child_pairs)
    population = [mutate(individual.copy(), mutate_rate) for individual in children]
  for pop in population:
    print([fitness(pop),pop])
  print(f'Average Fitness: {sum(population_fitness(children))/len(children)}')
  print(f'Best Fitness Score:{fitness(best_individual(children))}')
  print(f'Best Route: {best_individual(children)}')


main(population_size=100, max_generations=10, n=2, crossover_rate=0.4, mutate_rate=0.03)
