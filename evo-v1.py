#
# evo-v1.py
#
# a. Representação das soluções (indivíduos):
#
#    Vetor com 32 números. Os 30 primeiros codificam cada x_i da função de
#    Ackley, na sequência o desvio padrão e fitness do indivíduo.
#
# b. Função de Fitness:
#
#    A função de fitness retorna o resultado dos 30 primeiros números do
#    indivíduo, os x_i, na função de ackley.
#
# c. População (tamanho, inicialização, etc):
#
#    População de 30 indivíduos, inicializados com x_i aleatórios obedecendo
#    a regra -15 <= x_i <= 15. O desvio padrão é inicializado aleatoriamente
#    com números entra os argumentos "gaussian_deviation_min" e "...ation_max".
#
# d. Processo de seleção:
#
#    Os pais são escolhidos aleatoriamente. São gerados 200 filhos, os quais,
#    podem ser gerados por múltiplos pais, especificamente, de 2 a 15.
#
# e. Operadores Genéticos (Recombinação e Mutação)
#
#    Recombinação: Os x_i do filho são escolhidos aleatoriamente entre os x_i
#    dos pais que componhem a família. O sigma do filho é a média do dos pais.
#
#    Mutação: Foi implementada a mutação Uncorrelated Mutation with One Step Size,
#    descrita na página 58 do livro do Eiben (segunda edição).
#
# f. Processo de seleção por sobrevivência
#
#    Processo de seleção geracional, os 30 melhores filhos permanecem.
#
# g. Condições de término do Algoritmo Evolucionário
#
#    Um milhão de iterações (por hora, número escolhido arbitrariamente).
#


import math
import random


'''
Parameters.
'''

individuals_count = 30
childs_count = 200

ackley_n = 30
ackley_x_min = -15
ackley_x_max =  15

gaussian_deviation_min = 2
gaussian_deviation_max = 8
learning_rate = 1/math.sqrt(individuals_count)

max_iteration_count = 1000000 #200000


'''
Metrics.
'''

solution = []


'''
Ackley function implementation.
'''

def ackley(v):

    const_n = len(v)

    const_c1 = 20.0
    const_c2 = 0.2
    const_c3 = 2*math.pi

    a = sum([math.pow(x, 2) for x in v])
    b = (1/const_n)*a
    c = (-const_c2)*math.sqrt(b)
    d = (-const_c1)*math.exp(c)

    e = sum([math.cos(const_c3*x) for x in v])
    f = (1/const_n)*e
    g = math.exp(f)

    h = d - g + const_c1 + math.exp(1)

    return h


'''
Assumes the fitness is equal to the ackley function evaluation.
'''

def fitness(individual):

    '''
    The last two numbers in the individual (list) represents the
    standard deviation associated to it and its fitness. We don't
    need to pass them to the ackley evaluation, hence, the sublist.
    '''

    return ackley(individual[:-2])


'''
Generates a random individual. The individuals are represented as lists of
floats, composed by: the first "ackley_n" numbers are the solution vector, then
the mutation step size (gaussian standard deviation) and the individual fitness.
'''

def generateIndividual():

    individual = [random.uniform(ackley_x_min, ackley_x_max) for _ in range(ackley_n)]
    individual.extend([0, 0])
    individual[-2] = random.uniform(gaussian_deviation_min, gaussian_deviation_max)
    individual[-1] = fitness(individual)

    return individual


'''
Generate the initial population.
'''

population = []

for i in range(individuals_count):

    population.append(generateIndividual())

solution = population[0]


'''
Main algorithm loop.
'''

for iteration in range(max_iteration_count):

    '''
    Print the current iteration and the best fitness found.
    '''

    print("{:7.0f}".format(iteration) + " " + str(solution[-1]))

    '''
    Parents selection & Recombination. We adopt a global recombination strategy
    along with a discrete recombination for the object part and an intermediate
    recombination for the strategy parameters. So, basically, each family can
    have multiple parents generating a child whose alleles [x1...xi] are
    selected randomically from their parents and the strategy parameters are
    averaged.
    '''

    childs = []

    for i in range(childs_count):

        count = random.randint(2, (individuals_count/2))
        family = [random.choice(population) for _ in range(count)]

        child = [random.choice(family)[k] for k in range(ackley_n)]
        child.extend([0, 0])
        child[-2] = sum(a[-2] for a in family)/count
        child[-1] = fitness(child)

        childs.append(child)

    '''
    Mutate childs (Uncorrelated Mutation With One Step Size, Eiben Pag. 58).
    '''

    for child in childs:

        g = random.gauss(0, 1)
        sigma = child[-2]
        sigma_l = sigma*math.exp(learning_rate*g)

        if sigma_l < gaussian_deviation_min:

            sigma_l = gaussian_deviation_min

        for i in range(len(child)):

            g = random.gauss(0, 1)
            x = child[i]+(sigma_l*g)

            if ackley_x_min <= x and x <= ackley_x_max:

                child[i] = x

        child[-2] = sigma_l
        child[-1] = fitness(child)

    '''
    Survivor selection.
    '''

    childs.sort(key=lambda x: x[-1], reverse=False)

    population = childs[:individuals_count]

    '''
    Update metrics (retain best solution found).
    '''

    if solution[-1] > population[0][-1]:

        solution = population[0]

    if population[0][-1] < 0:

        print("")
        print("")
        print("ERROR:")
        print("")
        print(population[0])
        print("")
        print("")


'''
Present results.
'''

print("")
print("Best solution (" + str(solution[-1]) + "):")
print("")
