#
# evo-v2.py
#
# a. Representação das soluções (indivíduos):
#
#    Vetor com [(2*ackley_n)+1] números. Os primeiros ackley_n números codificam
#    cada x_i da função de Ackley, os  ackley_n seguintes representam o passo de
#    mutação em cada direção dimensão i e o último número representa o fitness
#    do indivíduo.
#
# b. Função de Fitness:
#
#    A função de fitness retorna o resultado dos primeiros ackley_n números do
#    indivíduo, os x_i, aplicados na função de Ackley.
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
#    podem ser gerados por múltiplos pais, especificamente, de 2 a 15 pais.
#
# e. Operadores Genéticos (Recombinação e Mutação)
#
#    Recombinação: Os x_i do filho são escolhidos aleatoriamente entre os x_i
#    dos pais que componhem a família. O passo de mutação de cada dimensão é
#    atualizado com a média dos pais.
#
#    Mutação: Foi implementada a mutação Uncorrelated Mutation with N Step Sizes,
#    descrita na página 60 do livro do Eiben (segunda edição).
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

gaussian_deviation_min = 0.0001
gaussian_deviation_max = 12

learning_rate = 1/math.sqrt(2*math.sqrt(individuals_count))
learning_rate_l = 1/math.sqrt(2*individuals_count)

max_iteration_count = 1000000


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
    We only need the first "ackley_n" values to evaluate the function,
    anything after is disregarded, hence, the sublist below.
    '''

    return ackley(individual[:ackley_n])


'''
Generates a random individual. The individuals are represented as lists of
floats, composed by: the first "ackley_n" numbers are the solution vector,
the following "ackley_n" values are the mutation step size in each  direction
and then a single number representing the individual fitness.
'''

def generateIndividual():

    min = gaussian_deviation_min
    max = gaussian_deviation_max

    x_vals = [random.uniform(ackley_x_min, ackley_x_max) for _ in range(ackley_n)]
    sigma_vals =  [random.uniform(min, max) for _ in range(ackley_n)]

    individual = x_vals + sigma_vals + [0]
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

        x_vals = [random.choice(family)[k] for k in range(ackley_n)]
        sigma_vals = [(sum(a[k+ackley_n] for a in family)/count) for k in range(ackley_n)]

        child = x_vals + sigma_vals + [0]
        child[-1] = fitness(child)

        childs.append(child)

    '''
    Mutate childs (Uncorrelated Mutation With N Step Sizes, Eiben Pag. 60).
    '''

    for child in childs:

        const_g = random.gauss(0, 1)

        for i in range(ackley_n):

            g = random.gauss(0, 1)
            a = learning_rate_l*const_g
            b = learning_rate*g

            sigma = child[i+ackley_n]
            sigma_l = sigma*math.exp(a*b)

            if sigma_l < gaussian_deviation_min:

                sigma_l = gaussian_deviation_min

            child[i+ackley_n] = sigma_l

        for i in range(ackley_n):

            g = random.gauss(0, 1)
            x = child[i]+(child[i+ackley_n]*g)

            if ackley_x_min <= x and x <= ackley_x_max:

                child[i] = x

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
