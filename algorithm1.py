#
# algorithm1.py
#
# a. Representação das soluções (indivíduos):
#
#    Vetor com (ackley_n+2) números. Os primeiros "ackley_n" números codificam
#    cada x_i da função de Ackley, na sequência vem o desvio padrão e o fitness
#    do indivíduo.
#
# b. Função de Fitness:
#
#    A função de fitness retorna o resultado dos primeiros "ackley_n" números do
#    indivíduo, os x_i, aplicados na função de Ackley.
#
# c. População (tamanho, inicialização, etc):
#
#    População de 30 indivíduos, inicializados com x_i aleatórios obedecendo
#    a regra -15 <= x_i <= 15. O desvio padrão é inicializado aleatoriamente
#    com números entre os argumentos "gaussian_deviation_min" e "...ation_max".
#
# d. Processo de seleção:
#
#    Os pais são escolhidos aleatoriamente. São gerados 200 filhos, os quais,
#    podem ser gerados por múltiplos pais, especificamente, de 2 a 15 pais.
#
# e. Operadores Genéticos (Recombinação e Mutação):
#
#    Recombinação: Os x_i do filho são escolhidos aleatoriamente entre os x_i
#    dos pais que componhem a família. O sigma do filho é a média do dos pais.
#
#    Mutação: Foi implementada a mutação Uncorrelated Mutation with One Step Size,
#    descrita na página 58 do livro do Eiben (segunda edição).
#
# f. Processo de seleção por sobrevivência:
#
#    Processo de seleção geracional, os 30 melhores filhos permanecem.
#
# g. Condições de término do Algoritmo Evolucionário:
#
#    100000 iterações (número escolhido empiricamente).
#


import math
import random
import numpy as np
import matplotlib.pyplot as plot


'''
Parameters.
'''

individuals_count = 30
childs_count = 100

ackley_n = 30
ackley_x_min = -15
ackley_x_max =  15

gaussian_deviation_min = 0.00000000000000000001
gaussian_deviation_max = 12
learning_rate = 1/math.sqrt(individuals_count)

max_iteration_count = 100000


'''
Metrics.
'''

solution = []

sample_rate = 10

sampled_axis = []
sampled_bests = []
sampled_averages = []
sampled_deviations = []


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

    return ackley(individual[:ackley_n])


'''
Generates a random individual. An individual is represented as a list
containing (ackley_n+2) floats. The first "ackley_n" numbers are the x_i, the
last two represent the individual's mutation step size (std deviation) and fitness.
'''

def generateIndividual():

    x_values = [random.uniform(ackley_x_min, ackley_x_max) for _ in range(ackley_n)]
    step_size = random.uniform(gaussian_deviation_min, gaussian_deviation_max)
    individual = x_values + [step_size, fitness(x_values)]

    return individual


'''
Generate the initial population.
'''

population = []

for i in range(individuals_count):

    population.append(generateIndividual())


'''
Include the initial population in the metrics calculation.
'''

solution = population[0]

fitnesses = [p[-1] for p in population]

best = population[0][-1]
mean = np.mean(fitnesses)
std_deviation = np.std(fitnesses)

sampled_axis.append(0)
sampled_bests.append(best)
sampled_averages.append(mean)
sampled_deviations.append(std_deviation)


'''
Main algorithm loop.
'''

for iteration in range(max_iteration_count):

    '''
    Print the current iteration and the best fitness found.
    '''

    print("{:6.0f}".format(iteration+1) + " " + str(solution[-1]))

    '''
    Parents selection & Recombination. We adopt a global recombination strategy
    along with a discrete recombination for the object part and an intermediate
    recombination for the strategy parameters. So, basically, each family can
    have multiple parents generating a child whose alleles [x1...xi] are
    selected randomically from their parents and the strategy parameters are
    averages from the parents.
    '''

    childs = []

    for i in range(childs_count):

        count = random.randint(2, (individuals_count/2))
        family = [random.choice(population) for _ in range(count)]

        x_values = [random.choice(family)[k] for k in range(ackley_n)]
        step_size = sum(a[-2] for a in family)/count
        child = x_values + [step_size, fitness(x_values)]

        childs.append(child)

    '''
    Mutate childs using the Uncorrelated Mutation With One Step Size.
    '''

    for child in childs:

        g = random.gauss(0, 1)
        sigma = child[-2]
        sigma_l = sigma*math.exp(learning_rate*g)

        if sigma_l < gaussian_deviation_min:

            sigma_l = gaussian_deviation_min

        for i in range(ackley_n):

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

    '''
    Sampling, populate data sources to generate graphs later.
    '''

    if (iteration % sample_rate == 0):

        fitnesses = [p[-1] for p in population]

        best = population[0][-1]
        mean = np.mean(fitnesses)
        std_deviation = np.std(fitnesses)

        sampled_axis.append(iteration+1)
        sampled_bests.append(best)
        sampled_averages.append(mean)
        sampled_deviations.append(std_deviation)


'''
Present results.
'''

print("")
print("Best solution (" + str(solution[-1]) + "):")
print("")


'''
Generate graphs for the best individual, fitnesses mean and fitnesses standard
deviation over the iterations.
'''

plot.figure()
plot.plot(sampled_axis, sampled_bests)
plot.xlabel("Iteration")
plot.ylabel("Best Fitness")
plot.savefig("results/algorithm1-best.png", bbox_inches="tight")

plot.figure()
plot.plot(sampled_axis, sampled_averages)
plot.xlabel("Iteration")
plot.ylabel("Fitness Averages")
plot.savefig("results/algorithm1-averages.png", bbox_inches="tight")

plot.figure()
plot.plot(sampled_axis, sampled_deviations)
plot.xlabel("Iteration")
plot.ylabel("Fitness Standard Deviations")
plot.savefig("results/algorithm1-deviations.png", bbox_inches="tight")
