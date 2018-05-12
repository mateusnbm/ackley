#
# evo-v3.py
#
# The code comments might be inacurate, I'll update and add the description soon.
#


import math
import random
import numpy as np


'''
Parameters.
'''

individuals_count = 30
childs_count = 200

ackley_n = 3#30
ackley_x_min = -15
ackley_x_max =  15

gaussian_deviation_min = 0.001
gaussian_deviation_max = 12

alpha_min = -math.pi
alpha_max =  math.pi

beta_degress = 5

learning_rate = 1/math.sqrt(2*math.sqrt(individuals_count))
learning_rate_l = 1/math.sqrt(2*individuals_count)

max_iteration_count = 1000000


'''
Metrics.
'''

solution = []


'''
The probability density function used by the Correlated Mutation..
'''

def correlated_mutation_pdf(v, co_m):

    v = np.array(v)
    co_m = np.array(co_m)

    a = v.transpose().dot(co_m)
    b = a.dot(v)
    c = math.exp(-0.5*b)

    d = np.linalg.det(co_m)
    e = math.pow((2*math.pi), len(v))
    f = d*e
    g = math.pow(math.fabs(f), 0.5)*np.sign(f)

    h = c/g

    return h


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
    Pass the individual's xi set to the Ackley function.
    '''

    return ackley(individual[0])


'''
Generates a random individual. The individuals are represented as lists of
floats, composed by: the first "ackley_n" numbers are the solution vector, then
the mutation step size (gaussian standard deviation) and the individual fitness.
'''

def generateIndividual():

    g_min = gaussian_deviation_min
    g_max = gaussian_deviation_max

    a_min = alpha_min
    a_max = alpha_max

    x_vals = [random.uniform(ackley_x_min, ackley_x_max) for _ in range(ackley_n)]
    sigma_vals = [random.uniform(g_min, g_max) for _ in range(ackley_n)]
    alpha_vals = [[random.uniform(a_min, a_max) for _ in range(ackley_n)] for __ in range(ackley_n)]

    individual = [x_vals, sigma_vals, alpha_vals, 0]
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

        x_vals = [random.choice(family)[0][k] for k in range(ackley_n)]
        sigma_vals = [(sum(a[1][k] for a in family)/count) for k in range(ackley_n)]
        alpha_vals = [[(sum(a[2][m_i][m_j] for a in family)/count) for m_j in range(ackley_n)] for m_i in range(ackley_n)]

        child = [x_vals, sigma_vals, alpha_vals, 0]
        child[-1] = fitness(child)

        childs.append(child)

    '''
    Mutate childs (Correlated Mutation, Eiben Pag. 60).
    '''

    for child in childs:

        '''
        Update mutation steps (sigmas).
        '''

        const_g = random.gauss(0, 1)

        for i in range(ackley_n):

            g = random.gauss(0, 1)
            a = learning_rate_l*const_g
            b = learning_rate*g

            sigma = child[1][i]
            sigma_l = sigma*math.exp(a*b)

            if sigma_l < gaussian_deviation_min:

                sigma_l = gaussian_deviation_min

            child[1][i] = sigma_l

        '''
        Update rotation angles (alphas).
        '''

        for i in range(ackley_n):

            for j in range(ackley_n):

                g = random.gauss(0, 1)
                alpha = child[2][i][j]
                alpha_l = alpha+(beta_degress*g)

                if (math.fabs(alpha_l) > math.pi):

                    alpha_l = alpha_l-(2*math.pi*np.sign(alpha_l))

                child[2][i][j] = alpha_l

        '''
        Calculate the covariance matrix.
        '''

        co_m = []

        for i in range(ackley_n):

            co_m.append([])

            for j in range(ackley_n):

                if i == j:

                    #print("(" + str(i) + ", " + str(j) + ") = " + str(child[1][i]) + "^2")

                    val = math.pow(child[1][i], 2)

                else:

                    #print("(" + str(i) + ", " + str(j) + ") = " + str(child[1][i]) + "^2")

                    a = math.pow(child[1][i], 2)-math.pow(child[1][j], 2)
                    b = math.tan(2*child[2][i][j])
                    val = 0.5*a*b

                co_m[i].append(val)


        '''
        Update individual objects (x0...xi).
        '''

        for i in range(ackley_n):

            v = [0 for _ in range(ackley_n)]
            x = child[0][i]+correlated_mutation_pdf(v, co_m)

            if ackley_x_min <= x and x <= ackley_x_max:

                child[0][i] = x

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
