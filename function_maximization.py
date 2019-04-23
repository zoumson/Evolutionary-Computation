#!/usr/bin/env python
__author__ = "Adama ZOUMA"
__copyright__ = "Copyright 2019, NCKU (EPS) Lab"
__email__ = "stargue49@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
"""
User Selection Interface
"""
userInput = '''

Please Select an Algorithm to minimize the function below:

f(x,y) = 80-x^2-y^2+10cos⁡(2πx)+10cos⁡(2πy)

1.Binary Genetic Algorithm
2.Float Genetic Algorithm
3.HillClimbing
4.Simulated Annealing

'''
outRange = userInput + '''
Choice should be in range[1~4]

'''
number = userInput +'''
Choice should be a number

'''
bGA = fGA = hill = sim = False
repeat = True
while repeat:
    repeat = False
    option = input(userInput)
    if option.isdigit():
        option = int(option)
        if option < 1 or option > 4:
            userInput = outRange
            repeat = True
        else:
            if option == 1:
                bGA = True
            elif option == 2:
                fGA = True
            elif option == 3:
                hill = True
            elif option == 4:
                sim = True
    else :
        userInput = number
        repeat = True

"""
End of User Selection
"""

if bGA :

    """
    Binary Genetic Algorithm to maximize f(x,y)=80-x^2-y^2+10cos⁡(2πx)+10cos⁡(2πy)
    """

    """
    Keep the same seed to get the same result as I do
    """
    for k in range(20, 25):
        np.random.seed(k)
        """
        Parameters setting
        """
        lChr = 20
        lGene = lChr//2
        domain = [-0.5, 1.5]
        crossProb = 0.25
        mutProb = 0.01
        popSize = 20
        popChr = popSize*lChr
        popCrossSize = int(popSize*crossProb)
        popMutSize = int(popChr*mutProb)

        def resizeCrosSize(popCrossSize):
            """
            chromosome number for crossover is an even number for pair matching
            """
            if popCrossSize%2 != 0:
                popCrossSize += 1
            return popCrossSize

        popCrossSize = resizeCrosSize(popCrossSize)
        nPair = popCrossSize//2

        """
        Functions definition
        """
        def chromosome(lchr):
            """
            A random binary string of length lchr
            """
            return np.random.randint(0,2, size = lChr)

        def genotype(chr):
            """
            Split a given binary string into two pieces, one leading to x, other to y
            """
            gene1 = chr[:lGene]
            gene2 = chr[lGene:]
            genotyp = np.array([gene1, gene2])
            return genotyp

        def binaryToDec(gene):
            """
            Convert a given binary string to a floating number
            """
            dec = 0
            for index in range(len(gene)):
                dec += gene[-1-index]*(2**index)
            return dec

        def allele(gene):
            """
            Convert a given piece of binary string to the real decimal number
            in range of the domain
            Precison is one decimal number
            """
            xprime = binaryToDec(gene)
            lb = domain[0]
            ub = domain[1]
            domainLength = ub-lb
            x = lb + (xprime*domainLength)/(2**lGene -1)
            x = int(x * 10)/10.0
            return x

        def phenotype(genotyp):
            """
            Convert a given genotype to the corresponding phenotype
            """
            gene1, gene2 = genotyp[0], genotyp[1]
            allele1 = allele(gene1)
            allele2 = allele(gene2)
            phenotyp = np.array([allele1, allele2])
            return phenotyp

        def fitness(phenotyp):
            """
            From x and y value Calculate f(x,y)=80-x^2-y^2+10cos⁡(2πx)+10cos⁡(2πy)
            """
            x, y = phenotyp
            fit = 80 - x**2 -y**2 +10*(np.cos(2*(np.pi)*x))+10*(np.cos(2*(np.pi)*y))
            return fit

        def chrToFitness(chr):
            """
            Calculate the fitness of a given binary string directly
            """
            genotyp = genotype(chr)
            phenotyp = phenotype(genotyp)
            return fitness(phenotyp)

        def crossover(father,mother, crossPoint):
            """
            Generate children from parents by doing crossover
            """
            child1 = np.concatenate([father[:crossPoint],mother[crossPoint:]])
            child2 = np.concatenate([mother[:crossPoint],father[crossPoint:]])
            children = (child1, child2)
            return children

        def flipBit(bit):
            """
            flip a bit, 0 replaced by 1 or 1 replaced by 0
            """
            if bit == 0:
                bit = 1
            else:
                bit = 0
            return np.array([bit])

        def mutation(chr, mutPoint):
            """
            flip one bit of a given binary string at position mutPoint
            """
            mutChr = chr.copy()
            mutChr[mutPoint]=flipBit(mutChr[mutPoint])
            return mutChr

        def population():
            """
            Generate popSize  chromosomes to undergo crossover, mutation, ...
            """
            pop = [chromosome(lChr) for individual in range(popSize)]
            pop = np.array(pop[:])
            return pop

        def popFitness(pop):
            """
            Calculate the fitness of whole population of chromosomes
            """
            popFit = [chrToFitness(pop[individual]) for individual in range(popSize)]
            popFit =  np.array(popFit[:])
            return popFit

        def popTotFitness(pop):
            """
            Calculate the total fitness of whole population of chromosomes
            """
            popFit = popFitness(pop)
            popTotFit = popFit.sum()
            return popTotFit

        def indRelFitness(individual, pop):
            """
            Calculate the relative individual fitness of a given chromosome in a population
            """
            popFit = popFitness(pop)
            indFit = popFit[individual]
            popTotFit = popTotFitness(pop)
            indRelFit = indFit/popTotFit
            return indRelFit

        def popRelFitness(pop):
            """
            Calculate the relative individual fitness of all chromosomes in a population
            """
            popRelFit = [indRelFitness(individual, pop) for individual in range(popSize)]
            popRelFit =  np.array(popRelFit[:])
            return popRelFit

        def popCumFitness(pop):
            """
            Calculate the cumulative  individual fitness of all chromosomes in a population
            """
            popRelFit =  popRelFitness(pop)
            cumFit = np.add.accumulate(popRelFit)
            return cumFit

        def indRoulWheel(pop):
            """
            Select one chromosome from one Roulette wheel Selection process
            """
            rnd = np.random.uniform(0,1)
            popCumFit = popCumFitness(pop)
            if rnd <= popCumFit[0]:
                select = [0]
            else:
                select = np.argwhere(popCumFit<=rnd)
                select = select[-1]
            return select[0]

        def popRoulWheel(pop):
            """
            Population after popSize Roulette wheel Selections
            """
            popAfRo = []
            popNew = np.copy(pop)
            indexx = [indRoulWheel(popNew) for individual in range(popSize)]
            indexx = list(indexx)
            for i in indexx:
                popAfRo.append(popNew[i])
            popAfRo = np.array(popAfRo[:])
            return popAfRo

        def rndSelection(rang, siz, prob):
            """
            Create siz distinct random number in the range rang
            Prob can be wether mutation wether crossover probability
            """
            rndIndex = []
            count = 0
            while True:
                rnd = np.random.randint(0, rang)
                while rnd in rndIndex:
                            rnd = np.random.randint(0, rang)
                else:
                    if np.random.uniform(0,1) < prob:
                        count += 1
                        rndIndex.append(rnd)
                if count == siz:
                    break
            return rndIndex

        def popAfMut(pop):
            """
            Population after Mutation
            """
            popNew = np.copy(pop)
            popAllChr = popNew.flatten()
            mutPoints = rndSelection(popChr, popMutSize, mutProb)
            for mutPoint in mutPoints:
                popAllChr = mutation(popAllChr, mutPoint)
            popMutDone = popAllChr.reshape(popSize, lChr)
            return popMutDone

        def crossPair():
            """
            Finding pair of chromosomes to be crossed
            """
            cPair = []
            fPair = []
            for i in range(nPair):
                rndCross1 = np.random.randint(0, popCrossSize)
                rndCross2 = np.random.randint(0, popCrossSize)
                while rndCross1 == rndCross2 or {rndCross1,rndCross2} in cPair:
                    rndCross1 = np.random.randint(0, popCrossSize)
                    rndCross2 = np.random.randint(0, popCrossSize)
                cPair.append({rndCross1,rndCross2})
                fPair.append([rndCross1,rndCross2])
            return fPair

        def crossMemb(pop):
            """
            Index of chromosomes to be crossed matching with the corresponding binary string
            """
            crosPair = crossPair()
            crossMem = []
            for i in range(nPair):
                cross = [pop[crosPair[i][0]], pop[crosPair[i][1]]]
                crossMem.append(cross)
            crossMem = np.array(crossMem)
            return crossMem

        def popAfCross(pop):
            """
            Population after crossover
            """
            popNew = np.copy(pop)
            crossMem = crossMemb(pop)
            popToCros = crossMem.flatten()
            popToCros = popToCros.reshape(popCrossSize, lChr)
            children = []
            crosPoint = np.random.randint(0, lChr, size = nPair)
            for i in range(nPair):
                father =crossMem[i][0]
                mother =crossMem[i][1]
                babies = crossover(father,mother, crosPoint[i])
                children.append(babies)
            children = np.array(children)
            children = children.flatten()
            children = children.reshape(popCrossSize, lChr)
        ## Replace parents by children
            for i in range(popSize):
                for j in range(popCrossSize):
                    if np.all(popNew[i]==popToCros[j]):
                        popNew[i]= children[j]
                        break
            return popNew

        def elitist(pPop, cPo):
            """
            keep the best chromosome in current population compared to previous population
            """
            cPop = np.copy(cPo)
            pBest = np.argwhere(popFitness(pPop)==max(popFitness(pPop)))
            pBest = pBest[0]
            pBest = pPop[pBest]
            pBest = pBest.flatten()
            cBest = cPop[np.argwhere(popFitness(cPop)==max(popFitness(cPop)))]
            cBest = cBest.flatten()
            iWorst = np.argwhere(popFitness(cPop)==min(popFitness(cPop)))
            pFit = chrToFitness(pBest)
            cFit = chrToFitness(cBest)
            if pFit>cFit:
                cPop[iWorst]=pBest
            return cPop

        def findBest(bpop):
            """
            Find best chromosome in a given population
            """
            fPop = popFitness(bpop)
            eq = fPop==max(fPop)
            maxFit = fPop[eq]
            maxFit = maxFit[0]
            argMaxFit = np.argwhere(fPop==maxFit)
            argMaxFit = argMaxFit[0][0]
            fBest = bpop[argMaxFit]
            return fBest

        """
        Binary Genetic Algorithm Starts here
        Sequence is as follow
        -Initialize a population to pop
        Repeat the following sections a given number of times, say Generation
        -Population after Roulette Wheeling Selection
        -Population after crossover
        -Population after mutation
        -Population after picking the best chromosome from the prevous population if there's one
        -Store the best chromosome in current Generation
        """
        Generation = 20
        best = []
        for generatio in range(Generation):
            if generatio == 0:
                pop = population()
                r = popRoulWheel(pop)
                c = popAfCross(r)
                m = popAfMut(c)
                e = elitist(pop, m)
                fPop = popFitness(e)
                bestGeneration = findBest(e)
                best.append(bestGeneration)
            else:
                r = popRoulWheel(e)
                c = popAfCross(r)
                m = popAfMut(c)
                e = elitist(e, m)
                fPop= popFitness(e)
                bestGeneration = findBest(e)
                best.append(bestGeneration)

        best = np.array(best[:])
        bestGenerationFit = popFitness(best)
        bestChr = findBest(best)
        bestGenotyp = genotype(bestChr)
        bestPenotyp = phenotype(bestGenotyp)
        bestFit =fitness(bestPenotyp)

        def binaryGA():
            """
            Display on termial screen
            """
            run = k+1
            print("\n\n\nRun number: ", run)
            print("\nMaximize f(x,y)=80-x^2-y^2+10cos⁡(2πx)+10cos⁡(2πy)")
            print("\nMethod used: Binary Genetic Algorithm")
            print('\nMaximum Value of function f is', bestFit)
            print('\nMaximum Point x =',bestPenotyp[0], 'and y =',bestPenotyp[1])
            plt.figure(run)
            plt.plot(bestGenerationFit, linewidth = 5)
            plt.title("Run number: %i\nBinary Genetic Algorithm" % run, color = 'red', fontsize = 25)
            plt.xlabel('Generation', fontsize=15)
            plt.ylabel('Fitness', fontsize=15)
            plt.xticks(range(Generation))
            plt.yticks(bestGenerationFit)
            plt.show()

        if __name__ == '__main__':
            binaryGA()

"""
End of Binary Genetic Algorithm
"""
if fGA:

    """
    Float Genetic Algorithm to maximize f(x,y)=80-x^2-y^2+10 cos⁡(2πx)+10 cos⁡(2πy)
    """

    """
    Keep the same seed to get the same result as I do
    """

    for k in [5, 8, 9, 10]:
        np.random.seed(k)
        """
        Parameters setting
        """
        lChr = 2
        lGene = lChr//2
        domain = [-0.5, 1.5]
        crossProb = 0.25
        mutProb = 0.01
        popSize = 20
        popChr = popSize*lChr
        popCrossSize = int(popSize*crossProb)
        popMutSize = int(popChr*mutProb)
        popChr = popSize*lChr

        """
        chromosome number for crossover is an even number for pair matching
        """
        def resizeCrosSize(popCrossSize):
            if popCrossSize%2 != 0:
                popCrossSize += 1
            return popCrossSize

        popCrossSize = resizeCrosSize(popCrossSize)
        nPair = popCrossSize//2

        """
        Functions definition
        """
        def chromosome():
            """
            Two random float numbers in range of domain
            One decimal number precison
            """
            x = np.random.uniform(domain[0], domain[1], size = 1)
            x = int(x * 10)/10.0
            y = np.random.uniform(domain[0], domain[1], size = 1)
            y = int(y * 10)/10.0
            chr = np.array([x, y])
            return chr

        def fitness(chr):
            """
            From x and y value caculate f(x,y)=80-x^2-y^2+10cos⁡(2πx)+10cos⁡(2πy)
            """
            x = chr[0]
            y = chr[1]
            fit = 80 - x**2 -y**2 +10*(np.cos(2*(np.pi)*x))+10*(np.cos(2*(np.pi)*y))
            return fit

        def population():
            """
            Generate popSize  chromosomes to undergo crossover, mutation, ...
            """
            pop = [chromosome() for individual in range(popSize)]
            pop = np.array(pop[:])
            return pop

        def popFitness(pop):
            """
            Calculate the fitness of whole population of chromosomes
            """
            popFit = [fitness(pop[individual]) for individual in range(popSize)]
            popFit =  np.array(popFit[:])
            return popFit

        def popTotFitness(pop):
            """
            Calculate the total fitness of whole population of chromosomes
            """
            popFit = popFitness(pop)
            popTotFit = popFit.sum()
            return popTotFit

        def indRelFitness(individual, pop):
            """
            Calculate the relative individual fitness of a given chromosome in a population
            """
            popFit = popFitness(pop)
            indFit = popFit[individual]
            popTotFit = popTotFitness(pop)
            indRelFit = indFit/popTotFit
            return indRelFit

        def popRelFitness(pop):
            """
            Calculate the relative individual fitness of all chromosomes in a population
            """
            popRelFit = [indRelFitness(individual, pop) for individual in range(popSize)]
            popRelFit =  np.array(popRelFit[:])
            return popRelFit

        def popCumFitness(pop):
            """
            Calculate the cumulative  individual fitness of all chromosomes in a population
            """
            popRelFit =  popRelFitness(pop)
            cumFit = np.add.accumulate(popRelFit)
            return cumFit

        def indRoulWheel(pop):
            """
            Select one chromosome from one Roulette wheel Selection process
            """
            rnd = np.random.uniform(0,1)
            popCumFit = popCumFitness(pop)
            if rnd <= popCumFit[0]:
                select = [0]
            else:
                select = np.argwhere(popCumFit<=rnd)
                select = select[-1]
            return select[0]

        def popRoulWheel(pop):
            """
            population after popSize Roulette wheel Selection process
            """
            popAfRo = []
            popNew = np.copy(pop)
            indexx = [indRoulWheel(popNew) for individual in range(popSize)]
            indexx = list(indexx)
            for i in indexx:
                popAfRo.append(popNew[i])
            popAfRo = np.array(popAfRo[:])
            return popAfRo

        def rndSelection(rang, siz, prob):
            """
            Create siz distinct random number in the range rang
            """
            rndIndex = []
            count = 0
            while True:
                rnd = np.random.randint(0, rang)
                while rnd in rndIndex:
                            rnd = np.random.randint(0, rang)
                else:
                    if np.random.uniform(0,1) < prob:
                        count += 1
                        rndIndex.append(rnd)
                if count == siz:
                    break
            return rndIndex

        def mutation(chr, mutPoint):
            """
            change the bit of a given float string at position mutPoint
            """
            mutchr = chr.copy()
            rnd = np.random.uniform(domain[0], domain[1], size = 1)
            rnd = int(rnd * 10)/10.0
            mutchr[mutPoint]=rnd
            return mutchr

        def popAfMut(pop):
            """
            Population after Mutation
            """
            popNew = np.copy(pop)
            popAllChr = popNew.flatten()
            mutPoints = rndSelection(popChr, popMutSize, mutProb)
            for mutPoint in mutPoints:
                popAllChr = mutation(popAllChr, mutPoint)
            popMutDone = popAllChr.reshape(popSize, lChr)
            return popMutDone

        def crossover(father,mother, crossPoint):
            """
            Generate children from parents by doing crossover
            """
            crossPoint += 1
            child1 = np.concatenate([father[:crossPoint],mother[crossPoint:]])
            child2 = np.concatenate([mother[:crossPoint],father[crossPoint:]])
            children = (child1, child2)
            return children

        def crossPair():
            """
            Finding pair of chromosome to be crossed
            """
            cPair = []
            fPair = []
            for i in range(nPair):
                rndCross1 = np.random.randint(0, popCrossSize)
                rndCross2 = np.random.randint(0, popCrossSize)
                while rndCross1 == rndCross2 or {rndCross1,rndCross2} in cPair:
                    rndCross1 = np.random.randint(0, popCrossSize)
                    rndCross2 = np.random.randint(0, popCrossSize)
                cPair.append({rndCross1,rndCross2})
                fPair.append([rndCross1,rndCross2])
            return fPair

        def crossMemb(pop):
            """
            Index of chromosomes to be crossed matching with the corresponding binary string
            """
            crosPair = crossPair()
            crossMem = []
            for i in range(nPair):
                cross = [pop[crosPair[i][0]], pop[crosPair[i][1]]]
                crossMem.append(cross)
            crossMem = np.array(crossMem)
            return crossMem

        def popAfCross(pop):
            """
            Population after crossover
            """
            popNew = np.copy(pop)
            crossMem = crossMemb(pop)
            popToCros = crossMem.flatten()
            popToCros = popToCros.reshape(popCrossSize, lChr)
            children = []
            crosPoint = np.random.randint(0, lChr, size = nPair)
            for i in range(nPair):
                father =crossMem[i][0]
                mother =crossMem[i][1]
                babies = crossover(father,mother, crosPoint[i])
                children.append(babies)
            children = np.array(children)
            children = children.flatten()
            children = children.reshape(popCrossSize, lChr)
        #Replace parents by children
            for i in range(popSize):
                for j in range(popCrossSize):
                    if np.all(popNew[i]==popToCros[j]):
                        popNew[i]= children[j]
                        break
            return popNew

        def elitist(pPop, cPo):
            """
            keep the best chromosome in current population compared to previous population
            """
            cPop = np.copy(cPo)
            pBest = np.argwhere(popFitness(pPop)==max(popFitness(pPop)))
            pBest = pBest[0][0]
            pBest = pPop[pBest]
            cBest = np.argwhere(popFitness(cPop)==max(popFitness(cPop)))
            cBest = cPop[cBest]
            cBest = cBest[0][0]
            iWorst = np.argwhere(popFitness(cPop)==min(popFitness(cPop)))
            pFit = fitness(pBest)
            cFit = fitness(cBest)
            if pFit>cFit:
                cPop[iWorst]=pBest
            return cPop

        def findBest(bpop):
            """
            Find best chromosome in a given population
            """
            fPop = popFitness(bpop)
            eq = fPop==max(fPop)
            maxFit = fPop[eq]
            maxFit = maxFit[0]
            argMaxFit = np.argwhere(fPop==maxFit)
            argMaxFit = argMaxFit[0][0]
            fBest = bpop[argMaxFit]
            return fBest

        """
        Float Genetic Algorithm Starts here
        Sequence is as follow
        -Initialize a population to pop
        Repeat the following sections a given number of times, say Generation
        -Population after Roulette Wheeling Selection
        -Population after crossover
        -Population after mutation
        -Population after picking the best chromosome from the prevous population if there's one
        -Store the best chromosome in current Generation
        """

        Generation = 20
        best = []

        for i in range(Generation):
            if i == 0:

                pop = population()
                r = popRoulWheel(pop)
                c = popAfCross(r)
                m = popAfMut(c)
                e = elitist(pop, m)
                fPop = popFitness(e)
                bb = findBest(e)
                best.append(bb)
            else:
                r = popRoulWheel(e)
                c = popAfCross(r)
                m = popAfMut(c)
                e = elitist(e, m)
                fPop= popFitness(e)
                bb = findBest(e)
                best.append(bb)

        best = np.array(best[:])
        bestGenerationFit = popFitness(best)
        bestChr = findBest(best)
        bestFit =fitness(bestChr)

        def floatGA():
            run = k+1
            """
            Display on termial screen
            """
            print("\n\n\nRun number: ", run)
            print("\nMaximize f(x,y)=80-x^2-y^2+10cos⁡(2πx)+10cos⁡(2πy)")
            print("\nMethod used: Float Genetic Algorithm")
            print('\nMaximum Value of function f is', bestFit)
            print('\nMaximum Point x =',bestChr[0], 'and y =',bestChr[1])

            plt.figure(run)
            plt.plot(bestGenerationFit, linewidth = 5)
            plt.title("Run number: %i\nFloat Genetic Algorithm" % run, color = 'red', fontsize = 25)
            plt.xlabel('Generation', fontsize=15)
            plt.ylabel('Fitness', fontsize=15)
            plt.xticks(range(Generation))
            plt.yticks(bestGenerationFit)
            plt.show()

        if __name__ == '__main__':
            floatGA()
"""
End of Float Genetic Algorithm
"""
if hill:

    """
    HillClimging to maximize f(x,y)=80-x^2-y^2+10 cos⁡(2πx)+10 cos⁡(2πy)
    """

    """
    Keep the same seed to get the same result as I do
    """
    for k in range(5):
        np.random.seed(k)
        """
        Parameters setting
        """
        lChr = 2
        lGene = lChr//2
        domain = [-0.5, 1.5]

        """
        Functions definition
        """
        def chromosome():
            """
            Two random float numbers in range of domain
            One decimal number precison
            """
            x = np.random.uniform(domain[0], domain[1], size = 1)
            x = int(x * 10)/10.0
            y = np.random.uniform(domain[0], domain[1], size = 1)
            y = int(y * 10)/10.0
            chr = np.array([x, y])
            return chr

        def fitness(chr):
            """
            From x and y value caculate f(x,y)=80-x^2-y^2+10 cos⁡(2πx)+10 cos⁡(2πy)
            """
            x = chr[0]
            y = chr[1]
            fit = 80 - x**2 -y**2 +10*(np.cos(2*(np.pi)*x))+10*(np.cos(2*(np.pi)*y))
            return fit

        def mutation(chr, mutPoint):
            """
            change the bit of a given float string at position mutPoint
            """
            mutchr = chr.copy()
            rnd = np.random.uniform(domain[0], domain[1], size = 1)
            rnd = int(rnd * 10)/10.0
            mutchr[mutPoint]=rnd
            return mutchr

        def population():
            """
            Generate popSize  chromosomes to undergo crossover, mutation, ...
            """
            pop = [chromosome(lChr) for individual in range(popSize)]
            pop = np.array(pop[:])
            return pop

        def rndSelection(rang, siz, prob):
            """
            Create siz distinct random number in the range rang
            """
            rndIndex = []
            count = 0
            while True:
                rnd = np.random.randint(0, rang)
                while rnd in rndIndex:
                            rnd = np.random.randint(0, rang)
                else:
                    if np.random.uniform(0,1) < prob:
                        count += 1
                        rndIndex.append(rnd)
                if count == siz:
                    break
            return rndIndex

        def popFitness(pop):

            """
            Calculate the fitness of a given whole population of chromosomes
            """
            popFit = [fitness(pop[individual]) for individual in range(lChr)]
            popFit =  np.array(popFit[:])
            m = np.argwhere(popFit == max(popFit))
            m = m[0]
            fmaxi = popFit[m]
            maxi = pop[m]
            return fmaxi, maxi

        def nearest(chr):
            """
            Find all neigbours of a given chromosome
            """
            vn = [mutation(vc, mutPoint) for mutPoint in range(lChr)]
            vn =  np.array(vn[:])
            return vn


        """
        HillClimging starts here
        Sequence is as follow
        Generate a random chromosome vc
        Repeat the following sections till most fit chromosome found
        -Find the most fit neigbour of the initial chromosome vn
        -Check if its more fit than the initial chromosome
        -If vc is less fit than the most fit neigbour, replace it by the given
        neigbour and repeat the steps above
        -If vc is more fit stop the infinite loop, optimum point found
        """

        vc = chromosome()

        count = 0
        while True:
            count += 1
            fc = fitness(vc)
            v = nearest(vc)
            f = popFitness(v)
            fn = f[0][0]
            vn = f[1][0]
            if fc < fn:
                vc = vn

            else:
                break

        bestChr = vc

        def hillClimbing():
            """
            Display on termial screen
            """
            print("\n\n\nRun number: ", k+1)
            print("\nMaximize f(x,y)=80-x^2-y^2+10cos⁡(2πx)+10cos⁡(2πy)\n")
            print("Method used: HillClimging")
            print('\nMaximum Point x =',bestChr[0], 'and y =',bestChr[1])
            print("\nMaximum function value is f(x =", bestChr[0], ", y = " ,bestChr[1],") =", fc, "\n")
            print("Maximum found after", count, "iterations\n")

        if __name__ == '__main__':
            hillClimbing()
"""
End of HillClimbing
"""
if sim:
    """
    Simulated Annealing to maximize f(x,y)=80-x^2-y^2+10 cos⁡(2πx)+10 cos⁡(2πy)
    """
    """
    Keep the same seed to get the same result as I do
    """
    for k in range(80, 85):
        np.random.seed(k)
        """
        Parameters setting
        """
        lChr = 2
        lGene = lChr//2
        domain = [-0.5, 1.5]
        T = 20
        """
        Functions definition
        """

        def chromosome():
            """
            Two random float numbers in range of domain
            One decimal number precison
            """
            x = np.random.uniform(domain[0], domain[1], size = 1)
            x = int(x * 10)/10.0
            y = np.random.uniform(domain[0], domain[1], size = 1)
            y = int(y * 10)/10.0
            chr = np.array([x, y])
            return chr

        def fitness(chr):
            """
            From x and y value caculate f(x,y)=80-x^2-y^2+10 cos⁡(2πx)+10 cos⁡(2πy)
            """
            x = chr[0]
            y = chr[1]
            fit = 80 - x**2 -y**2 +10*(np.cos(2*(np.pi)*x))+10*(np.cos(2*(np.pi)*y))
            return fit

        def mutation(chr, mutPoint):
            """
            change the bit of a given float string
            """
            mutchr = chr.copy()
            rnd = np.random.uniform(domain[0], domain[1], size = 1)
            rnd = int(rnd * 10)/10.0
            mutchr[mutPoint]=rnd
            return mutchr

        def population():
            """
            Generate popSize  chromosomes to undergo crossover, mutation, ...
            """
            pop = [chromosome() for individual in range(popSize)]
            pop = np.array(pop[:])
            return pop

        def rndSelection(rang, siz, prob):
            """
            Create siz distinct random number in the range rang
            """
            rndIndex = []
            count = 0
            while True:
                rnd = np.random.randint(0, rang)
                while rnd in rndIndex:
                            rnd = np.random.randint(0, rang)
                else:
                    if np.random.uniform(0,1) < prob:
                        count += 1
                        rndIndex.append(rnd)
                if count == siz:
                    break
            return rndIndex

        def popFitness(pop):
            popFit = [chrToFitness(pop[individual]) for individual in range(lChr)]
            popFit =  np.array(popFit[:])
            m = np.argwhere(popFit == max(popFit))
            m = m[0]
            fmaxi = popFit[m]
            maxi = pop[m]
            return fmaxi, maxi

        def nearest(chr):
            vn = [mutation(vc, mutPoint) for mutPoint in range(lChr)]
            vn =  np.array(vn[:])
            return vn

        def prob(f1, f2):
            return np.exp((f1-f2)/T)

        """
        Simulated Annealing starts here
        Sequence is as follow
        Generate a random chromosome vc
        Repeat the following sections till most fit chromosome found
        -Pick up a random neigbour of vc, say vn
        -Check if its more fit than the initial chromosome
        -If vc is less fit than the most fit neigbour, replace it by the given
        neigbour and repeat the steps above
        -If vc is more fit, its still can be replaced by a selection process
        -Generate a random number rnd and compare it to exp((fvn-fvc)/T)
        -If rnd < exp((fvn-fvc)/T) replace vc by vn
        -If rnd > exp((fvn-fvc)/T) process ends, vc is the maximum
        """

        vc = chromosome()
        count = 0
        while True:
            count += 1
            fc = fitness(vc)
            mutPoint = np.random.randint(0, lChr)
            vn = mutation(vc, mutPoint)
            fn = fitness(vn)
            if fc < fn:
                vc = vn
            elif np.random.uniform(0, 1)<prob(fn, fc):
                vc = vn
            else :
                break

        bestChr = vc

        def simAnnealing():
            """
            Display on termial screen
            """
            print("\n\n\nRun number: ", k+1)
            print("\nMaximize f(x,y)=80-x^2-y^2+10cos⁡(2πx)+10cos⁡(2πy)\n")
            print("Method used: Simulated Annealing")
            print('\nMaximum Point x =',bestChr[0], 'and y =',bestChr[1])
            print("\nMaximum function value is f(x =", bestChr[0], ", y = " ,bestChr[1],") =", fc, "\n")
            print("Maximum found after", count, "iterations\n")

        if __name__ == '__main__':
            simAnnealing()
"""
End of Simulated Annealing
"""
