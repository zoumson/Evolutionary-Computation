Maximize f (x, y) =80-x^2-y^2+10 cos(2πx) +10 cos(2πy)

					
    
Method1: Binary Genetic Algorithm or Float Genetic Algorithm 
 Sequence is as follow
    -Initialize a population to pop
    Repeat the following sections a given number of times, say Generation
    -Population after Roulette Wheeling Selection
    -Population after crossover
    -Population after mutation
    -Population after picking the best chromosome from the previous population if there's one
-Store the best chromosome in current Generation

Method2:  Hill Climbing

Sequence is as follow
Generate a random chromosome v_c
Repeat the following sections till most fit chromosome found
-Find the most fit neighbor of the initial chromosome v_n
-Check if its more fit than the initial chromosome
-If v_c is less fit than the most fit neighbor, replace it by the given
neighbor and repeat the steps above
-If v_c is more fit stop the infinite loop, optimum point found

Method2: Simulated Annealing 

Sequence is as follow
Generate a random chromosome v_c
Repeat the following sections till most fit chromosome found
-Pick up a random neighbor of v_c, say v_n
-Check if its more fit than the initial chromosome
-If v_c  is less fit than the random neighbor, replace it by the given
neighbor and repeat the steps above
-If v_c  is more fit, it still can be replaced by a selection process
-Generate a random number rnd and compare it to e^((〖Fit〗_(v_n )-〖Fit〗_(v_c ))/T)
-If rnd < e^((〖Fit〗_(v_n )-〖Fit〗_(v_c ))/T) replace v_c by v_n
-If rnd > e^(e^((〖Fit〗_(v_n )-〖Fit〗_(v_c ))/T) ) process ends, v_c is the maximum

Comparison: Float Genetic Algorithm is the most efficient as it reaches convergence faster 
