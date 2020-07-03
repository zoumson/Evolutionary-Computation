# Evolutionary Algorithms  

**Problem to slove: Maximize f (x, y) =80-x^2-y^2+10 cos(2πx) +10 cos(2πy)

					
    
<h2>Method1: Binary Genetic Algorithm or Float Genetic Algorithm</h2>
<pre><ul>
<li>Initialize a population to pop </li>
<li>Repeat the following sections a given number of times, say Generation </li>
<li>Population after Roulette Wheeling Selection </li>
<li>Population after crossover </li>
<li>Population after mutation </li>
<li>Population after picking the best chromosome from the previous population if there's one </li>
<li>Store the best chromosome in current Generation </li>
</ul></pre>
 
<h2>Method2:  Hill Climbing</h2>
<pre><ul>
<li>Generate a random chromosome vc</li>
<li>Repeat the following sections till most fit chromosome found</li>
<li>Find the most fit neighbor of the initial chromosome vn</li>
<li>Check if its more fit than the initial chromosome</li>
<li>If vc is less fit than the most fit neighbor, replace it by the given
neighbor and repeat the steps above</li>
<li>If vc is more fit stop the infinite loop, optimum point found</li>
</ul></pre>

<h2>Method2: Simulated Annealing</h2> 
<pre><ul>
<li>Generate a random chromosome vc</li>
<li>Repeat the following sections till most fit chromosome found</li>
<li>Pick up a random neighbor of vc, say vn</li>
<li>Check if its more fit than the initial chromosome</li>
<li>If vc  is less fit than the random neighbor, replace it by the given
neighbor and repeat the steps above</li>
<li>If vc  is more fit, it still can be replaced by random a selection process</li>
</ul></pre>

**Finding:** *Float Genetic Algorithm is the most efficient as it reaches convergence faster* 
