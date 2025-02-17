Optimizations Performed:

(1)Matrix Generation
-> Utilized OpenMP tasks to parallelize the generation of matrix blocks, enabling multiple blocks to be built concurrently.
-> "b" unique block positions were generated efficiently and managed using a set data structure, ensuring that no duplicate block assignments occur.
(2)Matrix Multiplication
-> Utilized OpenMP's taskloop to parallelize the multiplication of matrix blocks, assigning each pairwise block multiplication to a separate task for concurrent execution.
-> Changed the loop order for multiplying each pair of blocks from IJK to IKJ. Although the IKJ order had previously yielded the best performance in our last assignment, 
the improvement was minimal in this case—likely due to the small block sizes in a sparse matrix.
-> Instead of exponentiating by sequentially multiplying matrix A k times, we applied fast exponentiation to reduce the complexity of the exponentiation process to O(log k).

Explanation of the Fast Exponentiation Algorithm:

This algorithm implements fast matrix exponentiation using binary decomposition. It starts by initializing an accumulator (acc) as an identity-like value 
and setting the base matrix (curr) to the input matrix. 
In each loop iteration, if the current exponent  k is odd, the algorithm multiplies the accumulator(acc) with the current matrix(curr).
It then squares the current matrix (updating curr) and halves k (i.e. updating k to floor of k/2), thus reducing the overall number of multiplications to 
O(log k) and efficiently computing the exponentiation.