#!/bin/bash
# Exit immediately if any command exits with a non-zero status.
set -e

# Execute the commands one after the other

cd  ~/COL380/A2
module load compiler/gcc/9.1.0
module load compiler/gcc/9.1/mpich/3.3.1

mpirun -np 8 ./check0 TestCase1 OUTPUT_4/O0/T1/output0.txt 10
mpirun -np 8 ./check0 GenTestCase OUTPUT_4/O0/T2/output0.txt 100
mpirun -np 8 ./check0 EvenLargerTestCase OUTPUT_4/O0/T3/output0.txt 50
mpirun -np 8 ./check0 GRAPH_tough OUTPUT_4/O0/T4/output0.txt 55

mpirun -np 8 ./check1 TestCase1 OUTPUT_4/O1/T1/output0.txt 10
mpirun -np 8 ./check1 GenTestCase OUTPUT_4/O1/T2/output0.txt 100
mpirun -np 8 ./check1 EvenLargerTestCase OUTPUT_4/O1/T3/output0.txt 50
mpirun -np 8 ./check1 GRAPH_tough OUTPUT_4/O1/T4/output0.txt 55

mpirun -np 8 ./check2 TestCase1 OUTPUT_4/O2/T1/output0.txt 10
mpirun -np 8 ./check2 GenTestCase OUTPUT_4/O2/T2/output0.txt 100
mpirun -np 8 ./check2 EvenLargerTestCase OUTPUT_4/O2/T3/output0.txt 50
mpirun -np 8 ./check2 GRAPH_tough OUTPUT_4/O2/T4/output0.txt 55

mpirun -np 8 ./check3 TestCase1 OUTPUT_4/O3/T1/output0.txt 10
mpirun -np 8 ./check3 GenTestCase OUTPUT_4/O3/T2/output0.txt 100
mpirun -np 8 ./check3 EvenLargerTestCase OUTPUT_4/O3/T3/output0.txt 50
mpirun -np 8 ./check3 GRAPH_tough OUTPUT_4/O3/T4/output0.txt 55

mpirun -np 8 ./check0 SmallTestCase OUTPUT_4/O0/T0/output0.txt 10
mpirun -np 8 ./check1 SmallTestCase OUTPUT_4/O1/T0/output0.txt 10
mpirun -np 8 ./check2 SmallTestCase OUTPUT_4/O2/T0/output0.txt 10
mpirun -np 8 ./check3 SmallTestCase OUTPUT_4/O3/T0/output0.txt 10

mpirun -np 8 ./check0 TestCase1 OUTPUT_4_NEW/O0/T1/output0.txt 97
mpirun -np 8 ./check0 GenTestCase OUTPUT_4_NEW/O0/T2/output0.txt 257
mpirun -np 8 ./check0 EvenLargerTestCase OUTPUT_4_NEW/O0/T3/output0.txt 999
mpirun -np 8 ./check0 GRAPH_tough OUTPUT_4_NEW/O0/T4/output0.txt 1548

mpirun -np 8 ./check1 TestCase1 OUTPUT_4_NEW/O1/T1/output0.txt 97
mpirun -np 8 ./check1 GenTestCase OUTPUT_4_NEW/O1/T2/output0.txt 257
mpirun -np 8 ./check1 EvenLargerTestCase OUTPUT_4_NEW/O1/T3/output0.txt 999
mpirun -np 8 ./check1 GRAPH_tough OUTPUT_4_NEW/O1/T4/output0.txt 1548

mpirun -np 8 ./check2 TestCase1 OUTPUT_4_NEW/O2/T1/output0.txt 97
mpirun -np 8 ./check2 GenTestCase OUTPUT_4_NEW/O2/T2/output0.txt 257
mpirun -np 8 ./check2 EvenLargerTestCase OUTPUT_4_NEW/O2/T3/output0.txt 999
mpirun -np 8 ./check2 GRAPH_tough OUTPUT_4_NEW/O2/T4/output0.txt 1548

mpirun -np 8 ./check3 TestCase1 OUTPUT_4_NEW/O3/T1/output0.txt 97
mpirun -np 8 ./check3 GenTestCase OUTPUT_4_NEW/O3/T2/output0.txt 257
mpirun -np 8 ./check3 EvenLargerTestCase OUTPUT_4_NEW/O3/T3/output0.txt 999
mpirun -np 8 ./check3 GRAPH_tough OUTPUT_4_NEW/O3/T4/output0.txt 1548

mpirun -np 8 ./check0 SmallTestCase OUTPUT_4_NEW/O0/T0/output0.txt 15
mpirun -np 8 ./check1 SmallTestCase OUTPUT_4_NEW/O1/T0/output0.txt 15
mpirun -np 8 ./check2 SmallTestCase OUTPUT_4_NEW/O2/T0/output0.txt 15
mpirun -np 8 ./check3 SmallTestCase OUTPUT_4_NEW/O3/T0/output0.txt 15