#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <omp.h>
#include "check.h"

using namespace std;

map<pair<int, int>, vector<vector<int>>> generate_matrix(int n, int m, int b) {
    map<pair<int, int>, vector<vector<int>>> matrix_map;
    // 0<=i<n/m and 0<=j<n/m we need to b number of blocks of size m*m
    // for that b random pairs of i and j are generated in given range
    // for each pair generated we parallely generate a block of size m*m and store it in matrix_map[(i, j)] as vector<vector<int32>>
    // the value of each element of the block is randomly generated in the range 0 to 256
    // open mp task construct is used to parallelize the generation of blocks
    // we can assume m divides n , b<<(n/m)^2 as sparse matrix are being generated
    // Whenever we create an OpenMP task, make sure to add an if(black box()) clause to the task pragma. The black box() function assumed to be in check.h
    // example usage : #pragma omp task shared(A, B, C) depend(in:B) if(black_box()) mult_block(A, B, C);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, n/m);
    vector<pair<int, int>> keys;
    //let us generate b random pairs of i and j
    for (int i = 0; i < b; i++) {
        pair<int, int> key = {dist(gen), dist(gen)};
        //if key in keys, generate another key
        while (find(keys.begin(), keys.end(), key) != keys.end()) {
            key = {dist(gen), dist(gen)};
        }
        keys.push_back(key);
    }
    std::uniform_int_distribution<int> dist_block(0, 256);
    //now we generate the blocks parallely using pragma omp task
    #pragma omp parallel//to if with black box
    #pragma omp single//to if with black box
    {
        // #pragma omp taskloop shared(matrix_map, keys)
        // for (int i=0; i<b; i++){
        //     pair<int, int> key = keys[i];
        //     //let us first create a block then later parallelize the block generation
        //     vector<vector<int>> block(m, vector<int>(m, 0));
        //     for (int i=0; i<m; i++){
        //         for (int j=0; j<m; j++){
        //             block[i][j] = dist_block(gen);
        //         }
        //     }
        //     matrix_map[key] = block;
        // }
        for (int k = 0; k < b; k++) {
            #pragma omp task shared(matrix_map, keys)//to if with black box
            {
                pair<int, int> key = keys[k];
                vector<vector<int>> block(m, vector<int>(m, 0));
                for (int i = 0; i < m; i++){
                    for (int j = 0; j < m; j++){
                        block[i][j] = dist_block(gen);
                    }
                }
                matrix_map[key] = block;
            }
        }
    }
    return matrix_map;
}

vector<float> matmul(map<pair<int, int>, vector<vector<int>>>& blocks, int n, int m, int k) {
    vector<float> row_statistics(n, 0.0f); // For storing S[i] when k=2
    //let us first try a naive approach to multiply the matrices k=2 case
    //very basic sequential algorithm
    map<pair<int, int>, vector<vector<int>>> result;
    //A*A
    for (int i = 0; i < n/m; i++) {
        for (int j = 0; j < n/m; j++) {
            for (int l = 0; l < n/m; l++) {
                //block multiplication
                //if either of block not present, continue
                if (blocks.find({i, l}) == blocks.end() || blocks.find({l, j}) == blocks.end()) {
                    continue;
                }
                for (int x = 0; x < m; x++) {
                    for (int y = 0; y < m; y++) {
                        for (int z = 0; z < m; z++) {
                            result[{i, j}][x][y] += blocks[{i, l}][x][z] * blocks[{l, j}][z][y];
                        }
                    }
                }
            }
        }
    }
    //if k=2, we need to calculate row statistics
    return (k == 2) ? row_statistics : vector<float>();
}