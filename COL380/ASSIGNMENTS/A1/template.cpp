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

//yused from checker script recheck
void removeMultiplesOf5_own(map<pair<int, int>, vector<vector<int>>>& matrixBlocks) {
    for (auto it = matrixBlocks.begin(); it != matrixBlocks.end(); ) {
        vector<vector<int>>& block = it->second;
        bool isBlockNonZero = false;

        for (auto& row : block) {
            for (auto& value : row) {
                if (value % 5 == 0) {
                    value = 0;
                }
                if (value != 0) {
                    isBlockNonZero = true;
                }
            }
        }

        if (!isBlockNonZero) {
            it = matrixBlocks.erase(it);
        } else {
            ++it;
        }
    }
}

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
    std::uniform_int_distribution<int> dist(0, (n/m)-1);
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
                bool nonZeroOccured = false;
                while (!nonZeroOccured){
                    for (int i = 0; i < m; i++){
                        for (int j = 0; j < m; j++){
                            block[i][j] = dist_block(gen);
                            if (block[i][j] != 0) {
                                nonZeroOccured = true;
                            }
                        }
                    }
                }
                matrix_map[key] = block;
            }
        }
    }
    return matrix_map;
}

vector<float> matmul_serial(map<pair<int, int>, vector<vector<int>>>& blocks, int n, int m, int k) {
    vector<float> row_statistics(n, 0.0f); // For storing S[i] when k=2
    vector<int> P(n, 0);
    vector<int> B(n, 0);
    removeMultiplesOf5_own(blocks);
    //let us first try a naive approach to multiply the matrices k=2 case
    //very basic sequential algorithm
    map<pair<int, int>, vector<vector<int>>> result;
    map<pair<int, int>, vector<vector<int>>> blocks_dash = blocks;
    for (int o=0;o<k-1;o++){
        //A^k = A^(k-1) * A
        for (int i = 0; i < n/m; i++) {
            for (int j = 0; j < n/m; j++) {
                result[{i, j}]=std::vector<std::vector<int>>(m, std::vector<int>(m, 0));
                bool nonZeroOccured = false;
                for (int l = 0; l < n/m; l++) {
                    //block multiplication
                    //if either of block not present, continue
                    if (blocks_dash.find({i, l}) == blocks_dash.end() || blocks.find({l, j}) == blocks.end()) {
                        continue;
                    }
                    //ice al entries are positive , so we can assume if two non zero blocks are multiplied, the result is non zero
                    nonZeroOccured = true;
                    for (int x = 0; x < m; x++) {
                        for (int y = 0; y < m; y++) {
                            for (int z = 0; z < m; z++) {
                                int value=blocks_dash[{i, l}][x][z] * blocks[{l, j}][z][y];
                                result[{i, j}][x][y] += value;
                                // cout<<x<<" "<<y<<" "<<z<<" "<<value<<endl;
                                // cout<<value<<endl;
                                //if k=2, we need to calculate row statistics for each row
                                if (k == 2 && value!=0) {
                                    P[i*m + x] += 1;
                                }
                            }
                        }
                    }
                }
                if (!nonZeroOccured) {
                    result.erase({i, j});
                }
            }
        }
        //copy result to blocks_dash
        blocks_dash = result;
    }
    //let us see how we can genralize the above code for any k to compute A^k
    //also note for k>2 we do not need to calculate row statistics
    //let us see how can we calculate row statistics
    if (k==2){
        for (int i = 0; i < n/m; i++) {
            for (int j = 0; j < n/m; j++) {
                if (blocks.find({i, j}) == blocks.end()) {
                    continue;
                }
                for (int x = 0; x < m; x++) {
                    B[i*m + x] += m;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            row_statistics[i] = (float)P[i] / B[i];
        }
    }
    blocks = result;
    //print blocks
    // for (auto& entry : blocks) {
    //     cout << "Block (" << entry.first.first << ", " << entry.first.second << "):\n";
    //     for (auto& row : entry.second) {
    //         for (int val : row) {
    //             cout << val << " ";
    //         }
    //         cout << "\n";
    //     }
    // }
    return (k == 2) ? row_statistics : vector<float>();
}

vector<float> matmul(map<pair<int, int>, vector<vector<int>>>& blocks, int n, int m, int k) {
    vector<float> row_statistics(n, 0.0f); // For storing S[i] when k=2
    vector<int> P(n, 0);
    vector<int> B(n, 0);
    removeMultiplesOf5_own(blocks);
    //let us first try a naive approach to multiply the matrices k=2 case
    //very basic sequential algorithm
    map<pair<int, int>, vector<vector<int>>> result;
    map<pair<int, int>, vector<vector<int>>> blocks_dash = blocks;
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int o = 0; o < k - 1; o++) {
                std::unordered_set<std::pair<int, int>> toErase;  // Store keys to erase later
                
                #pragma omp taskloop collapse(2) shared(blocks_dash, blocks, result, P, B, toErase)
                for (int i = 0; i < n / m; i++) {
                    for (int j = 0; j < n / m; j++) {
                        bool nonZeroOccured = false;
    
                        // Ensure only one thread initializes `result[{i, j}]`
                        #pragma omp critical
                        result[{i, j}] = std::vector<std::vector<int>>(m, std::vector<int>(m, 0));
    
                        for (int l = 0; l < n / m; l++) {
                            if (blocks_dash.find({i, l}) == blocks_dash.end() || blocks.find({l, j}) == blocks.end()) {
                                continue;
                            }
    
                            nonZeroOccured = true;
    
                            for (int x = 0; x < m; x++) {
                                for (int y = 0; y < m; y++) {
                                    for (int z = 0; z < m; z++) {
                                        int value = blocks_dash[{i, l}][x][z] * blocks[{l, j}][z][y];
    
                                        #pragma omp atomic update
                                        result[{i, j}][x][y] += value;
    
                                        if (k == 2 && value != 0) {
                                            #pragma omp atomic update
                                            P[i * m + x] += 1;
                                        }
                                    }
                                }
                            }
                        }
    
                        if (!nonZeroOccured) {
                            #pragma omp critical
                            toErase.insert({i, j});
                        }
                    }
                }
    
                #pragma omp taskwait // Ensure all tasks complete before erasing
    
                // Remove empty blocks outside the parallel region
                for (auto &key : toErase) {
                    result.erase(key);
                }
    
                // Copy result to blocks_dash (Ensure single-thread execution)
                #pragma omp single
                blocks_dash = result;
            }
        }
    }
        
    //let us see how we can genralize the above code for any k to compute A^k
    //also note for k>2 we do not need to calculate row statistics
    //let us see how can we calculate row statistics
    if (k==2){
        for (int i = 0; i < n/m; i++) {
            for (int j = 0; j < n/m; j++) {
                if (blocks.find({i, j}) == blocks.end()) {
                    continue;
                }
                for (int x = 0; x < m; x++) {
                    B[i*m + x] += m;
                }
            }
        }
        for (int i = 0; i < n; i++) {
            row_statistics[i] = (float)P[i] / B[i];
        }
    }
    blocks = result;
    //print blocks
    // for (auto& entry : blocks) {
    //     cout << "Block (" << entry.first.first << ", " << entry.first.second << "):\n";
    //     for (auto& row : entry.second) {
    //         for (int val : row) {
    //             cout << val << " ";
    //         }
    //         cout << "\n";
    //     }
    // }
    return (k == 2) ? row_statistics : vector<float>();
}
