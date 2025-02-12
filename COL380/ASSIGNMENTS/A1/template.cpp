#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <omp.h>
#include <set>
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
    set<pair<int, int>> keySet;
    //let us generate b random pairs of i and j
    for (int i = 0; i < b; i++) {
        pair<int, int> key = {dist(gen), dist(gen)};
        //if key in keys, generate another key
        while (keySet.find(key) != keySet.end()) {//inefficient : find(keys.begin(), keys.end(), key) != keys.end()
            key = {dist(gen), dist(gen)};
        }
        keys.push_back(key);
        keySet.insert(key);
    }
    std::uniform_int_distribution<int> dist_block(0, 256);
    //now we generate the blocks parallely using pragma omp task
    #pragma omp parallel if(black_box())//to if with black box
    #pragma omp single //if(black_box())//to if with black box
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
            #pragma omp task shared(matrix_map, keys) if(black_box())//to if with black box
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

vector<float> matmul_parallel_1(map<pair<int, int>, vector<vector<int>>>& blocks, int n, int m, int k) {
    vector<float> row_statistics(n, 0.0f); // For storing S[i] when k=2
    vector<int> P(n, 0);
    vector<int> B(n, 0);
    removeMultiplesOf5_own(blocks);
    //let us first try a naive approach to multiply the matrices k=2 case
    //very basic sequential algorithm
    map<pair<int, int>, vector<vector<int>>> result;
    map<pair<int, int>, vector<vector<int>>> blocks_dash = blocks;
    std::vector<std::pair<int, int>> keys1, keys2;
    for (const auto &entry : blocks_dash) keys1.push_back(entry.first);
    for (const auto &entry : blocks) keys2.push_back(entry.first);
    size_t n1 = keys1.size();
    size_t n2 = keys2.size();
    #pragma omp parallel if(black_box())
    {
        #pragma omp single //if(black_box())
        {
            for (int o = 0; o < k - 1; o++) {
                #pragma omp taskloop collapse(2) shared(blocks_dash, blocks, result, keys1, keys2, n1, n2, m, n, k, P) if(black_box()) //to if with black box
                for (size_t i_loop = 0; i_loop < n1; i_loop++) {
                    for (size_t j_loop = 0; j_loop < n2; j_loop++) {
                        int i = keys1[i_loop].first;
                        int l1 = keys1[i_loop].second;
                        int l2 = keys2[j_loop].first;
                        int j = keys2[j_loop].second;
                        if(l1!=l2){
                            continue;
                        }
                        auto &entry = blocks_dash[keys1[i_loop]];
                        auto &entry2 = blocks[keys2[j_loop]];
                        int l=l1;
                        // Ensure only one thread initializes `result[{i, j}]`
                        //#pragma omp critical
                        std::vector<std::vector<int>>temp_result(m, std::vector<int>(m, 0));
                        for (int x = 0; x < m; x++) {
                            for (int y = 0; y < m; y++) {
                                for (int z = 0; z < m; z++) {
                                    int value = entry[x][z] * entry2[z][y];
                                    //#pragma omp atomic update
                                    temp_result[x][y] += value;
                                    if (k == 2 && value != 0) {
                                        #pragma omp atomic update //if(black_box())
                                        P[i * m + x] += 1;
                                    }
                                }
                            }
                        }
                        // Ensure only one thread writes to `result[{i, j}]`
                        #pragma omp critical //if(black_box())
                        {
                            if (result.find({i, j}) == result.end()) {
                                result[{i, j}] = temp_result;
                            }
                            else {
                                for (int x = 0; x < m; x++) {
                                    for (int y = 0; y < m; y++) {
                                        result[{i, j}][x][y] += temp_result[x][y];
                                    }
                                }
                            }
                        }
                    }
                }
                #pragma omp taskwait //if(black_box())// Ensure all tasks complete before erasing
                blocks_dash = result;
                result.clear();
                keys1.clear();
                keys2.clear();
                for (const auto &entry : blocks_dash) keys1.push_back(entry.first);
                for (const auto &entry : blocks) keys2.push_back(entry.first);
                n1 = keys1.size();
                n2 = keys2.size();
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
    blocks = blocks_dash;
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

vector<float> matmul_multiply(const map<pair<int, int>, vector<vector<int>>>& blocks_dash,map<pair<int, int>, vector<vector<int>>>& blocks, int n, int m, bool stats_needed) {
    vector<float> row_statistics;
    vector<int> P,B;
    if(stats_needed){
        row_statistics=vector<float>(n, 0.0f); // For storing S[i] when k=2
        P=vector<int>(n, 0);
        B=vector<int>(n, 0);
    }
    //let us first try a naive approach to multiply the matrices k=2 case
    //very basic sequential algorithm
    map<pair<int, int>, vector<vector<int>>> result;
    std::vector<std::pair<int, int>> keys1, keys2;
    for (const auto &entry : blocks_dash) keys1.push_back(entry.first);
    for (const auto &entry : blocks) keys2.push_back(entry.first);
    // std::vector<std::pair<int, int>> keys1(blocks_dash.size()), keys2(blocks.size());
    // #pragma omp parallel
    // {
    //     #pragma omp single
    //     {
    //         // Task for extracting keys1
    //         int index = 0;
    //         for (const auto &entry : blocks_dash) {
    //             #pragma omp task firstprivate(index, entry)
    //             keys1[index] = entry.first;
    //             index++;
    //         }

    //         // Task for extracting keys2
    //         index = 0;
    //         for (const auto &entry : blocks) {
    //             #pragma omp task firstprivate(index, entry)
    //             keys2[index] = entry.first;
    //             index++;
    //         }
    //     }
    // }    
    size_t n1 = keys1.size();
    size_t n2 = keys2.size();
    #pragma omp parallel if(black_box())
    {
        #pragma omp single
        {
            #pragma omp taskloop collapse(2) shared(blocks_dash, blocks, result, keys1, keys2, n1, n2, m, n, P, stats_needed) if(black_box())//to if with black box
            for (size_t i_loop = 0; i_loop < n1; i_loop++) {
                for (size_t j_loop = 0; j_loop < n2; j_loop++) {
                    int i = keys1[i_loop].first;
                    int l1 = keys1[i_loop].second;
                    int l2 = keys2[j_loop].first;
                    int j = keys2[j_loop].second;
                    if(l1!=l2){
                        continue;
                    }
                    auto &entry = blocks_dash.at(keys1[i_loop]);
                    auto &entry2 = blocks[keys2[j_loop]];
                    int l=l1;
                    std::vector<std::vector<int>>temp_result(m, std::vector<int>(m, 0));
                    for (int x = 0; x < m; x++) {
                        for (int y = 0; y < m; y++) {
                            for (int z = 0; z < m; z++) {
                                int value = entry[x][z] * entry2[z][y];
                                //#pragma omp atomic update
                                temp_result[x][y] += value;
                                if (stats_needed && value != 0) {
                                    #pragma omp atomic update
                                    P[i * m + x] += 1;
                                }
                            }
                        }
                    }
                    #pragma omp critical
                    {
                        if (result.find({i, j}) == result.end()) {
                            result[{i, j}] = temp_result;
                        }
                        else {
                            for (int x = 0; x < m; x++) {
                                for (int y = 0; y < m; y++) {
                                    result[{i, j}][x][y] += temp_result[x][y];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    blocks = result;// if B[i] is calculated using output
    if (stats_needed){
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
            if (B[i]!=0){
                row_statistics[i] = (float)P[i] / B[i];
            }else{
                row_statistics[i] = 0;
            }
        }
    }
    //blocks = result; if B[i] is calculated using input
    return (stats_needed) ? row_statistics : vector<float>();
}

vector<float> matmul(map<pair<int, int>, vector<vector<int>>>& blocks, int n, int m, int k) {
    //we will implement fast exponentiation
    removeMultiplesOf5_own(blocks);
    if (k==1){
        //do nothing to blocks
        return vector<float>();
    }
    else if (k==2){
        map<pair<int, int>, vector<vector<int>>>left=blocks;
        vector<float> s = matmul_multiply(left,blocks,n,m,true);//blocks got updated
        return s;
    }
    else {
        map<pair<int, int>, vector<vector<int>>>temp,left;
        left=blocks;
        bool temp_is_identity=true;
        while (k!=0)
        {
            if(k%2==1){
                if(temp_is_identity){
                    temp=blocks;
                    temp_is_identity=false;
                }
                else{
                    matmul_multiply(left,temp,n,m,false);//temp got updated
                }
                
            }
            matmul_multiply(left,blocks,n,m,false);//blocks got updated
            k=k/2;
            left=blocks;
        }
        blocks=temp;
    }
    return vector<float>();
}