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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, (n/m)-1);
    vector<pair<int, int>> keys;
    set<pair<int, int>> keySet;
    for (int i = 0; i < b; i++) {
        pair<int, int> key = {dist(gen), dist(gen)};
        while (keySet.find(key) != keySet.end()) {
            key = {dist(gen), dist(gen)};
        }
        keys.push_back(key);
        keySet.insert(key);
    }
    std::uniform_int_distribution<int> dist_block(0, 255);
    #pragma omp parallel if(black_box())
    #pragma omp single
    {
            for (int k = 0; k < b; k++) {
                #pragma omp task shared(matrix_map, keys) firstprivate(k) if(black_box())
                {
                    pair<int, int> key = keys[k];
                    vector<vector<int>> block(m, vector<int>(m, 0));
                    bool nonZeroOccured = false;
                    std::random_device rd;
                    std::mt19937 task_gen(rd());
                    while (!nonZeroOccured) {
                        for (int i = 0; i < m; i++) {
                            for (int j = 0; j < m; j++) {
                                block[i][j] = dist_block(task_gen);
                                if (block[i][j] != 0) {
                                    nonZeroOccured = true;
                                }
                            }
                        }
                    }
                    #pragma omp critical
                    {
                        matrix_map[key] = block;
                    }
                }
            }
            #pragma omp taskwait
    }
    return matrix_map;
}

vector<float> matmul_serial(map<pair<int, int>, vector<vector<int>>>& blocks, int n, int m, int k) {
    vector<float> row_statistics(n, 0.0f);
    vector<int> P(n, 0);
    vector<int> B(n, 0);
    removeMultiplesOf5_own(blocks);
    map<pair<int, int>, vector<vector<int>>> result;
    map<pair<int, int>, vector<vector<int>>> blocks_dash = blocks;
    for (int o=0;o<k-1;o++){
        for (int i = 0; i < n/m; i++) {
            for (int j = 0; j < n/m; j++) {
                result[{i, j}]=std::vector<std::vector<int>>(m, std::vector<int>(m, 0));
                bool nonZeroOccured = false;
                for (int l = 0; l < n/m; l++) {
                    if (blocks_dash.find({i, l}) == blocks_dash.end() || blocks.find({l, j}) == blocks.end()) {
                        continue;
                    }
                    nonZeroOccured = true;
                    for (int x = 0; x < m; x++) {
                        for (int y = 0; y < m; y++) {
                            for (int z = 0; z < m; z++) {
                                int value=blocks_dash[{i, l}][x][z] * blocks[{l, j}][z][y];
                                result[{i, j}][x][y] += value;
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
        blocks_dash = result;
    }
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
    return (k == 2) ? row_statistics : vector<float>();
}

vector<float> matmul_multiply(const map<pair<int, int>, vector<vector<int>>>& blocks_dash,map<pair<int, int>, vector<vector<int>>>& blocks, int n, int m, bool stats_needed) {
    vector<float> row_statistics;
    vector<int> P,B;
    if(stats_needed){
        row_statistics=vector<float>(n, 0.0f);
        P=vector<int>(n, 0);
        B=vector<int>(n, 0);
    }
    map<pair<int, int>, vector<vector<int>>> result;
    std::vector<std::pair<int, int>> keys1, keys2;
    for (const auto &entry : blocks_dash) keys1.push_back(entry.first);
    for (const auto &entry : blocks) keys2.push_back(entry.first);

    size_t n1 = keys1.size();
    size_t n2 = keys2.size();
    #pragma omp parallel if(black_box())
    {
        #pragma omp single
        {
            #pragma omp taskloop collapse(2) shared(blocks_dash, blocks, result, keys1, keys2, n1, n2, m, n, P, stats_needed) if(black_box())
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
    blocks = result;
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
    return (stats_needed) ? row_statistics : vector<float>();
}

bool has_non_zero_element_own(vector<vector<int>>& block) {
    for (auto& row : block)
        for (int val : row)
            if (val != 0)
                return true;
    return false;
}

void remove_zero_blocks(map<pair<int, int>, vector<vector<int>>>& blocks) {
    std::vector<std::pair<int, int>> keys_to_erase;
    #pragma omp parallel
    #pragma omp single
    {
        for (auto& entry : blocks) {
            #pragma omp task firstprivate(entry) shared(keys_to_erase)
            {
                vector<vector<int>>& block = entry.second;
                if (!has_non_zero_element_own(block))
                    #pragma omp critical
                    keys_to_erase.push_back(entry.first);
            }
        }
    }
    for (auto& key : keys_to_erase) {
        blocks.erase(key);
    }
    return;
}

vector<float> matmul(map<pair<int, int>, vector<vector<int>>>& blocks, int n, int m, int k) {
    removeMultiplesOf5_own(blocks);
    if (k==1){
        return vector<float>();
    }
    else if (k==2){
        map<pair<int, int>, vector<vector<int>>>left=blocks;
        vector<float> s = matmul_multiply(left,blocks,n,m,true);
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
                    matmul_multiply(left,temp,n,m,false);
                }
                
            }
            matmul_multiply(left,blocks,n,m,false);
            k=k/2;
            left=blocks;
        }
        blocks=temp;
        remove_zero_blocks(blocks);
    }
    return vector<float>();
}