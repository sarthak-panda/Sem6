#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <filesystem>
#include <string>
#include "check.h"

using namespace std;
namespace fs = std::filesystem;

void removeMultiplesOf5(map<pair<int, int>, vector<vector<int>>>& matrixBlocks) {
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

vector<vector<int>> multiply_blocks(vector<vector<int>>& block1,
                                    vector<vector<int>>& block2, int m) {
    vector<vector<int>> product(m, vector<int>(m, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < m; ++k) {
                product[i][j] += block1[i][k] * block2[k][j];
            }
        }
    }
    return product;
}

// Function to multiply two matrices
map<pair<int, int>, vector<vector<int>>> multiply_matrix(
    map<pair<int, int>, vector<vector<int>>>& mat1,
    map<pair<int, int>, vector<vector<int>>>& mat2, int m) {

    map<pair<int, int>, vector<vector<int>>> result;

    for (auto& [block_pos1, block1] : mat1) {
        for (auto& [block_pos2, block2] : mat2) {
            if (block_pos1.second == block_pos2.first) {
                vector<vector<int>> product = multiply_blocks(block1, block2, m);
                auto key = make_pair(block_pos1.first, block_pos2.second);

                if (result.find(key) != result.end()) {
                    vector<vector<int>>& existing_block = result[key];
                    for (int i = 0; i < m; ++i) {
                        for (int j = 0; j < m; ++j) {
                            existing_block[i][j] += product[i][j];
                        }
                    }
                } else {
                    result[key] = product;
                }
            }
        }
    }
    return result;
}

// Function to save matrix as a CSV file
void save_csv(const map<pair<int, int>, vector<vector<int>>>& matrix, const string& filename) {
    ofstream file(filename);
    if (!file) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    for (auto& entry : matrix) {
        file << "Block (" << entry.first.first << ", " << entry.first.second << "):\n";
        for (auto& row : entry.second) {
            for (int val : row) {
                file << val << " ";
            }
            file << "\n";
        }
    }
    file.close();
}

// Function to save matrix as a binary file
void save_bin(const map<pair<int, int>, vector<vector<int>>>& matrix, const string& filename) {
    ofstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error opening binary file: " << filename << endl;
        return;
    }

    size_t size = matrix.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));

    for (const auto& [key, block] : matrix) {
        file.write(reinterpret_cast<const char*>(&key), sizeof(key));
        for (const auto& row : block) {
            file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(int));
        }
    }

    file.close();
}

// Function to generate powers of matrix1 up to "limit"
void generate(map<pair<int, int>, vector<vector<int>>>& matrix1, int limit, int m) {
    removeMultiplesOf5(matrix1);

    fs::create_directories("precomputed_result/bin");
    fs::create_directories("precomputed_result/csv");

    map<pair<int, int>, vector<vector<int>>> power_result = matrix1;

    for (int exp = 1; exp <= limit; ++exp) {
        string bin_filename = "precomputed_result/bin/mat_exp_" + to_string(exp) + ".bin";
        string csv_filename = "precomputed_result/csv/mat_exp_" + to_string(exp) + ".csv";

        save_bin(power_result, bin_filename);
        save_csv(power_result, csv_filename);

        if (exp < limit) {
            power_result = multiply_matrix(power_result, matrix1, m);
        }
    }
}

bool has_non_zero_element(vector<vector<int>>& block) {
    for (auto& row : block)
        for (int val : row)
            if (val != 0)
                return true;
    return false;
}

int count_non_zero_blocks(map<pair<int, int>, vector<vector<int>>>& blocks) {
    int non_zero_count = 0;

    for (auto& entry : blocks) {
        vector<vector<int>>& block = entry.second;

        if (has_non_zero_element(block))
            non_zero_count++;
    }

    return non_zero_count;
}

bool black_box() {
    return true;
}

int main() {
    // int n = 1000;
    // int m = 50;
    // int b = 60;
    // int k = 2;
    // int n=9;
    // int m=3;
    // int b=3;
    // int k=2;
    int n = 100000;
    int m = 8;
    int b = 8095;
    int k = 2;

    srand(time(0));
    map<pair<int, int>, vector<vector<int>>> my_blocks;
    map<pair<int, int>, vector<vector<int>>> blocks = generate_matrix(n, m, b);
    // my_blocks[{0, 0}] = {{1, 0, 2}, {0, 3, 0}, {2, 0, 4}};
    // my_blocks[{0, 2}] = {{0, 0, 0}, {0, 3, 1}, {0, 0, 5}};
    // my_blocks[{2, 0}] = {{0, 0, 0}, {0, 3, 0}, {0, 1, 5}};
    // blocks=my_blocks;    
    cout<<blocks.size()<<endl;
    if(count_non_zero_blocks(blocks)==blocks.size() && blocks.size()>=b)
        cout<<"You have generated the matrix correctly\n";
    else
        cout<<"You have NOT generated the matrix correctly\n";
	cout<<"Precomputation Started..."<<endl;
	generate(blocks,20,m);
	cout<<"Precomputation Ended..."<<endl;
    return 0;
}