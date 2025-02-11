#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <filesystem>
#include <string>

using namespace std;
namespace fs = std::filesystem;
// Function to load a matrix from a binary file
map<pair<int, int>, vector<vector<int>>> load_bin(const string& filename, int m) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error opening binary file: " << filename << endl;
        return {};
    }

    map<pair<int, int>, vector<vector<int>>> matrix;
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));

    for (size_t i = 0; i < size; ++i) {
        pair<int, int> key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));

        vector<vector<int>> block(m, vector<int>(m));
        for (auto& row : block) {
            file.read(reinterpret_cast<char*>(row.data()), m * sizeof(int));
        }

        matrix[key] = block;
    }

    file.close();
    return matrix;
}

// Function to check if matrix2 matches the precomputed k-th power
bool check(map<pair<int, int>, vector<vector<int>>>& matrix2, int k, int m) {
    string bin_filename = "precomputed_result/bin/mat_exp_" + to_string(k) + ".bin";
    map<pair<int, int>, vector<vector<int>>> loaded_matrix = load_bin(bin_filename, m);

    if (loaded_matrix.size() != matrix2.size()) {
        cout << "Size mismatch\n";
        return false;
    }

    for (auto& [block_pos, block] : loaded_matrix) {
        if (matrix2.find(block_pos) == matrix2.end()) {
            cout << "Mismatch in blocks\n";
            return false;
        }

        vector<vector<int>>& block2 = matrix2.at(block_pos);
        if (block != block2) {
            cout << "Matrix values mismatch\n";
            return false;
        }
    }

    return true;
}