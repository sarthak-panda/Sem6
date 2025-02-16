#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <filesystem>
#include <string>
#include "check.h"

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

bool black_box() {
    return true;
}

int main() {
    // int n = 100000;
    // int m = 50;
    // int b = 1073;
    // int k = 18;
    int n=9;
    int m=3;
    int b=3;
    int k=2;    
    // int n = 1000;
    // int m = 50;
    // int b = 60;
    // int k = 95;
    bool print_enable=true;

    srand(time(0));
    cout<<"ij"<<endl;

    string bin_filename = "precomputed_result/bin/mat_exp_" + to_string(1) + ".bin";
    map<pair<int, int>, vector<vector<int>>> blocks = load_bin(bin_filename, m);
    cout<<"jiih"<<endl;
    cout<<blocks.size()<<endl;
    vector<float> s = matmul(blocks, n, m, k);

    bool res = check(blocks, k, m);
    if(res)
        cout<<"Your function computed the {"+to_string(k)+"}th power correctly\n";
    else
        cout<<"Your function did NOT compute {"+to_string(k)+"}th power correctly\n";
    cout << "Size of S = " << s.size()<<endl;
    //cout<<s[0]<<endl;
    if (print_enable){
        fs::create_directories("results");
        save_csv(blocks,"results/exp_"+to_string(k)+".csv");
        std::ofstream outfile("output.txt"); // Open file for writing
        if (!outfile) {
            std::cerr << "Error opening file!" << std::endl;
            return 1; // Return error code
        }

        for (const float& num : s) {
            outfile << num << " "; // Write each float separated by space
        }

        outfile.close(); // Close file
        std::cout << "Vector written to output.txt successfully!" << std::endl;        
    }

    return 0;
}