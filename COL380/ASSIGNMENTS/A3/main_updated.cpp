#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>  // for std::fixed, std::setprecision
#include "modify.cuh"

#define MODIFY_ON
#define CHECK_ON

using namespace std;

void print(const vector<vector<int>>& matrix) {
    for (int i = 0; i < (int)matrix.size(); i++) {
        for (int j = 0; j < (int)matrix[i].size(); j++) {
            cout << matrix[i][j] << ' ';
        }
        cout << endl;
    }
    cout << endl;
}

int main() {
    vector< tuple<int,int,int> > testParams = {
        // r=10^3, c=10^3, M=1
        make_tuple(1000, 1000, 1),
        // r=10^3, c=10^3, M=10
        make_tuple(1000, 1000, 10),
        // r=10^3, c=10^3, M=100 
        //make_tuple(1000, 1000, 100),
        // r=10^4, c=10^4, M=1
        make_tuple(10000, 10000, 1),
        // r=10^4, c=10^4, M=10
        make_tuple(10000, 10000, 10),
		// r=10^4, c=10^5, M=1
		make_tuple(10000, 100000, 1)
    };

    // 4 columns of maxValue
    vector<long long> maxValues = { 1024, 4096, 100000, 100000000 };

    // We'll store times in a 2D array: times[row][col]
    // row -> which (r,c,M) triple
    // col -> which maxValue
    // We'll store -1.0 if test fails.
    vector<vector<double>> times(testParams.size(), vector<double>(maxValues.size(), -1.0));

    bool allPassed = true;  // track if all tests pass

    // For CSV output, we might keep the same order:
    // Rows:  (10^3 x 10^3, M=1), (10^3 x 10^3, M=10), (10^3 x 10^3, M=100),
    //        (10^4 x 10^4, M=1), (10^4 x 10^4, M=10)
    // Cols:  maxValue in [1024, 4096, 10^5, 10^8]

    // We'll loop over each row and column:
    for (int i = 0; i < (int)testParams.size(); i++) {
        int rows, cols, M;
        tie(rows, cols, M) = testParams[i];

        for (int j = 0; j < (int)maxValues.size(); j++) {
            
            long long maxVal = maxValues[j];
            if(rows==10000 && cols==10000 && maxVal==100000000) continue;
            cout << "\n-------------------------------------------------\n";
            cout << "Testcase " << (i*maxValues.size() + j + 1) << " of 19\n";
            cout << "Matrix size = " << rows << " x " << cols 
                 << ", |M| = " << M 
                 << ", maxValue = " << maxVal << "\n";

            // 1) Generate M matrices
            vector<vector<vector<int>>> matrices;
            matrices.reserve(M);

            // We create a random engine for generating matrix elements
            try {
                for(int m = 0; m < M; m++){
                    matrices.push_back(gen_matrix(maxVal, rows, cols));
                }
            }
            catch (std::bad_alloc&) {
                cerr << "Memory allocation failed for " 
                     << rows << "x" << cols << " with M=" << M 
                     << ". Skipping this test.\n";
                allPassed = false;
                continue;  // proceed to next test
            }

            // 2) Build ranges array, each = maxVal
            vector<int> ranges(M, (int)maxVal);

            // 3) Call modify(), measure time
            using clock_type = std::chrono::high_resolution_clock;
            auto start = clock_type::now();

            vector<vector<vector<int>>> upd_matrices;
            try {
                upd_matrices = modify(matrices, ranges);
            } catch (...) {
                cerr << "modify() threw an exception for this test.\n";
                allPassed = false;
                continue; 
            }

            auto end = clock_type::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            double elapsedMs = elapsed.count();

            // 4) check() to confirm correctness
            bool pass = false;
            try {
                pass = check(upd_matrices, matrices);
            } catch (...) {
                cerr << "check() threw an exception for this test.\n";
                allPassed = false;
                continue;
            }

            if (pass) {
                // Store time (rounded to 2 decimals if you like, but store raw for CSV).
                times[i][j] = elapsedMs;
                cout << "Test Passed in " 
                     << fixed << setprecision(2) << elapsedMs 
                     << " ms\n";
            } else {
                cout << "Test Failed\n";
                allPassed = false;
            }
        }
    }

    // If all 19 testcases passed, write times to data.csv in the table order
    if (allPassed) {
        cout << "\nAll 19 tests passed. Writing times to data.csv...\n";

        ofstream fout("data.csv");
        if (!fout) {
            cerr << "Could not open data.csv for writing.\n";
            return 0;
        }

        // We match the table:
        //    r x c, |M|, then 4 columns for maxValue times
        //    We'll just do a row per testParams[i].
        // You can format it as you wish; here's a simple approach:

        // First, print a header row in CSV:
        fout << "r x c,|M|,1024,4096,10^5,10^8\n";

        // Then each row:
        for (int i = 0; i < (int)testParams.size(); i++) {
            int rows, cols, M;
            tie(rows, cols, M) = testParams[i];

            fout << "(" << rows << "x" << cols << ")," << M << ",";

            for (int j = 0; j < (int)maxValues.size(); j++) {
                fout << fixed << setprecision(2) << times[i][j];
                if (j+1 < (int)maxValues.size()) fout << ",";
            }
            fout << "\n";
        }

        fout.close();
        cout << "CSV file data.csv has been created.\n";
    } else {
        cout << "\nNot all tests passed, so no CSV is generated.\n";
    }

    return 0;
}
