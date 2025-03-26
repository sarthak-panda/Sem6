#include <random>
#include <iostream>
#include "modify.cuh"

#define MODIFY_ON
#define CHECK_ON

void print(vector<vector<int>>& matrix) {
  for (int i = 0; i < matrix.size(); i++) {
    for (int j = 0; j < matrix[0].size(); j++)
      cout << matrix[i][j] << ' ';
  }
  cout << endl;
}

int main() {

  int range{ 1000 }, rows{ 10 }, cols{ 20 };

  vector<vector<vector<int>>> matrices;
  matrices.push_back(gen_matrix(range, rows, cols));
  vector<int> ranges(1, range);
  print(matrices[0]);

#ifdef MODIFY_ON
  vector<vector<vector<int>>> upd_matrices = modify(matrices, ranges);
#endif

print(upd_matrices[0]);

#ifdef CHECK_ON
  if (check(upd_matrices, matrices)) cout << "Test Passed";
  else cout << "Test Failed";
#endif
  return 0;
}
