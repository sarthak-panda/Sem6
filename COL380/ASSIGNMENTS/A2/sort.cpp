#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

void my_sort(vector<pair<int,int>>& nodes, int mode, int k) {
    auto comp = [](const pair<int,int>& a, const pair<int,int>& b) {
        return (a.second == b.second) ? (a.first < b.first) : (a.second > b.second);
    };
    switch(mode) {
        case 1:
            sort(nodes.begin(), nodes.end(), comp);
            break;
        case 2:
            if(nodes.size() > (size_t)k) {
                nth_element(nodes.begin(), nodes.begin() + k, nodes.end(), comp);
                sort(nodes.begin(), nodes.begin() + k, comp);
            } else {
                sort(nodes.begin(), nodes.end(), comp);
            }
            break;
        case 3:
            if(nodes.size() > (size_t)k) {
                partial_sort(nodes.begin(), nodes.begin() + k, nodes.end(), comp);
            } else {
                sort(nodes.begin(), nodes.end(), comp);
            }
            break;
        default:
            sort(nodes.begin(), nodes.end(), comp);
            break;
    }
}

void print_vector(const vector<pair<int,int>>& nodes) {
    for (const auto& p : nodes)
        cout << "{" << p.first << "," << p.second << "} ";
    cout << "\n";
}

int main() {
    // Test Case 1 – Mode 1 (Full Sort)
    vector<pair<int,int>> nodes1 = { {5,2}, {3,3}, {4,2}, {2,3}, {1,1} };
    cout << "Test Case 1 - Mode 1 (Full Sort):\n";
    my_sort(nodes1, 1, 3);
    print_vector(nodes1);

    // Test Case 2 – Mode 2 (n <= k: Full Sort)
    vector<pair<int,int>> nodes2 = { {7,1}, {2,2}, {3,2}, {5,3} };
    cout << "\nTest Case 2 - Mode 2 (n <= k, so full sort):\n";
    my_sort(nodes2, 2, 5);
    print_vector(nodes2);

    // Test Case 3 – Mode 2 (n > k: Partial Sort + Full Sort of Top k)
    vector<pair<int,int>> nodes3 = { {10,5}, {6,3}, {8,5}, {4,2}, {7,4}, {2,3} };
    cout << "\nTest Case 3 - Mode 2 (n > k, partial sort for top k):\n";
    my_sort(nodes3, 2, 3);
    print_vector(nodes3);
    // Note: Only the first k elements are fully sorted. The rest may be in unspecified order.

    // Test Case 4 – Mode 3 (n <= k: Full Sort)
    vector<pair<int,int>> nodes4 = { {9,9}, {3,8}, {5,8} };
    cout << "\nTest Case 4 - Mode 3 (n <= k, so full sort):\n";
    my_sort(nodes4, 3, 3);
    print_vector(nodes4);

    // Test Case 5 – Mode 3 (n > k: Partial Sort)
    vector<pair<int,int>> nodes5 = { {1,4}, {2,6}, {3,5}, {4,6}, {0,7}, {10,5}, {6,3}, {8,5}, {4,2}, {7,4}, {2,3} };
    cout << "\nTest Case 5 - Mode 2 (n > k, partial sort for top k):\n";
    my_sort(nodes5, 2, 5);
    print_vector(nodes5);
    // Note: The first k elements are sorted, the remainder are in unspecified order.

    // Test Case 6 – Mode 1 (Full Sort with duplicates)
    vector<pair<int,int>> nodes6 = { {4,4}, {4,4}, {2,4}, {5,5} };
    cout << "\nTest Case 6 - Mode 1 (Full Sort with duplicate values):\n";
    my_sort(nodes6, 1, 2); // k is ignored in mode 1.
    print_vector(nodes6);

    return 0;
}
