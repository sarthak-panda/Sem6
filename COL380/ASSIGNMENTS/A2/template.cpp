#include "template.hpp"
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <mpi.h>
#include <unordered_set>
// #include <utility>
// #include <functional>

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

void init_mpi(int argc, char* argv[]) {
    //Code Here
    MPI_Init(&argc, &argv);
}

void end_mpi() {
    //Code Here
    MPI_Finalize();
}

vector<vector<int>> degree_cen(vector<pair<int, int>>& partial_edge_list, map<int, int>& partial_vertex_color, int k) {
    //To check if i get same rank and size as in check.cpp
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes
    //code to generate the complete color map
    cout<<"Rank: "<<rank<<" Size: "<<size<<endl;
    vector<int> local_buff;
    local_buff.reserve(partial_vertex_color.size() * 2);
    for(auto e: partial_vertex_color){
        local_buff.push_back(e.first);
        local_buff.push_back(e.second);
    }
    int local_buff_size = local_buff.size();
    vector<int> recv_counts(size);
    MPI_Allgather(&local_buff_size,1,MPI_INT,recv_counts.data(),1,MPI_INT,MPI_COMM_WORLD);
    vector<int> displacements(size);
    displacements[0] = 0;
    for(int i = 1; i < size; i++){
        displacements[i] = displacements[i-1] + recv_counts[i-1];
    }
    vector<int> global_buff(displacements[size-1] + recv_counts[size-1]);
    MPI_Allgatherv(local_buff.data(),local_buff_size,MPI_INT,global_buff.data(),recv_counts.data(),displacements.data(),MPI_INT,MPI_COMM_WORLD);
    map<int,int> vertex_color;
    set<int> colors;
    for(int i = 0; i < global_buff.size(); i+=2){
        vertex_color[global_buff[i]] = global_buff[i+1];
        if(rank == 0){
            colors.insert(global_buff[i+1]);
        }
    }
    //work that each thread will do after getting the complete color map
    unordered_map<int,unordered_map<int,int>> partial_color_vertex_map;
    for(auto e: partial_edge_list){
        int u = e.first;
        int v = e.second;
        int color_u = vertex_color[u];
        int color_v = vertex_color[v];
        partial_color_vertex_map[color_u][v]++;
        partial_color_vertex_map[color_v][u]++;
    }
    size_t total_local_results = 0;
    for (const auto &entry : partial_color_vertex_map) {
        total_local_results += 2; // One for the color and one for the marker (-9)
        total_local_results += entry.second.size() * 2; // Each neighbor adds two integers.
    }
    vector<int> local_results;
    local_results.reserve(total_local_results);
    for(auto &e: partial_color_vertex_map){
        int color= e.first;
        local_results.push_back(color);
        int color_idenrifier = -9;
        local_results.push_back(color_idenrifier);
        for(auto &v: e.second){
            local_results.push_back(v.first);
            local_results.push_back(v.second);
        }
    }
    int local_results_size = local_results.size();
    vector<int> recv_counts_results;
    if(rank == 0){
        recv_counts_results.resize(size);
    }
    MPI_Gather(&local_results_size,1,MPI_INT,recv_counts_results.data(),1,MPI_INT,0,MPI_COMM_WORLD);
    vector<int> displacements_results;
    vector<int> global_results;
    if(rank == 0){
        displacements_results.resize(size);
        displacements_results[0] = 0;
        for(int i = 1; i < size; i++){
            displacements_results[i] = displacements_results[i-1] + recv_counts_results[i-1];
        }
        global_results.resize(displacements_results[size-1] + recv_counts_results[size-1]);
    }
    MPI_Gatherv(local_results.data(),local_results_size,MPI_INT,global_results.data(),recv_counts_results.data(),displacements_results.data(),MPI_INT,0,MPI_COMM_WORLD);
    vector<vector<int>>output;
    if(rank == 0){
        unordered_map<int,map<int,int>> color_vertex_map;
        int color=-1;
        for(int i = 0; i < global_results.size(); i+=2){
            int vertex = global_results[i];
            int partial_deg = global_results[i+1];
            if(partial_deg == -9){
                color = vertex;
                continue;
            }
            color_vertex_map[color][vertex] += partial_deg;
        }
        // set<int> colors;
        // for(auto e: color_vertex_map){
        //     colors.insert(e.first);
        // }
        for(auto color: colors){
            vector<int>top_k_nodes;
            top_k_nodes.reserve(k);
            if(color_vertex_map.find(color) == color_vertex_map.end()){
                //push top k nodes from 0 to k-1 consequetively assuming nodes labelled from 0 to n-1
                for(int i = 0; i < k; i++){
                    top_k_nodes.push_back(i);
                }
            }
            else{
                vector<pair<int, int>> nodes;
                nodes.reserve(color_vertex_map[color].size());
                for(auto &e: color_vertex_map[color]){
                    nodes.push_back({e.first, e.second});
                }
                // sort(nodes.begin(),nodes.end(),[&](const pair<int,int>& a, const pair<int,int>& b){
                //     if(a.second == b.second){
                //         return a.first < b.first;
                //     }
                //     return a.second > b.second;
                // });
                my_sort(nodes, 2, k);
                int limit = min(k,(int)nodes.size());
                unordered_set<int> top_k_nodes_set;
                for(int i = 0; i < limit; i++){
                    top_k_nodes.push_back(nodes[i].first);
                    top_k_nodes_set.insert(nodes[i].first);
                }
                int n=0;
                while(top_k_nodes.size() < k){
                    if(top_k_nodes_set.find(n) == top_k_nodes_set.end()){
                        top_k_nodes.push_back(n);
                    }
                    n++;
                }
            }
            output.push_back(top_k_nodes);
        }
    }
    return output;
}
