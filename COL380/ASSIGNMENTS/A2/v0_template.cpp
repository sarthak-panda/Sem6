#include "template.hpp"
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <mpi.h>

using namespace std;

void init_mpi(int argc, char* argv[]) {
    //Code Here
    MPI_Init(argc, &argv);
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
    vector<int> local_buff;
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
    for(int i = 0; i < global_buff.size(); i+=2){
        vertex_color[global_buff[i]] = global_buff[i+1];
    }
    //work that each thread will do after getting the complete color map
    map<int,map<int,int>> partial_color_vertex_map;
    for(auto e: partial_edge_list){
        int u = e.first;
        int v = e.second;
        int color_u = vertex_color[u];
        int color_v = vertex_color[v];
        // if(partial_color_vertex_map.find(color_u) == partial_color_vertex_map.end()){
        //     partial_color_vertex_map[color_u] = map<int,int>();
        // }
        // if(partial_color_vertex_map.find(color_v) == partial_color_vertex_map.end()){
        //     partial_color_vertex_map[color_v] = map<int,int>();
        // }
        // if(partial_color_vertex_map[color_u].find(v) == partial_color_vertex_map[color_u].end()){
        //     partial_color_vertex_map[color_u][v] = 0;
        // }
        // if(partial_color_vertex_map[color_v].find(u) == partial_color_vertex_map[color_v].end()){
        //     partial_color_vertex_map[color_v][u] = 0;
        // }
        partial_color_vertex_map[color_u][v]++;
        partial_color_vertex_map[color_v][u]++;
    }
    vector<int> local_results;
    for(auto &e: partial_color_vertex_map){
        int color= e.first;
        for(auto &v: e.second){
            local_results.push_back(color);
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
        map<int,map<int,int>> color_vertex_map;
        for(int i = 0; i < global_results.size(); i+=3){
            int color = global_results[i];
            int vertex = global_results[i+1];
            int partial_deg = global_results[i+2];
            // if(color_vertex_map.find(color) == color_vertex_map.end()){
            //     color_vertex_map[color] = map<int,int>();
            // }
            color_vertex_map[color][vertex] += partial_deg;
        }
        set<int> colors;
        for(auto e: color_vertex_map){
            colors.insert(e.first);
        }
        for(auto color: colors){
            vector<pair<int, int>> nodes;
            for(auto &e: color_vertex_map[color]){
                nodes.push_back({e.first, e.second});
            }
            sort(nodes.begin(),nodes.end(),[&](const pair<int,int>& a, const pair<int,int>& b){
                if(a.second == b.second){
                    return a.first < b.first;
                }
                return a.second > b.second;
            });
            vector<int>top_k_nodes;
            int limit = min(k,(int)nodes.size());
            for(int i = 0; i < limit; i++){
                top_k_nodes.push_back(nodes[i].first);
            }
            output.push_back(top_k_nodes);
        }
    }
    return output;
}
