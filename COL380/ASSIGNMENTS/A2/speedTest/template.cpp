#include "template.hpp"
#include <algorithm>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <mpi.h>
#include <unordered_set>
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
    MPI_Init(&argc, &argv);
}
void end_mpi() {
    MPI_Finalize();
}
vector<vector<int>> degree_cen(vector<pair<int, int>>& partial_edge_list, map<int, int>& partial_vertex_color, int k) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &size);  
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
    int total_num_vertices = global_buff.size()/2;
    vector<int> vertex_color(global_buff.size()/2);
    set<int> colors;
    for(int i = 0; i < global_buff.size(); i+=2){
        vertex_color[global_buff[i]] = global_buff[i+1];
        //if(rank == 0){
        colors.insert(global_buff[i+1]);
        //}
    }
    unordered_map<int,int>virtual_color;
    //unordered_map<int,int>reverse_virtual_color;
    int vc=0;
    for(auto color: colors){
        virtual_color[color] = vc;
        //reverse_virtual_color[vc] = color;
        vc++;
    }
    vector<vector<int>> partial_color_vertex_map(colors.size(), vector<int>(total_num_vertices, 0));
    for(auto e: partial_edge_list){
        int u = e.first;
        int v = e.second;
        int color_u = virtual_color[vertex_color[u]];
        int color_v = virtual_color[vertex_color[v]];
        partial_color_vertex_map[color_u][v]++;
        partial_color_vertex_map[color_v][u]++;
    }
    size_t total_local_results = 0;
    for(int color = 0; color < partial_color_vertex_map.size(); color++){
        total_local_results += 2; 
        total_local_results += partial_color_vertex_map[color].size() * 2; 
    }
    vector<int> local_results;
    local_results.reserve(total_local_results);
    for(int color = 0; color < partial_color_vertex_map.size(); color++){
        local_results.push_back(color);
        int color_idenrifier = -9;
        local_results.push_back(color_idenrifier);
        for(int i = 0; i < partial_color_vertex_map[color].size(); i++){
            local_results.push_back(i);
            local_results.push_back(partial_color_vertex_map[color][i]);
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
        vector<vector<int>> color_vertex_map(colors.size(), vector<int>(total_num_vertices, 0));
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
        for(int color = 0; color < color_vertex_map.size(); color++){
            vector<int>top_k_nodes;
            top_k_nodes.reserve(k);
            vector<pair<int, int>> nodes;
            nodes.reserve(color_vertex_map[color].size());
            for(int i = 0; i < color_vertex_map[color].size(); i++){
                nodes.push_back({i, color_vertex_map[color][i]});
            }
            my_sort(nodes, 2, k);
            int limit = min(k,(int)nodes.size());
            for(int i = 0; i < limit; i++){
                top_k_nodes.push_back(nodes[i].first);
            }
            output.push_back(top_k_nodes);
        }
    }
    return output;
}
