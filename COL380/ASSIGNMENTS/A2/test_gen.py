import os
import random
import networkx as nx
import matplotlib.pyplot as plt

# --------------------
# Parameters (set these as needed)
# --------------------
graph_name = "example1N"          # Graph name used for folder naming
total_vertices = 1000            # Total number of vertices in the graph
total_edges = 1500               # Total number of undirected edges
color_set = [0, 12, 52, 99, 4, 200, 999]       # Available colors for vertices (numeric)
is_connected = False            # Ensure the graph is connected (spanning tree + extra edges)
num_files = 40                  # Number of files to generate

# --------------------
# Helper function to partition a list into nearly equal parts.
# --------------------
def partition(lst, parts):
    avg = len(lst) // parts
    rem = len(lst) % parts
    result = []
    start = 0
    for i in range(parts):
        extra = 1 if i < rem else 0
        result.append(lst[start: start + avg + extra])
        start += avg + extra
    return result

# --------------------
# Step 1: Generate vertex color mapping.
# Each vertex gets a random color from color_set.
# --------------------
vertex_colors = [(v, random.choice(color_set)) for v in range(total_vertices)]

# --------------------
# Step 2: Generate edges.
# For a connected graph, first create a spanning tree then add extra edges.
# Edges are stored as tuples (u, v) with u < v to avoid duplicate undirected edges.
# --------------------
edges = set()
if is_connected:
    if total_edges < total_vertices - 1:
        raise ValueError("For a connected graph, total_edges must be at least total_vertices - 1.")
    # Create a spanning tree by shuffling vertices and connecting consecutive ones.
    vertices = list(range(total_vertices))
    random.shuffle(vertices)
    for i in range(len(vertices) - 1):
        u, v = vertices[i], vertices[i + 1]
        edge = (min(u, v), max(u, v))
        edges.add(edge)

# Add additional random edges until we reach the total_edges count.
while len(edges) < total_edges:
    u = random.randint(0, total_vertices - 1)
    v = random.randint(0, total_vertices - 1)
    if u == v:
        continue  # no self-loops
    edge = (min(u, v), max(u, v))
    if edge in edges:
        continue  # skip duplicates
    edges.add(edge)

edges = list(edges)  # convert the set to a list for partitioning

# --------------------
# Step 3: Partition the vertex mappings and edges among the files.
# --------------------
vertex_partitions = partition(vertex_colors, num_files)
edge_partitions = partition(edges, num_files)

# --------------------
# Step 4: Create the output folder and write each file.
# Each file is named "graph_{i}.txt" (i from 1 to num_files) and has:
#   - First line: <number_of_vertices_in_file>,<number_of_edges_in_file>
#   - Next lines: vertex color mappings ("vertex color")
#   - Next lines: edges ("u v")
# --------------------
folder_name = f"GRAPH_{graph_name}"
os.makedirs(folder_name, exist_ok=True)

for i in range(num_files):
    filename = os.path.join(folder_name, f"graph_{i+1}.txt")
    with open(filename, 'w') as f:
        num_vertices_in_file = len(vertex_partitions[i])
        num_edges_in_file = len(edge_partitions[i])
        # Write header line
        f.write(f"{num_vertices_in_file} {num_edges_in_file}\n")
        # Write vertex color mappings
        for vertex, color in vertex_partitions[i]:
            f.write(f"{vertex} {color}\n")
        # Write edges
        for u, v in edge_partitions[i]:
            f.write(f"{u} {v}\n")

print(f"Graph files generated in folder: {folder_name}")

# --------------------
# Step 5: Generate and save an overall visualization of the graph.
# Nodes are drawn with their numbers and colored based on the vertex color mapping.
# --------------------
# Create the overall graph
G = nx.Graph()
G.add_nodes_from(range(total_vertices))
G.add_edges_from(edges)

# Create a dictionary for vertex colors
vertex_color_dict = {v: color for v, color in vertex_colors}

# Map numeric colors to actual color names for visualization.
# (You can adjust the mapping as needed.)
default_color_mapping = {
    0: "red", 
    1: "blue", 
    2: "green", 
    3: "orange", 
    4: "purple"
}
node_colors = [default_color_mapping.get(vertex_color_dict[v], "gray") for v in G.nodes()]

# Use a spring layout for visualization (with fixed seed for consistency).
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(8, 8))
nx.draw(G, pos, labels={v: str(v) for v in G.nodes()}, node_color=node_colors, 
        with_labels=True, edge_color='black', font_size=8, node_size=300)
plt.title("Overall Graph")
plt.tight_layout()
plt.savefig("pic.jpg")
plt.close()

print("Graph image saved as pic.jpg")
