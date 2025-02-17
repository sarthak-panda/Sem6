# def filter_lines_with_substrings(input_file, output_file, substrings):
#     with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#         for line in infile:
#             if any(sub in line for sub in substrings):
#                 outfile.write(line)

# # Example usage
# substrings = ["ncpus=", "perf stat ./check_", "seconds time elapsed"]  # Replace with your desired substrings
# filter_lines_with_substrings("a.txt", "out.txt", substrings)
import matplotlib.pyplot as plt

# Data from CSV
cores = [2, 4, 8, 16, 32, 40]
data = {
    '64':   [9.77,  9.14,  9.49, 12.16, 16.58, 17.24],
    '256':  [19.39, 15.20, 14.65, 16.85, 22.53, 23.23],
    '1024': [63.41, 46.89, 40.91, 41.87, 49.76, 50.67],
    '4096': [408.15,279.10,219.37,196.55,194.39,193.01]
}

plt.figure(figsize=(10, 6))

# Plot each line with markers and a label
for b_value, times in data.items():
    plt.plot(cores, times, marker='o', linewidth=2, label=f'b = {b_value}')

# Beautify the graph
plt.title("Performance vs. Number of Cores for Different b Values", fontsize=14)
plt.xlabel("Number of Cores", fontsize=12)
plt.ylabel("Time (in ms)", fontsize=12)
plt.xticks(cores)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="b Values", fontsize=10)
plt.tight_layout()

plt.show()
