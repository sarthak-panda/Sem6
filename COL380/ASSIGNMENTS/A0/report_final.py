import csv
from tabulate import tabulate
import pandas as pd
import numpy as np, os, time
import matplotlib.pyplot as plt

os.system("make")
def plot(data_plot, sizes):
    type_map = ["IJK", "IKJ", "JIK", "JKI", "KIJ", "KJI"]
    # total user time vs matrix multiplication size
    y_axis_data = {}
    for i in range(6):
        y_axis_data[i] = []
        for size in sizes:
            y_axis_data[i].append(data_plot[size][i][0])
    for i in range(6):
        plt.plot(sizes, y_axis_data[i], label=type_map[i])
    plt.legend()
    plt.title("user time vs matrix multiplication size")
    plt.savefig(
        "Analysis_output/user time vs matrix multiplication size.png", dpi=400, bbox_inches="tight"
    )
    plt.show()
    # user time in matrix multiplication vs matrix multiplication size
    y_axis_data = {}
    for i in range(6):
        y_axis_data[i] = []
        for size in sizes:
            y_axis_data[i].append(data_plot[size][i][1])
    for i in range(6):
        plt.plot(sizes, y_axis_data[i], label=type_map[i])
    plt.legend()
    plt.title("user time in matrix multiplication vs matrix multiplication size")
    plt.savefig(
        "Analysis_output/user time in matrix multiplication vs matrix multiplication size.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.show()
    # cache hit rate vs matrix multiplication size
    y_axis_data = {}
    for i in range(6):
        y_axis_data[i] = []
        for size in sizes:
            y_axis_data[i].append(data_plot[size][i][2])
    for i in range(6):
        plt.plot(sizes, y_axis_data[i], label=type_map[i])
    plt.legend()
    plt.title("cache hit rate vs matrix multiplication size")
    plt.savefig(
        "Analysis_output/cache hit rate vs matrix multiplication size.png", dpi=400, bbox_inches="tight"
    )
    plt.show()


def helper(perm_num, size, data_plot):
    type_map = ["IJK", "IKJ", "JIK", "JKI", "KIJ", "KJI"]
    file = open(f"Analysis_output/perf_report_{size}_{perm_num}.txt")
    lines = file.readlines()
    data_to_collect = [
        "task-clock",
        "cpu_core/cycles/",
        "cpu_core/instructions/",
        "cpu_core/cache-references/",
        "cpu_core/cache-misses/",
    ]
    data_idx = 0
    data_map = {}
    task_clock_phase = False
    task_clock_sub_field_recorded = 0
    for idx, line in enumerate(lines):
        if task_clock_sub_field_recorded == 4:
            task_clock_phase = False
        if task_clock_phase:
            if "[.] matrixMultiply" in line:
                tokens = line.split()
                percentage = float(tokens[1][:-1])
                data_map["task-clock-matrix-multiply"] = percentage
                task_clock_sub_field_recorded += 1
            elif "[.] writeMatrix" in line:
                tokens = line.split()
                percentage = float(tokens[1][:-1])
                data_map["task-clock-write-matrix"] = percentage
                task_clock_sub_field_recorded += 1
            elif "[.] readMatrix" in line:
                tokens = line.split()
                percentage = float(tokens[1][:-1])
                data_map["task-clock-read-matrix"] = percentage
                task_clock_sub_field_recorded += 1
            elif "[.] main" in line:
                tokens = line.split()
                percentage = float(tokens[1][:-1])
                data_map["task-clock-main"] = percentage
                task_clock_sub_field_recorded += 1
        elif "# Samples:" in line:
            if data_to_collect[data_idx] in line:
                nxt_line = lines[idx + 1]
                tokens = nxt_line.split()
                data = int(tokens[-1])
                data_map[data_to_collect[data_idx]] = data
                if data_idx == 0:
                    task_clock_phase = True
                data_idx += 1
    data_plot[size][perm_num] = [
        data_map["task-clock"] / (10**9),
        (data_map["task-clock-matrix-multiply"] * data_map["task-clock"]) / (100 * (10**9)),
        1 - (data_map["cpu_core/cache-misses/"] / data_map["cpu_core/cache-references/"]),
    ]
    # gprof ./main gmon.out > analysis.txt
    # read  analysis.txt here
    file = open(f"Analysis_output/analysis_{size}_{perm_num}.txt")
    lines = file.readlines()
    count = 0
    for line in lines:
        if count == 3:
            break
        if "matrixMultiply" in line:
            tokens = line.split()
            data_map["gprof-matrix-multiply"] = float(tokens[0])
            count += 1
        elif "readMatrix" in line:
            tokens = line.split()
            data_map["gprof-read-matrix"] = float(tokens[0])
            count += 1
        elif "writeMatrix" in line:
            tokens = line.split()
            data_map["gprof-write-matrix"] = float(tokens[0])
            count += 1
    # let us create the tex table comparing gprof and perf percentage time
    row_labels = [
        "%age of time spent in matrixMultiply",
        "%age of time spent in readMatrix",
        "%age of time spent in writeMatrix",
        "%age of time spent in main",
    ]
    table_data = {
        "gprof": [
            data_map["gprof-matrix-multiply"],
            data_map["gprof-read-matrix"],
            data_map["gprof-write-matrix"],
            "N/A",
        ],
        "perf": [
            data_map["task-clock-matrix-multiply"],
            data_map["task-clock-read-matrix"],
            data_map["task-clock-write-matrix"],
            data_map["task-clock-main"],
        ],
    }
    df = pd.DataFrame(table_data, index=row_labels)
    latex_table = tabulate(df, tablefmt="latex", headers="keys", showindex=True)
    latex_doccument = f"""
    \\documentclass{{article}}
    \\begin{{document}}
    Matrix Size: {size}x{size}, Permutation Number: {type_map[perm_num]} \\\\
    {latex_table}
    \\end{{document}}
    """
    with open(f"Analysis_output/perf_gprof_table_{size}_{perm_num}.tex", "w") as f:
        f.write(latex_doccument)
    # write to csv and draw graph
    f = open("Analysis_output/raw_data.csv", "a")
    writer = csv.writer(f)
    row = [
        perm_num,
        size,
        data_map["task-clock"] / (10**9),
        (data_map["task-clock-matrix-multiply"] * data_map["task-clock"]) / (100 * (10**9)),
        data_map["task-clock-matrix-multiply"],
        data_map["cpu_core/cache-misses/"],
        1 - (data_map["cpu_core/cache-misses/"] / data_map["cpu_core/cache-references/"]),
        (data_map["cpu_core/instructions/"] / data_map["cpu_core/cycles/"]),
    ]
    writer.writerow(row)
    f.close()
    # important remember to remove -g and -pg flags from makefile
f = open("Analysis_output/raw_data.csv", "w")
header = [
    "permutation_number",
    "dimension",
    "total_user_time(in sec)",
    "user_time_in_matrix_multiplication(in sec)",
    "percent_of_user_time_taken_by_matrix_multiplication",
    "total_cache_misses",
    "cache_hit_rate",
    "instructions_per_cycle",
]
writer = csv.writer(f)
writer.writerow(header)
f.close()
data_plot = {}
sizes = []
for size in range(1000, 6000, 1000):
    sizes.append(size)
    data_plot[size] = {}
    path_input, path_output = f"./Analysis_output/input_path_{size}/", f"./Analysis_output/output_path_{size}/"
    # Generate random matrices
    mtx_A = np.random.random(size=(size, size)) * 1e2  # dtype = float64
    mtx_B = np.random.random(size=(size, size)) * 1e2  # dtype = float64
    mtx_C = (mtx_A @ mtx_B).flatten()  # dtpye = float64
    if not os.path.exists(path_input):
        os.makedirs(path_input)
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    with open(f"{path_input}/mtx_A.bin", "wb") as fp:
        fp.write(mtx_A.tobytes())
    with open(f"{path_input}/mtx_B.bin", "wb") as fp:
        fp.write(mtx_B.tobytes())    
    for perm_num in range(6):
        os.system(
            f"perf record -e task-clock,cycles,instructions,cache-references,cache-misses,user_time -F 1000 -g --no-kernel ./main {perm_num} {size} {size} {size} ./Analysis_output/input_path_{size}/ ./Analysis_output/output_path_{size}/"
        )
        os.system(f"perf report --stdio > Analysis_output/perf_report_{size}_{perm_num}.txt")
        os.system(f"gprof ./main gmon.out > Analysis_output/analysis_{size}_{perm_num}.txt")
        helper(perm_num, size, data_plot)
        with open(f"{path_output}/mtx_C.bin", "rb") as fp:
            student_result = np.frombuffer(fp.read(), dtype=mtx_C.dtype)
        if mtx_C.shape != student_result.shape:
            print("The result matrix shape didn't match")
            # return False
        result = np.allclose(mtx_C, student_result, rtol=1e-10, atol=1e-12)
        if not result:
            print("Test Case failed")
plot(data_plot, sizes)
# perf record -e task-clock,cycles,instructions,cache-references,cache-misses,user_time -F 1000 -g --no-kernel ./main 0 1000 1000 1000 ./input_path/ ./output_path/
# perf report --stdio > perf_report.txt
