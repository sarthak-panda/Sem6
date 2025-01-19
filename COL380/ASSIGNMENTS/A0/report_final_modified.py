from tabulate import tabulate
import pandas as pd

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
                percentage = float(tokens[0][:-1])
                data_map["task-clock-matrix-multiply"] = percentage
                task_clock_sub_field_recorded += 1
            elif "[.] writeMatrix" in line:
                tokens = line.split()
                percentage = float(tokens[0][:-1])
                data_map["task-clock-write-matrix"] = percentage
                task_clock_sub_field_recorded += 1
            elif "[.] readMatrix" in line:
                tokens = line.split()
                percentage = float(tokens[0][:-1])
                data_map["task-clock-read-matrix"] = percentage
                task_clock_sub_field_recorded += 1
            elif "[.] main" in line:
                tokens = line.split()
                percentage = float(tokens[0][:-1])
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
    with open(f"Analysis_output_new/perf_gprof_table_{size}_{perm_num}.tex", "w") as f:
        f.write(latex_doccument)
    # write to csv and draw graph
    # important remember to remove -g and -pg flags from makefile

data_plot = {}
sizes = []
for size in range(1000, 6000, 1000):
    sizes.append(size)
    data_plot[size] = {}
    for perm_num in range(6):
        helper(perm_num, size, data_plot)
# perf record -e task-clock,cycles,instructions,cache-references,cache-misses,user_time -F 1000 -g --no-kernel ./main 0 1000 1000 1000 ./input_path/ ./output_path/
# perf report --stdio > perf_report.txt
