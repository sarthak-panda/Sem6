import numpy as np
import os
import time
import matplotlib.pyplot as plt

def execute_test_case(type, number_row1, number_col1, number_col2, path_input, path_output, mtx_A, mtx_B):
    
    mtx_C = (mtx_A @ mtx_B).flatten()  

    
    if not os.path.exists(path_input):
        os.makedirs(path_input)
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    
    with open(f"{path_input}/mtx_A.bin", "wb") as fp:
        fp.write(mtx_A.tobytes())
    with open(f"{path_input}/mtx_B.bin", "wb") as fp:
        fp.write(mtx_B.tobytes())
    
    
    os.system("make")

    
    time_start = time.perf_counter()
    os.system(f"./main {type} {number_row1} {number_col1} {number_col2} {path_input} {path_output}")
    time_duration = time.perf_counter() - time_start  

    
    with open(f"{path_output}/mtx_C.bin", "rb") as fp:
        student_result = np.frombuffer(fp.read(), dtype=mtx_C.dtype)
    
    
    if mtx_C.shape != student_result.shape:
        print("The result matrix shape didn't match")
        return False, time_duration
    
    
    result = np.allclose(mtx_C, student_result, rtol=1e-10, atol=1e-12)

    return result, time_duration


if __name__ == "__main__":
    file  ="data_question_1.txt"
    with open(file,"w") as f:
        data = {}
        sizes = []
        for n in range(1000, 6000, 1000):
            sizes.append(n)
            
            mtx_A = np.random.random(size=(n, n)) * 1e2  
            mtx_B = np.random.random(size=(n, n)) * 1e2  
            
            for type in range(6):
                cumulative_time = 0
                for _ in range(1):  
                    result, time_duration = execute_test_case(
                        type, n, n, n, "./input_path/", "./output_path/", mtx_A, mtx_B
                    )
                    cumulative_time += time_duration
                    if not result:
                        print("Test Case failed")
                    else:
                        print(f"Test Case passed in {time_duration} seconds")
                        f.write(f"Size: {n}, Permutation: {type}, Time: {time_duration:.6f} seconds\n")
                cumulative_time /= 1
                data[type] = data.get(type, [])
                data[type].append(cumulative_time)
    
    
    type_map = ["IJK", "IKJ", "JIK", "JKI", "KIJ", "KJI"]
    for i in range(6):
        plt.plot(sizes, data[i], label=type_map[i])
    plt.legend()
    plt.title("Execution time vs matrix multiplication size in experiment 1")
    plt.savefig(
        "Execution time vs matrix multiplication size in experiment 1.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.show()

#     ##########perf record -e task-clock,cycles,instructions,cache-references,cache-misses,user_time -F 1000 -g --no-kernel ./main 0 1000 1000 1000 ./input_path/ ./output_path/