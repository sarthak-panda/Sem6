import csv
# perf record -e task-clock,cycles,instructions,cache-references,cache-misses,user_time -F 1000 -g --no-kernel ./main 0 1000 1000 1000 ./input_path/ ./output_path/
# perf report --stdio > perf_report.txt
perm_num=0
size=1000
file=open('perf_report.txt')
lines=file.readlines()
data_to_collect = ['task-clock','cpu_core/cycles/','cpu_core/instructions/','cpu_core/cache-references/', 'cpu_core/cache-misses/']
data_idx=0
data_map={}
task_clock_phase=False
task_clock_sub_field_recorded=0
for idx,line in enumerate(lines):
    if task_clock_sub_field_recorded==4:
        task_clock_phase=False
    if task_clock_phase:
        if '[.] matrixMultiply' in line:
            tokens=line.split()
            percentage=float(tokens[0][:-1])
            data_map['task-clock-matrix-multiply']=percentage
            task_clock_sub_field_recorded+=1
        elif '[.] writeMatrix' in line:
            tokens=line.split()
            percentage=float(tokens[0][:-1])
            data_map['task-clock-write-matrix']=percentage     
            task_clock_sub_field_recorded+=1
        elif '[.] readMatrix' in line:
            tokens=line.split()
            percentage=float(tokens[0][:-1])
            data_map['task-clock-read-matrix']=percentage   
            task_clock_sub_field_recorded+=1                          
        elif '[.] main' in line:
            tokens=line.split()
            percentage=float(tokens[0][:-1])
            data_map['task-clock-main']=percentage
            task_clock_sub_field_recorded+=1    
    elif '# Samples:' in line:
        if data_to_collect[data_idx] in line:
            nxt_line=lines[idx+1]
            tokens=nxt_line.split()
            data=int(tokens[-1])
            data_map[data_to_collect[data_idx]]=data
            if data_idx==0:
                task_clock_phase=True
            data_idx+=1
print(data_map)
#gprof data
#gprof ./main gmon.out > analysis.txt
#read  analysis.txt here
#write to csv and draw graph
'''
Permutation  number,  the  row  dimension  of  the  input  matrices 
(assuming all are square), total user time, user time in matrix multiplication, the percent of 
user  time  taken  by  matrix  multiplication,  total  number  of  cache  misses,  cache-hit  rate, 
instructions per cycle 
'''
f = open('raw_data.csv', 'w')
header = ['permutation_number', 'dimension', 'total_user_time(in sec)', 'user_time_in_matrix_multiplication(in sec)','percent_of_user_time_taken_by_matrix_multiplication', 'total_cache_misses', 'cache_hit_rate', 'instructions_per_cycle']
writer = csv.writer(f)
writer.writerow(header)
row=[perm_num,size,data_map['task-clock']/(10**9),(data_map['task-clock-matrix-multiply']*data_map['task-clock'])/(100*(10**9)),data_map['task-clock-matrix-multiply'],data_map['cpu_core/cache-misses/'],1-(data_map['cpu_core/cache-misses/']/data_map['cpu_core/cache-references/']),(data_map['cpu_core/instructions/']/data_map['cpu_core/cycles/'])]
writer.writerow(row)
f.close()
#important remember to remove -g and -pg flags from makefile