matrix_sizes=(1000)
permutations=(0 1 2 3 4 5)
input_path="./input_path/"
output_path="./output_path/"

# Create the results file and add the header
echo "Permutation,MatrixSize,AtomCacheReferences,CoreCacheReferences,AtomCacheMisses,CoreCacheMisses,AtomCacheHitRate,CoreCacheHitRate,ExecutionTime,UserTime,InstrctionsPerCycle" > results.csv

# Iterate over matrix sizes and permutations
for size in "${matrix_sizes[@]}"; do
    for perm in "${permutations[@]}"; do
        cummulative_atom_cache_references=0
        cummulative_count_atom_cache_references=0
        cummulative_core_cache_references=0
        cummulative_atom_cache_misses=0
        cummulative_count_atom_cache_misses=0
        cummulative_core_cache_misses=0
        cummulative_exec_time=0
        cummulative_atom_cache_hit_rate=0
        cummulative_count_atom_cache_hit_rate=0
        cummulative_core_cache_hit_rate=0
        for i in {1..3}; do
            # Run the program with perf and capture the output
            output=$(perf stat -e cache-references,cache-misses,task-clock ./main $perm $size $size $size $input_path $output_path 2>&1)

            # Extract performance metrics
            atom_cache_references=$(echo "$output" | grep 'cpu_atom/cache-references/' | awk '{print $1}' | tr -d ',')
            if [[ ! "$atom_cache_references" =~ ^[0-9]+$ ]]; then
                atom_cache_references="N/A"
            else
                cummulative_atom_cache_references=$(echo "$cummulative_atom_cache_references + $atom_cache_references" | bc)
                cummulative_count_atom_cache_references=$(($cummulative_count_atom_cache_references + 1))
            fi        
            core_cache_references=$(echo "$output" | grep 'cpu_core/cache-references/' | awk '{print $1}' | tr -d ',')
            cummulative_core_cache_references=$(echo "$cummulative_core_cache_references + $core_cache_references" | bc)
            atom_cache_misses=$(echo "$output" | grep 'cpu_atom/cache-misses/' | awk '{print $1}' | tr -d ',')
            if [[ ! "$atom_cache_misses" =~ ^[0-9]+$ ]]; then
                atom_cache_misses="N/A"
            else
                cummulative_atom_cache_misses=$(echo "$cummulative_atom_cache_misses + $atom_cache_misses" | bc)
                cummulative_count_atom_cache_misses=$(($cummulative_count_atom_cache_misses + 1))
            fi          
            core_cache_misses=$(echo "$output" | grep 'cpu_core/cache-misses/' | awk '{print $1}' | tr -d ',')
            cummulative_core_cache_misses=$(echo "$cummulative_core_cache_misses + $core_cache_misses" | bc)
            exec_time=$(echo "$output" | grep 'seconds time elapsed' | awk '{print $1}')
            cummulative_exec_time=$(echo "$cummulative_exec_time + $exec_time" | bc)
            =$(echo "$output" | grep 'cpu_atom/instructions/' | awk '{print $1}' | tr -d ',')
            =$(echo "$output" | grep 'cpu_core/instructions/' | awk '{print $1}' | tr -d ',')
            # Calculate cache hit rates
            if [[ "$atom_cache_misses" =~ ^[0-9]+$ ]] && [[  "$atom_cache_references" =~ ^[0-9]+$ ]]; then
                atom_cache_hit_rate=$(bc -l <<< "1 - $atom_cache_misses/$atom_cache_references")
                cummulative_atom_cache_hit_rate=$(echo "$cummulative_atom_cache_hit_rate + $atom_cache_hit_rate" | bc)
                cummulative_count_atom_cache_hit_rate=$(($cummulative_count_atom_cache_hit_rate + 1))
            else
                atom_cache_hit_rate="N/A"
            fi
            core_cache_hit_rate=$(bc -l <<< "1 - $core_cache_misses/$core_cache_references")
            cummulative_core_cache_hit_rate=$(echo "$cummulative_core_cache_hit_rate + $core_cache_hit_rate" | bc)
        done

        if [[ "$cummulative_count_atom_cache_references" == 0 ]]; then
            cummulative_atom_cache_references="N/A"
        else
            cummulative_atom_cache_references=$(bc -l <<< "$cummulative_atom_cache_references / $cummulative_count_atom_cache_references")
            cummulative_atom_cache_references=$(printf "%.3f" "$cummulative_atom_cache_references")
        fi

        cummulative_core_cache_references=$(bc -l <<< "$cummulative_core_cache_references / 3")
        cummulative_core_cache_references=$(printf "%.3f" "$cummulative_core_cache_references")

        if [[ "$cummulative_count_atom_cache_misses" == 0 ]]; then
            cummulative_atom_cache_misses="N/A"
        else
            cummulative_atom_cache_misses=$(bc -l <<< "$cummulative_atom_cache_misses / $cummulative_count_atom_cache_misses")
            cummulative_atom_cache_misses=$(printf "%.3f" "$cummulative_atom_cache_misses")
        fi
        
        cummulative_core_cache_misses=$(bc -l <<< "$cummulative_core_cache_misses / 3")
        cummulative_core_cache_misses=$(printf "%.3f" "$cummulative_core_cache_misses")
        cummulative_exec_time=$(bc -l <<< "$cummulative_exec_time / 3")
        cummulative_exec_time=$(printf "%.3f" "$cummulative_exec_time")

        if [[ "$cummulative_count_atom_cache_hit_rate" == 0 ]]; then
            cummulative_atom_cache_hit_rate="N/A"
        else
            cummulative_atom_cache_hit_rate=$(bc -l <<< "$cummulative_atom_cache_hit_rate / $cummulative_count_atom_cache_hit_rate")
            cummulative_atom_cache_hit_rate=$(printf "%.3f" "$cummulative_atom_cache_hit_rate")
        fi

        cummulative_core_cache_hit_rate=$(bc -l <<< "$cummulative_core_cache_hit_rate / 3")
        cummulative_core_cache_hit_rate=$(printf "%.3f" "$cummulative_core_cache_hit_rate")

        # Append to results.csv
        echo "$perm,$size,$cummulative_atom_cache_references,$cummulative_core_cache_references,$cummulative_atom_cache_misses,$cummulative_core_cache_misses,$cummulative_atom_cache_hit_rate,$cummulative_core_cache_hit_rate,$cummulative_exec_time" >> results.csv
    done
done