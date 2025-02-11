CC = g++
CFLAGS = -fopenmp -std=c++17
GEN_SRC = check_generation.cpp template.cpp
CHK_SRC = check_exponentiation.cpp template.cpp
GEN_EXE = generate
CHK_EXE = check
PRECOMPUTED_DIR = precomputed_result
RESULTS = results

.PHONY: all clean

all: $(GEN_EXE) $(CHK_EXE) run_check

$(GEN_EXE): $(GEN_SRC)
	$(CC) $(CFLAGS) $^ -o $@

$(CHK_EXE): $(CHK_SRC)
	$(CC) $(CFLAGS) $^ -o $@

run_check: $(GEN_EXE) $(CHK_EXE)
	./$(GEN_EXE)
	sudo perf stat ./$(CHK_EXE)

clean:
	rm -rf $(GEN_EXE) $(CHK_EXE) $(PRECOMPUTED_DIR) $(RESULTS)
