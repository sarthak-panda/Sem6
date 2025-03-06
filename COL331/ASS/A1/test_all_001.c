#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>
#include <sys/wait.h>

// Define syscall numbers as per your setup
#define SYS_HELLO          451
#define SYS_REGISTER       452
#define SYS_FETCH          453
#define SYS_DEREGISTER     454
#define SYS_RESOURCE_CAP   455
#define SYS_RESOURCE_RESET 456

// Updated per_proc_resource structure
struct per_proc_resource {
    pid_t pid;                    /* process id */
    unsigned long heapsize;       /* Total memory allocated using brk and mmap (in MB) */
    unsigned long openfile_count; /* Total number of open files */
    long heap_quota;              /* Heap quota in MB */
    long file_quota;              /* File quota */
    bool quotas_defined;          /* Indicates if quotas are defined */
};

/*
 * Helper function: allocate total "mb" megabytes,
 * half using sbrk and half using malloc.
 * Returns the malloc-allocated pointer (so it can later be freed)
 * while the sbrk part remains allocated.
 */
void* allocate_mb(size_t mb) {
    size_t total_bytes = mb * 1024 * 1024;
    size_t half_bytes = total_bytes / 2;
    void *sbrk_mem = sbrk(half_bytes);
    if (sbrk_mem == (void *) -1) {
        perror("sbrk");
        exit(EXIT_FAILURE);
    }
    void *malloc_mem = malloc(half_bytes);
    if (!malloc_mem) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    memset(sbrk_mem, 0, half_bytes);
    memset(malloc_mem, 0, half_bytes);
    printf("Allocated %zu MB: %zu MB using sbrk, %zu MB using malloc\n",
           mb, half_bytes/(1024*1024), half_bytes/(1024*1024));
    return malloc_mem;
}

void run_test(int i) {
    pid_t pid = getpid();
    int ret;

    printf("\n==== PID %d: Starting tests ====\n", pid);

    // 1. Test sys_hello (just for sanity)
    ret = syscall(SYS_HELLO);
    printf("PID %d: sys_hello returned: %d\n", pid, ret);

    // --- Tests for sys_register ---
    // (a) Negative case: pid < 1  --> should return -22
    ret = syscall(SYS_REGISTER, -1);
    printf("PID %d: sys_register(-1) returned: %d (expected -22)\n", pid, ret);

    // (b) Negative case: non-existent pid (choose 99999) --> should return -3
    ret = syscall(SYS_REGISTER, 99999);
    printf("PID %d: sys_register(99999) returned: %d (expected -3)\n", pid, ret);

    // (c) Valid registration
    ret = syscall(SYS_REGISTER, pid);
    printf("PID %d: sys_register(valid pid) returned: %d (expected 0)\n", pid, ret);

    // (d) Negative case: duplicate registration --> should return -23
    ret = syscall(SYS_REGISTER, pid);
    printf("PID %d: sys_register(duplicate) returned: %d (expected -23)\n", pid, ret);

    // --- Memory allocation ---
    // Instead of 5 MB in the original code, we now allocate 6 MB (an even number)
    // so that exactly 50% is allocated via sbrk and 50% via malloc.
    void *malloc_mem = allocate_mb(6);  // returns malloc-allocated pointer

    // --- Open 3 dummy files ---
    int fds[3];
    char filename[50];
    for (int j = 0; j < 3; j++) {
        snprintf(filename, sizeof(filename), "dummy%d_%d.txt", pid, j+1);
        fds[j] = open(filename, O_RDWR | O_CREAT, 0666);
        if (fds[j] < 0) {
            perror("open");
            exit(EXIT_FAILURE);
        }
        printf("PID %d: Opened file: %s\n", pid, filename);
    }

    // --- Tests for sys_resource_cap ---
    // (a) Negative case: non-existent pid --> expect -3
    ret = syscall(SYS_RESOURCE_CAP, 99999, 7L, 5L);
    printf("PID %d: sys_resource_cap(99999, 7, 5) returned: %d (expected -3)\n", pid, ret);

    // (b) Negative case: valid pid but not registered.
    // Using parent's pid (assumed not registered) for testing.
    pid_t ppid = getppid();
    ret = syscall(SYS_RESOURCE_CAP, ppid, 7L, 5L);
    printf("PID %d: sys_resource_cap(ppid=%d, 7, 5) returned: %d (expected -22)\n", pid, ppid, ret);

    // (c) Valid case: set resource limits for our pid.
    ret = syscall(SYS_RESOURCE_CAP, pid, 7L, 5L);
    printf("PID %d: sys_resource_cap(valid pid, 7, 5) returned: %d (expected 0)\n", pid, ret);

    // (d) Negative case: quotas already defined --> expect -23
    ret = syscall(SYS_RESOURCE_CAP, pid, 9L, 6L);
    printf("PID %d: sys_resource_cap(duplicate cap) returned: %d (expected -23)\n", pid, ret);

    // --- Fetch and display current resource usage ---
    struct per_proc_resource stats;
    ret = syscall(SYS_FETCH, &stats, pid);
    printf("\nPID %d: sys_fetch returned: %d\n", pid, ret);
    printf("PID %d: Process ID: %d\n", pid, stats.pid);
    printf("PID %d: Heap Size: %lu MB\n", pid, stats.heapsize);
    printf("PID %d: Open File Count: %lu\n", pid, stats.openfile_count);
    printf("PID %d: Heap Quota: %ld, File Quota: %ld, Quotas defined: %s\n",
           pid, stats.heap_quota, stats.file_quota,
           stats.quotas_defined ? "true" : "false");

    printf("\n----->PID %d: Doing Dummy Work (sleeping for %d seconds)<-----\n", pid, 5 + i);
    sleep(5 + i);
    printf("\n----->PID %d: Completed Dummy Work (waking up)<-----\n", pid);

    // --- Additional memory allocation increase ---
    // Allocate 2 MB (again split equally between sbrk and malloc)
    void *malloc_mem_extra = allocate_mb(2);

    // Open one more file (4th file)
    snprintf(filename, sizeof(filename), "dummy4_%d.txt", pid);
    int fd4 = open(filename, O_RDWR | O_CREAT, 0666);
    if (fd4 < 0) {
        perror("open dummy4.txt");
        exit(EXIT_FAILURE);
    }
    printf("PID %d: Opened file: %s\n", pid, filename);

    // --- Tests for sys_resource_reset ---
    // (a) Negative case: non-existent pid --> expect -3
    ret = syscall(SYS_RESOURCE_RESET, 99999);
    printf("PID %d: sys_resource_reset(99999) returned: %d (expected -3)\n", pid, ret);

    // (b) Negative case: valid pid not in monitored list (using parent's pid) --> expect -22
    ret = syscall(SYS_RESOURCE_RESET, ppid);
    printf("PID %d: sys_resource_reset(ppid=%d) returned: %d (expected -22)\n", pid, ppid, ret);

    // (c) Valid case: reset resource limits for our pid.
    ret = syscall(SYS_RESOURCE_RESET, pid);
    printf("PID %d: sys_resource_reset(valid pid) returned: %d (expected 0)\n", pid, ret);

    // --- Tests for sys_deregister ---
    // (a) Negative case: pid < 1 --> expect -22
    ret = syscall(SYS_DEREGISTER, -1);
    printf("PID %d: sys_deregister(-1) returned: %d (expected -22)\n", pid, ret);

    // (b) Negative case: valid pid not in monitored list (using parent's pid) --> expect -3
    ret = syscall(SYS_DEREGISTER, ppid);
    printf("PID %d: sys_deregister(ppid=%d) returned: %d (expected -3)\n", pid, ppid, ret);

    // (c) Valid case: deregister our pid.
    ret = syscall(SYS_DEREGISTER, pid);
    printf("PID %d: sys_deregister(valid pid) returned: %d (expected 0)\n", pid, ret);



    // --- Cleanup: Close opened files ---
    for (int j = 0; j < 3; j++) {
        close(fds[j]);
    }
    close(fd4);

    // Free only the malloc-allocated memory.
    free(malloc_mem);
    free(malloc_mem_extra);
}

int main(void) {
    int num_processes = 5; // Adjust as needed
    pid_t pids[num_processes];

    for (int i = 0; i < num_processes; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            exit(EXIT_FAILURE);
        } else if (pid == 0) {
            // Child process
            if (setsid() < 0) {
                perror("setsid");
            }
            run_test(i);
            exit(EXIT_SUCCESS);
        } else {
            // Parent process: store child PID and stagger fork creation
            pids[i] = pid;
            sleep(1);
        }
    }

    // Parent waits for all children to complete.
    for (int i = 0; i < num_processes; i++) {
        int status;
        waitpid(pids[i], &status, 0);
        printf("Parent: Child process %d finished with status %d\n", pids[i], status);
    }

    return EXIT_SUCCESS;
}
