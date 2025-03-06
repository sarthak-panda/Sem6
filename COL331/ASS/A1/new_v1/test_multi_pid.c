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
    unsigned long heapsize;       /* Total memory allocated using brk and mmap */
    unsigned long openfile_count; /* Total number of open files */
    long heap_quota;              /* Heap quota in MB */
    long file_quota;              /* File quota */
    bool quotas_defined;          /* Indicates if quotas are defined */
};

void run_test(int i) {
    pid_t pid = getpid();
    int ret;

    // 1. Test sys_hello
    ret = syscall(SYS_HELLO);
    printf("PID %d: sys_hello returned: %d\n", pid, ret);

    // 2. Register the process for resource monitoring
    ret = syscall(SYS_REGISTER, pid);
    printf("PID %d: sys_register returned: %d\n", pid, ret);

    // 3. Allocate 5 MB of memory
    size_t size_5mb = 5 * 1024 * 1024;
    void *mem = malloc(size_5mb);
    if (!mem) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    memset(mem, 0, size_5mb);
    printf("PID %d: Allocated 5 MB of memory\n", pid);

    // 4. Open 3 dummy files (create if not exists)
    int fds[3];
    char filename[30];
    for (int j = 0; j < 3; j++) {
        snprintf(filename, sizeof(filename), "dummy%d_%d.txt", pid, j+1);
        fds[j] = open(filename, O_RDWR | O_CREAT, 0666);
        if (fds[j] < 0) {
            perror("open");
            exit(EXIT_FAILURE);
        }
        printf("PID %d: Opened file: %s\n", pid, filename);
    }

    // 5. Fetch and display current resource usage statistics
    struct per_proc_resource stats;
    ret = syscall(SYS_FETCH, &stats, pid);
    printf("\nPID %d: Initial sys_fetch returned: %d\n", pid, ret);
    printf("PID %d: Process ID: %d\n", pid, stats.pid);
    printf("PID %d: Heap Size: %lu MB\n", pid, stats.heapsize / (1024 * 1024));
    printf("PID %d: Open File Count: %lu\n", pid, stats.openfile_count);
    printf("PID %d: Heap Quota: %ld, File Quota: %ld, Quotas defined: %s\n",
           pid, stats.heap_quota, stats.file_quota, stats.quotas_defined ? "true" : "false");

    // 6. Set resource limits: 7 MB heap and 5 open files
    ret = syscall(SYS_RESOURCE_CAP, pid, 7 * 1024 * 1024L, 5L);
    printf("\nPID %d: sys_resource_cap (7 MB heap, 5 files) returned: %d\n", pid, ret);

    printf("\n----->PID %d: Doing Dummy Work (going to sleep)<-----\n",pid);
    sleep(5 + i);
    printf("\n----->PID %d: Completed Dummy Work (waking up)<-----\n",pid);
    // 7. Increase allocation: add 1 MB (total becomes ~6 MB)
    size_t size_delta = ((1+i) * 1024 * 1024)/2;
    void *mem_extra = malloc(size_delta);
    if (!mem_extra) {
        perror("malloc extra");
        exit(EXIT_FAILURE);
    }
    memset(mem_extra, 0, size_delta);
    printf("PID %d: Allocated additional %.2f MB of memory\n", pid, (i+1)/2.0);

    // Open one more file (4th file)
    snprintf(filename, sizeof(filename), "dummy4_%d.txt", pid);
    int fd4 = open(filename, O_RDWR | O_CREAT, 0666);
    if (fd4 < 0) {
        perror("open dummy4.txt");
        exit(EXIT_FAILURE);
    }
    printf("PID %d: Opened file: %s\n", pid, filename);

    // 8. Fetch and display updated resource usage statistics
    ret = syscall(SYS_FETCH, &stats, pid);
    printf("\nPID %d: After increase, sys_fetch returned: %d\n", pid, ret);
    printf("PID %d: Process ID: %d\n", pid, stats.pid);
    printf("PID %d: Heap Size: %.2f MB\n", pid, stats.heapsize / (1024.0 * 1024.0));
    printf("PID %d: Open File Count: %lu\n", pid, stats.openfile_count);
    printf("PID %d: Heap Quota: %ld, File Quota: %ld, Quotas defined: %s\n",
           pid, stats.heap_quota, stats.file_quota, stats.quotas_defined ? "true" : "false");

    // 9. Reset resource limits (quotas are set to -1)
    ret = syscall(SYS_RESOURCE_RESET, pid);
    printf("\nPID %d: sys_resource_reset returned: %d\n", pid, ret);

    // 10. Deregister the process from the monitored list
    ret = syscall(SYS_DEREGISTER, pid);
    printf("PID %d: sys_deregister returned: %d\n", pid, ret);

    // Clean up: Close opened files
    for (int i = 0; i < 3; i++) {
        close(fds[i]);
    }
    close(fd4);

    // Free allocated memory
    free(mem);
    free(mem_extra);
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
            // Optional: Create a new session to ensure this process is not in the parent's process group.
            if (setsid() < 0) {
                perror("setsid");
            }
            // Stagger execution: each child sleeps for a different amount of time.
            // For example, the first child waits 5 sec, the second 6 sec, etc.
            
            run_test(i);
            exit(EXIT_SUCCESS);
        } else {
            // Parent process: store child PID and optionally wait a second before forking the next child.
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
