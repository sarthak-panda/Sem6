#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>

// Define syscall numbers as per your setup
#define SYS_HELLO          451
#define SYS_REGISTER       452
#define SYS_FETCH          453
#define SYS_DEREGISTER     454
#define SYS_RESOURCE_CAP   455
#define SYS_RESOURCE_RESET 456

// Updated per_proc_resource structure
struct per_proc_resource {
    pid_t pid;                  /* process id */
    unsigned long heapsize;     /* Total memory allocated by a process using brk and mmap */
    unsigned long openfile_count; /* Total number of open files using open, openat, openat2 */
    long heap_quota;            /* Heap quota in MB */
    long file_quota;            /* File quota */
    bool quotas_defined;        /* Indicates if quotas are defined */
};

int main(void) {
    pid_t pid = getpid();
    int ret;

    // 1. Test sys_hello
    ret = syscall(SYS_HELLO);
    printf("sys_hello returned: %d\n", ret);

    // 2. Register the process for resource monitoring
    ret = syscall(SYS_REGISTER, pid);
    printf("sys_register (pid %d) returned: %d\n", pid, ret);

    // 3. Allocate 5 MB of memory
    size_t size_5mb = 5 * 1024 * 1024;
    void *mem = malloc(size_5mb);
    if (!mem) {
        perror("malloc");
        return EXIT_FAILURE;
    }
    memset(mem, 0, size_5mb);
    printf("Allocated 5 MB of memory\n");

    // 4. Open 3 dummy files (create if not exists)
    int fds[3];
    char filename[20];
    for (int i = 0; i < 3; i++) {
        snprintf(filename, sizeof(filename), "dummy%d.txt", i+1);
        fds[i] = open(filename, O_RDWR | O_CREAT, 0666);
        if (fds[i] < 0) {
            perror("open");
            return EXIT_FAILURE;
        }
        printf("Opened file: %s\n", filename);
    }

    // 5. Fetch and display current resource usage statistics
    struct per_proc_resource stats;
    ret = syscall(SYS_FETCH, &stats, pid);
    printf("\nInitial sys_fetch returned: %d\n", ret);
    printf("Process ID: %d\n", stats.pid);
    printf("Heap Size: %lu MB\n", stats.heapsize / (1024 * 1024));  // Convert bytes to MB
    printf("Open File Count: %lu\n", stats.openfile_count);
    printf("Heap Quota: %ld, File Quota: %ld, Quotas defined: %s\n",
           stats.heap_quota, stats.file_quota, stats.quotas_defined ? "true" : "false");

    // 6. Set resource limits: 7 MB heap and 5 open files
    ret = syscall(SYS_RESOURCE_CAP, pid, 7*1024*1024L, 5L);
    printf("\nsys_resource_cap (7 MB heap, 5 files) returned: %d\n", ret);

    // 7. Increase allocation: add 1 MB (total becomes ~6 MB)
    size_t size_1mb = 1 * 1024 * 1024;
    void *mem_extra = malloc(size_1mb);
    if (!mem_extra) {
        perror("malloc extra");
        return EXIT_FAILURE;
    }
    memset(mem_extra, 0, size_1mb);
    printf("Allocated additional 1 MB of memory (total ~6 MB)\n");

    // Open one more file (4th file)
    snprintf(filename, sizeof(filename), "dummy4.txt");
    int fd4 = open(filename, O_RDWR | O_CREAT, 0666);
    if (fd4 < 0) {
        perror("open dummy4.txt");
        return EXIT_FAILURE;
    }
    printf("Opened file: %s\n", filename);

    // 8. Fetch and display updated resource usage statistics
    ret = syscall(SYS_FETCH, &stats, pid);
    printf("\nAfter increase, sys_fetch returned: %d\n", ret);
    printf("Process ID: %d\n", stats.pid);
    printf("Heap Size: %lu MB\n", stats.heapsize / (1024 * 1024));  // Convert bytes to MB
    printf("Open File Count: %lu\n", stats.openfile_count);
    printf("Heap Quota: %ld, File Quota: %ld, Quotas defined: %s\n",
           stats.heap_quota, stats.file_quota, stats.quotas_defined ? "true" : "false");

    // 9. Reset resource limits (quotas are set to -1)
    ret = syscall(SYS_RESOURCE_RESET, pid);
    printf("\nsys_resource_reset returned: %d\n", ret);

    // 10. Deregister the process from the monitored list
    ret = syscall(SYS_DEREGISTER, pid);
    printf("sys_deregister returned: %d\n", ret);

    // Clean up: Close opened files
    for (int i = 0; i < 3; i++) {
        close(fds[i]);
    }
    close(fd4);

    // Free allocated memory
    free(mem);
    free(mem_extra);

    return EXIT_SUCCESS;
}
