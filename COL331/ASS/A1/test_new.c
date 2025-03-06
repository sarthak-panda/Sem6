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
    pid_t pid;                   /* process id */
    unsigned long heapsize;      /* Total memory allocated by a process using brk and mmap */
    unsigned long openfile_count;/* Total number of open files using open, openat, openat2 */
    long heap_quota;             /* Heap quota in MB */
    long file_quota;             /* File quota */
    bool quotas_defined;         /* Indicates if quotas are defined */
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

    // 3. Allocate 150 KB of memory using brk (sbrk)
    void *current_break = sbrk(0);
    void *alloc1 = sbrk(15* 1024 * 1024);
    if (alloc1 == (void *)-1) {
        perror("sbrk for 15 MB failed");
        return EXIT_FAILURE;
    }
    printf("Allocated 15 MB of memory using brk (sbrk)\n");

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
    printf("Heap Size: %lu MB\n", stats.heapsize);
    printf("Open File Count: %lu\n", stats.openfile_count);
    printf("Heap Quota: %ld, File Quota: %ld, Quotas defined: %s\n",
           stats.heap_quota, stats.file_quota, stats.quotas_defined ? "true" : "false");

    // 6. Set resource limits: 1 MB heap and 5 open files.
    // Note: Since the quota is defined in MB, 1 MB is simply 1.
    ret = syscall(SYS_RESOURCE_CAP, pid, 25L, 5L);
    printf("\nsys_resource_cap (1 MB heap, 5 files) returned: %d\n", ret);

    // 7. Increase allocation: add 50 KB using brk (total becomes ~200 KB)
    void *alloc2 = sbrk(6 * 1024* 1024);
    if (alloc2 == (void *)-1) {
        perror("sbrk for additional 6 MB failed");
        return EXIT_FAILURE;
    }
    printf("Allocated additional 6 MB of memory using brk (total ~200 KB)\n");

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
    printf("Heap Size: %lu MB\n", stats.heapsize); 
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

    // Note: Memory allocated with sbrk cannot be freed using free()

    return EXIT_SUCCESS;
}
