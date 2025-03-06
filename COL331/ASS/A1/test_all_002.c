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

#define SYS_HELLO          451
#define SYS_REGISTER       452
#define SYS_FETCH          453
#define SYS_DEREGISTER     454
#define SYS_RESOURCE_CAP   455
#define SYS_RESOURCE_RESET 456

struct per_proc_resource {
    pid_t pid;                    
    unsigned long heapsize;       
    unsigned long openfile_count; 
    long heap_quota;              
    long file_quota;              
    bool quotas_defined;          
};

#define TEST_SYSCALL(call, desc) do {                  \
    errno = 0;                                         \
    ret = (call);                                      \
    if(ret == -1) {                                    \
        printf("PID %d: %s returned: %d, errno: %d (%s)\n", \
               pid, desc, ret, errno, strerror(errno));\
    } else {                                           \
        printf("PID %d: %s returned: %d\n", pid, desc, ret); \
    }                                                  \
} while(0)

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

    TEST_SYSCALL(syscall(SYS_HELLO), "sys_hello");

    TEST_SYSCALL(syscall(SYS_REGISTER, -1), "sys_register(-1) (expected -22)");

    TEST_SYSCALL(syscall(SYS_REGISTER, 99999), "sys_register(99999) (expected -3)");

    TEST_SYSCALL(syscall(SYS_REGISTER, pid), "sys_register(valid pid) (expected 0)");

    TEST_SYSCALL(syscall(SYS_REGISTER, pid), "sys_register(duplicate) (expected -23)");

    void *malloc_mem = allocate_mb(6);

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

    TEST_SYSCALL(syscall(SYS_RESOURCE_CAP, 99999, 7L, 5L),
                 "sys_resource_cap(99999, 7, 5) (expected -3)");

    pid_t ppid = getppid();
    TEST_SYSCALL(syscall(SYS_RESOURCE_CAP, ppid, 7L, 5L),
                 "sys_resource_cap(ppid, 7, 5) (expected -22)");

    TEST_SYSCALL(syscall(SYS_RESOURCE_CAP, pid, 15L, 6L),
                 "sys_resource_cap(valid pid, 15, 6) (expected 0)");

    TEST_SYSCALL(syscall(SYS_RESOURCE_CAP, pid, 9L, 7L),
                 "sys_resource_cap(duplicate cap) (expected -23)");

    struct per_proc_resource stats;
    TEST_SYSCALL(syscall(SYS_FETCH, &stats, pid), "sys_fetch");

    printf("PID %d: Process ID: %d\n", pid, stats.pid);
    printf("PID %d: Heap Size: %lu MB\n", pid, stats.heapsize);
    printf("PID %d: Open File Count: %lu\n", pid, stats.openfile_count);
    printf("PID %d: Heap Quota: %ld, File Quota: %ld, Quotas defined: %s\n",
           pid, stats.heap_quota, stats.file_quota,
           stats.quotas_defined ? "true" : "false");

    printf("\n----->PID %d: Doing Dummy Work (sleeping for %d seconds)<-----\n", pid, 8 + i);
    sleep(8 + i);
    printf("\n----->PID %d: Completed Dummy Work (waking up)<-----\n", pid);

    void *malloc_mem_extra = allocate_mb(2*(i+1));

    int fds_new[3];
    for (int j = 0; j < i+1; j++) {
        snprintf(filename, sizeof(filename), "new_dummy%d_%d.txt", pid, j+1);
        fds_new[j] = open(filename, O_RDWR | O_CREAT, 0666);
        if (fds_new[j] < 0) {
            perror("open");
            exit(EXIT_FAILURE);
        }
        printf("PID %d: Opened file: %s\n", pid, filename);
    }

    TEST_SYSCALL(syscall(SYS_FETCH, &stats, pid), "sys_fetch");

    printf("PID %d: Process ID: %d\n", pid, stats.pid);
    printf("PID %d: Heap Size: %lu MB\n", pid, stats.heapsize);
    printf("PID %d: Open File Count: %lu\n", pid, stats.openfile_count);
    printf("PID %d: Heap Quota: %ld, File Quota: %ld, Quotas defined: %s\n",
           pid, stats.heap_quota, stats.file_quota,
           stats.quotas_defined ? "true" : "false");

    TEST_SYSCALL(syscall(SYS_RESOURCE_RESET, 99999), "sys_resource_reset(99999) (expected -3)");

    TEST_SYSCALL(syscall(SYS_RESOURCE_RESET, ppid), "sys_resource_reset(ppid) (expected -22)");

    TEST_SYSCALL(syscall(SYS_RESOURCE_RESET, pid), "sys_resource_reset(valid pid) (expected 0)");

    TEST_SYSCALL(syscall(SYS_DEREGISTER, -1), "sys_deregister(-1) (expected -22)");

    TEST_SYSCALL(syscall(SYS_DEREGISTER, ppid), "sys_deregister(ppid) (expected -3)");

    TEST_SYSCALL(syscall(SYS_DEREGISTER, pid), "sys_deregister(valid pid) (expected 0)");

    for (int j = 0; j < 3; j++) {
        close(fds[j]);
    }
    for (int j = 0; j < i+1; j++) {
        close(fds_new[j]);
    }

    free(malloc_mem);
    free(malloc_mem_extra);
}

int main(void) {
    int num_processes = 5; 
    pid_t pids[num_processes];

    for (int i = 0; i < num_processes; i++) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            exit(EXIT_FAILURE);
        } else if (pid == 0) {

            if (setsid() < 0) {
                perror("setsid");
            }
            run_test(i);
            exit(EXIT_SUCCESS);
        } else {

            pids[i] = pid;
            sleep(1);
        }
    }

    for (int i = 0; i < num_processes; i++) {
        int status;
        waitpid(pids[i], &status, 0);
        printf("Parent: Child process %d finished with status %d\n", pids[i], status);
    }

    return EXIT_SUCCESS;
}