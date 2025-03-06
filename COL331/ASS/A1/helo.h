#ifndef HELLO_H
#define HELLO_H

#include <linux/list.h>
#include <linux/types.h>

struct pid_node {
    struct per_proc_resource* proc_resource; /* Resource
    utilization of a process */
    struct list_head next_prev_list; /* contains pointers
    to previous and next elements */
};
struct per_proc_resource {
    pid_t pid; /* process id */
    unsigned long heapsize; /* Total memory allocated by
    a process using the brk and mmap system calls. */
    unsigned long openfile_count; /* Total number of open
    files of a process using the open , openat , and
    openat2 system calls*/
    long heap_quota;
    long file_quota;
    bool quotas_defined;
};

extern struct list_head monitored_pids_head_node;
extern struct mutex monitored_list_mutex;

#endif