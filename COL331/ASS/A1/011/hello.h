#ifndef HELLO_H
#define HELLO_H

#include <linux/list.h>
#include <linux/types.h>

struct pid_node {
    struct per_proc_resource* proc_resource;
    struct list_head next_prev_list;
};
struct per_proc_resource {
    pid_t pid;
    unsigned long heapsize;
    unsigned long openfile_count;
    long heap_quota;
    long file_quota;
    bool quotas_defined;
};

extern struct list_head monitored_pids_head_node;
extern struct mutex monitored_list_mutex;

#endif