SYSCALL_DEFINE2(fetch, struct per_proc_resource __user *, stats, pid_t, pid)
{
    struct pid_node *node;
    struct list_head *pos;
    struct per_proc_resource kstats = {0};
    bool found = false;
    if (!stats || !access_ok(stats, sizeof(struct per_proc_resource))) {
        return -22;
    }
    kstats.pid = pid;
    kstats.tgid = pid;
    kstats.heapsize = 0;
    kstats.openfile_count = 0;
    kstats.heap_quota = -1;
    kstats.file_quota = -1;
    kstats.quotas_defined = false;
    mutex_lock(&monitored_list_mutex);
    list_for_each(pos, &monitored_pids_head_node) {
        node = list_entry(pos, struct pid_node, next_prev_list);
        if (node->proc_resource->tgid == pid) {
            kstats.heapsize += node->proc_resource->heapsize;
            kstats.openfile_count += node->proc_resource->openfile_count;
            if (!found) {
                kstats.heap_quota = node->proc_resource->heap_quota;
                kstats.file_quota = node->proc_resource->file_quota;
                kstats.quotas_defined = node->proc_resource->quotas_defined;
                found = true;
            }
        }
    }
    mutex_unlock(&monitored_list_mutex);
    if (!found) {
        printk(KERN_INFO "sys_fetch: No monitored threads found for tgid %d\n", pid);
        return -22;
    }
    kstats.heapsize = kstats.heapsize / (1024 * 1024);
    if (copy_to_user(stats, &kstats, sizeof(struct per_proc_resource))) {
        printk(KERN_ERR "sys_fetch: copy_to_user failed for tgid %d\n", pid);
        return -22;
    }
    return 0;
}
