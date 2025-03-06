#include <linux/list.h>
#include <linux/types.h>
#include <linux/kernel.h>
#include <linux/rcupdate.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/syscalls.h>
#include <linux/sched.h>
#include <linux/uaccess.h>
#include <linux/hello.h>

LIST_HEAD(monitored_pids_head_node);
DEFINE_MUTEX(monitored_list_mutex);

SYSCALL_DEFINE0(hello)

{
    printk("I am linux modified....\n");
    return 0;
}

SYSCALL_DEFINE1(register, pid_t, pid)
{
    struct task_struct *task;
    struct pid_node *node;
    struct list_head *pos;

    printk(KERN_INFO "Syscall sys_register called with pid: %d\n", pid);
    if(pid<1){
        printk(KERN_INFO "Syscall sys_register: PID %d is less than 1 \n", pid);
        return -22;
    }

    rcu_read_lock();
    task = find_task_by_vpid(pid);
    rcu_read_unlock();
    if (!task) {
        printk(KERN_INFO "Syscall sys_register: PID %d does not exist\n", pid);
        return -3; 
    }

    //rcu_read_lock(); RCU is designed for read-mostly data structures
    mutex_lock(&monitored_list_mutex);//atomicity of possibly concurrent read and write
    list_for_each(pos, &monitored_pids_head_node){
        node = list_entry(pos, struct pid_node, next_prev_list);
        if (node->proc_resource->pid == pid) {
            //rcu_read_unlock();
            mutex_unlock(&monitored_list_mutex);
            printk(KERN_INFO "Syscall sys_register: PID %d is already monitored\n", pid);
            return -23;
        }
    }
    //rcu_read_unlock();
    //mutex_unlock(&monitored_list_mutex);

    //Node Creation

    node = kmalloc(sizeof(struct pid_node), GFP_KERNEL);//tries to get free pages in kernel of pid_node size
    if(!node){
        mutex_unlock(&monitored_list_mutex);
        printk(KERN_ERR "Syscall sys_register: Memory allocation failed for PID %d\n", pid);
        return -ENOMEM; 
    }

    node->proc_resource = kmalloc(sizeof(struct per_proc_resource), GFP_KERNEL);
    if (!node->proc_resource) {
        kfree(node);
        mutex_unlock(&monitored_list_mutex);
        printk(KERN_ERR "Syscall sys_register: Memory allocation failed for PID %d\n", pid);
        return -ENOMEM; 
    }

    node->proc_resource->pid = pid;
    //unsigned long heapsize=0;
    node->proc_resource->heapsize = 0;
    //unsigned long openfile_count = 0;
    node->proc_resource->openfile_count = 0;

    node->proc_resource->heap_quota = -1;
    node->proc_resource->file_quota = -1;
    node->proc_resource->quotas_defined = false;

    //let us calculate heap size, the problem is that t is static monitoring
    // struct mm_struct *mm = get_task_mm(task);
    // struct vm_area_struct *vma;
    // if (mm) {
    //     down_read(&mm->mmap_lock);
    //     //Heap from brk
    //     heapsize = mm->brk - mm->start_brk;
    //     up_read(&mm->mmap_lock);
    //     for_each_vma(mm, vma){
    //         if ((!(vma->vm_flags & VM_SHARED) && vma->vm_file == NULL)) {//(vma->vm_flags & VM_ANONYMOUS) && !(vma->vm_flags & VM_SHARED)
    //             if (vma->vm_start == mm->start_brk && vma->vm_end == mm->brk)
    //                 continue;
    //             heapsize += vma->vm_end - vma->vm_start;
    //         }
    //     }
    //     mmput(mm); 
    // }
    // node->proc_resource->heapsize=heapsize;

    //let us calculate number of open files
    
    list_add(&node->next_prev_list, &monitored_pids_head_node);
    mutex_unlock(&monitored_list_mutex);
    return 0;
}

SYSCALL_DEFINE2(fetch, struct per_proc_resource __user *, stats, pid_t, pid)
{
    struct pid_node *node;
    struct list_head *pos;
    struct per_proc_resource kstats;
    if (!stats || !access_ok(stats, sizeof(struct per_proc_resource))) {
        return -22;
    }
    mutex_lock(&monitored_list_mutex);
    list_for_each(pos, &monitored_pids_head_node) {
        node = list_entry(pos, struct pid_node, next_prev_list);
        if (node->proc_resource->pid == pid) {
            kstats = *node->proc_resource;
            mutex_unlock(&monitored_list_mutex);
            if (copy_to_user(stats, &kstats, sizeof(struct per_proc_resource))) {//copy_to_user reurns 0 on success
                printk(KERN_ERR "sys_fetch: copy_to_user failed for pid %d\n", pid);
                return -22;
            }
            return 0;
        }
    }
    mutex_unlock(&monitored_list_mutex);
    return -22;
}

SYSCALL_DEFINE1(deregister, pid_t, pid)
{
    bool found = false;
    struct pid_node *node;
    struct list_head *pos, *n;
    if(pid<1){
        //printk(KERN_INFO "Syscall sys_register: PID %d is less than 1 \n", pid);
        return -22;
    }
    mutex_lock(&monitored_list_mutex);
    list_for_each_safe(pos, n, &monitored_pids_head_node) {// allows deletion during traversal
        node = list_entry(pos, struct pid_node, next_prev_list);
        if (node->proc_resource->pid == pid) {
            list_del(&node->next_prev_list);    
            kfree(node->proc_resource);
            kfree(node);
            found=true;
            break;
        }
    }
    mutex_unlock(&monitored_list_mutex);
    return found ? 0 : -3;
}

SYSCALL_DEFINE3(resource_cap, pid_t, pid, long, heap_quota, long, file_quota) {
    struct task_struct *task;
    struct pid_node *node = NULL;
    struct list_head *pos, *n;
    bool should_kill = false;
    int found = 0;
    rcu_read_lock();
    task = find_task_by_vpid(pid);
    rcu_read_unlock();
    if (!task) {
        printk(KERN_INFO "Syscall sys_register: PID %d does not exist\n", pid);
        return -3;
    }
    mutex_lock(&monitored_list_mutex);
    list_for_each_safe(pos, n, &monitored_pids_head_node) {
        node = list_entry(pos, struct pid_node, next_prev_list);
        if (node->proc_resource->pid == pid) {
            found = 1;
            break;
        }
    }
    if (!found) {
        mutex_unlock(&monitored_list_mutex);
        return -22;
    }
    if (node->proc_resource->quotas_defined) {
        mutex_unlock(&monitored_list_mutex);
        return -23;
    }
    node->proc_resource->heap_quota = heap_quota;
    node->proc_resource->file_quota = file_quota;
    node->proc_resource->quotas_defined = true;

    //checking if it already exceeds the limit disscussion lik:
    if ((node->proc_resource->file_quota >= 0 && node->proc_resource->openfile_count > node->proc_resource->file_quota)
        ||(node->proc_resource->heap_quota >= 0 && node->proc_resource->heapsize > node->proc_resource->heap_quota)) 
    {
        printk(KERN_INFO "File/Heap quota exceeded for PID %d. Sending SIGKILL.\n", pid);
        list_del(&node->next_prev_list);
        kfree(node->proc_resource);
        kfree(node);
        should_kill = true;
    }
    mutex_unlock(&monitored_list_mutex);
    if(should_kill){
        printk(KERN_INFO "RES_CAP killing the process...\n");
        force_sig(SIGKILL);//--TO fix
    }
    return 0;    
}

SYSCALL_DEFINE1(resource_reset, pid_t, pid)
{
    struct task_struct *task;
    struct pid_node *node = NULL;
    struct list_head *pos;
    rcu_read_lock();
    task = find_task_by_vpid(pid);
    rcu_read_unlock();
    if (!task) {
        printk(KERN_INFO "sys_resource_reset: PID %d does not exist\n", pid);
        return -3;
    }
    mutex_lock(&monitored_list_mutex);
    list_for_each(pos, &monitored_pids_head_node) {
        node = list_entry(pos, struct pid_node, next_prev_list);
        if (node->proc_resource->pid == pid) {
            node->proc_resource->heap_quota = -1;
            node->proc_resource->file_quota = -1;
            node->proc_resource->quotas_defined = false;
            mutex_unlock(&monitored_list_mutex);
            printk(KERN_INFO "sys_resource_reset: Reset quotas for pid %d\n", pid);
            return 0;
        }
    }
    mutex_unlock(&monitored_list_mutex);

    return -22;
}
