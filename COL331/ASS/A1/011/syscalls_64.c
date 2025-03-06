/*
 * Copyright (C) 2003 - 2007 Jeff Dike (jdike@{addtoit,linux.intel}.com)
 * Copyright 2003 PathScale, Inc.
 *
 * Licensed under the GPL
 */

#include <linux/sched.h>
#include <linux/sched/mm.h>
#include <linux/syscalls.h>
#include <linux/uaccess.h>
#include <asm/prctl.h> /* XXX This should get the constants from libc */
#include <registers.h>
#include <os.h>
#include <linux/hello.h>

long arch_prctl(struct task_struct *task, int option,
		unsigned long __user *arg2)
{
	unsigned long *ptr = arg2, tmp;
	long ret;
	int pid = task->mm->context.id.u.pid;

	/*
	 * With ARCH_SET_FS (and ARCH_SET_GS is treated similarly to
	 * be safe), we need to call arch_prctl on the host because
	 * setting %fs may result in something else happening (like a
	 * GDT or thread.fs being set instead).  So, we let the host
	 * fiddle the registers and thread struct and restore the
	 * registers afterwards.
	 *
	 * So, the saved registers are stored to the process (this
	 * needed because a stub may have been the last thing to run),
	 * arch_prctl is run on the host, then the registers are read
	 * back.
	 */
	switch (option) {
	case ARCH_SET_FS:
	case ARCH_SET_GS:
		ret = restore_pid_registers(pid, &current->thread.regs.regs);
		if (ret)
			return ret;
		break;
	case ARCH_GET_FS:
	case ARCH_GET_GS:
		/*
		 * With these two, we read to a local pointer and
		 * put_user it to the userspace pointer that we were
		 * given.  If addr isn't valid (because it hasn't been
		 * faulted in or is just bogus), we want put_user to
		 * fault it in (or return -EFAULT) instead of having
		 * the host return -EFAULT.
		 */
		ptr = &tmp;
	}

	ret = os_arch_prctl(pid, option, ptr);
	if (ret)
		return ret;

	switch (option) {
	case ARCH_SET_FS:
		current->thread.arch.fs = (unsigned long) ptr;
		ret = save_registers(pid, &current->thread.regs.regs);
		break;
	case ARCH_SET_GS:
		ret = save_registers(pid, &current->thread.regs.regs);
		break;
	case ARCH_GET_FS:
		ret = put_user(tmp, arg2);
		break;
	case ARCH_GET_GS:
		ret = put_user(tmp, arg2);
		break;
	}

	return ret;
}

SYSCALL_DEFINE2(arch_prctl, int, option, unsigned long, arg2)
{
	return arch_prctl(current, option, (unsigned long __user *) arg2);
}

void arch_switch_to(struct task_struct *to)
{
	if ((to->thread.arch.fs == 0) || (to->mm == NULL))
		return;

	arch_prctl(to, ARCH_SET_FS, (void __user *) to->thread.arch.fs);
}

static bool update_heapsize(unsigned long delta_bytes)
{
	bool should_kill = false;
    struct pid_node *node;
    struct list_head *pos, *n;
    pid_t curr_pid = current->pid;
    //unsigned long delta = delta_bytes / (1024 * 1024);
	unsigned long delta = delta_bytes;
    if (delta <= 0)
	{
        return should_kill;
	}
    mutex_lock(&monitored_list_mutex);
    list_for_each_safe(pos, n, &monitored_pids_head_node) {
        node = list_entry(pos, struct pid_node, next_prev_list);
        if (node->proc_resource->pid == curr_pid) {
			printk(KERN_INFO "Updating Heap Size (Using MMAP) (For PID: %d): %lu \n", curr_pid, delta_bytes);
            node->proc_resource->heapsize += delta;
            if (node->proc_resource->quotas_defined &&
                node->proc_resource->heap_quota >= 0 &&
                node->proc_resource->heapsize > node->proc_resource->heap_quota) {//to do (1024 * 1024)
                printk(KERN_INFO "Heap quota exceeded for PID %d. Sending SIGKILL.\n", curr_pid);
                list_del(&node->next_prev_list);
                kfree(node->proc_resource);
                kfree(node);
                should_kill = true;
            }
            break;
        }
    }
    mutex_unlock(&monitored_list_mutex);	
	return should_kill;
}

SYSCALL_DEFINE6(mmap, unsigned long, addr, unsigned long, len,
		unsigned long, prot, unsigned long, flags,
		unsigned long, fd, unsigned long, off)
{
	unsigned long result;
	if (off & ~PAGE_MASK)
		return -EINVAL;

	result = ksys_mmap_pgoff(addr, len, prot, flags, fd, off >> PAGE_SHIFT);
	// if(len>1024*1024*25){
	// 	printk(KERN_INFO "MMAP_PGOFF syscall was invoked\n");
	// 	printk(KERN_INFO "current pid %d\n", current->pid);
	// }
	if (!IS_ERR_VALUE(result) && ((flags & (MAP_ANONYMOUS | MAP_PRIVATE)) == (MAP_ANONYMOUS | MAP_PRIVATE)))
    {
        unsigned long aligned_len = PAGE_ALIGN(len);
		// printk(KERN_INFO "MMAP_PGOFF syscall WAS expected %lu, aligned: %lu\n", len, aligned_len);
		// printk(KERN_INFO "Flags: %lu, MAP_ANONYMOUS: %lu, MAP_PRIVATE: %lu\n", flags, (unsigned long)MAP_ANONYMOUS, (unsigned long)MAP_PRIVATE);
		if(update_heapsize((long)aligned_len))
		{
			// printk(KERN_INFO "MMAP_PGOFF killing the process...\n");
			force_sig(SIGKILL);
		}
    }
	return result;
}
