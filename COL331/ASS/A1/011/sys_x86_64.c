#include <linux/compat.h>
#include <linux/errno.h>
#include <linux/sched.h>
#include <linux/sched/mm.h>
#include <linux/syscalls.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/smp.h>
#include <linux/sem.h>
#include <linux/msg.h>
#include <linux/shm.h>
#include <linux/stat.h>
#include <linux/mman.h>
#include <linux/file.h>
#include <linux/utsname.h>
#include <linux/personality.h>
#include <linux/random.h>
#include <linux/uaccess.h>
#include <linux/elf.h>
#include <linux/hello.h>
#include <asm/elf.h>
#include <asm/ia32.h>
static unsigned long get_align_mask(void)
{
	if (va_align.flags < 0 || !(va_align.flags & (2 - mmap_is_ia32())))
		return 0;
	if (!(current->flags & PF_RANDOMIZE))
		return 0;
	return va_align.mask;
}
static unsigned long get_align_bits(void)
{
	return va_align.bits & get_align_mask();
}
unsigned long align_vdso_addr(unsigned long addr)
{
	unsigned long align_mask = get_align_mask();
	addr = (addr + align_mask) & ~align_mask;
	return addr | get_align_bits();
}
static int __init control_va_addr_alignment(char *str)
{
	if (va_align.flags < 0)
		return 1;
	if (*str == 0)
		return 1;
	if (!strcmp(str, "32"))
		va_align.flags = ALIGN_VA_32;
	else if (!strcmp(str, "64"))
		va_align.flags = ALIGN_VA_64;
	else if (!strcmp(str, "off"))
		va_align.flags = 0;
	else if (!strcmp(str, "on"))
		va_align.flags = ALIGN_VA_32 | ALIGN_VA_64;
	else
		pr_warn("invalid option value: 'align_va_addr=%s'\n", str);
	return 1;
}
__setup("align_va_addr=", control_va_addr_alignment);
static bool update_heapsize(unsigned long delta_bytes)
{
	bool should_kill = false;
    struct pid_node *node;
    struct list_head *pos, *n;
    pid_t curr_pid = current->pid;
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
                node->proc_resource->heapsize > node->proc_resource->heap_quota) {
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
	if (!IS_ERR_VALUE(result) && ((flags & (MAP_ANONYMOUS | MAP_PRIVATE)) == (MAP_ANONYMOUS | MAP_PRIVATE)))
    {
        unsigned long aligned_len = PAGE_ALIGN(len);
		if(update_heapsize((long)aligned_len))
		{
			force_sig(SIGKILL);
		}
    }
	return result;
}
static void find_start_end(unsigned long addr, unsigned long flags,
		unsigned long *begin, unsigned long *end)
{
	if (!in_32bit_syscall() && (flags & MAP_32BIT)) {
		*begin = 0x40000000;
		*end = 0x80000000;
		if (current->flags & PF_RANDOMIZE) {
			*begin = randomize_page(*begin, 0x02000000);
		}
		return;
	}
	*begin	= get_mmap_base(1);
	if (in_32bit_syscall())
		*end = task_size_32bit();
	else
		*end = task_size_64bit(addr > DEFAULT_MAP_WINDOW);
}
unsigned long
arch_get_unmapped_area(struct file *filp, unsigned long addr,
		unsigned long len, unsigned long pgoff, unsigned long flags)
{
	struct mm_struct *mm = current->mm;
	struct vm_area_struct *vma;
	struct vm_unmapped_area_info info;
	unsigned long begin, end;
	if (flags & MAP_FIXED)
		return addr;
	find_start_end(addr, flags, &begin, &end);
	if (len > end)
		return -ENOMEM;
	if (addr) {
		addr = PAGE_ALIGN(addr);
		vma = find_vma(mm, addr);
		if (end - len >= addr &&
		    (!vma || addr + len <= vm_start_gap(vma)))
			return addr;
	}
	info.flags = 0;
	info.length = len;
	info.low_limit = begin;
	info.high_limit = end;
	info.align_mask = 0;
	info.align_offset = pgoff << PAGE_SHIFT;
	if (filp) {
		info.align_mask = get_align_mask();
		info.align_offset += get_align_bits();
	}
	return vm_unmapped_area(&info);
}
unsigned long
arch_get_unmapped_area_topdown(struct file *filp, const unsigned long addr0,
			  const unsigned long len, const unsigned long pgoff,
			  const unsigned long flags)
{
	struct vm_area_struct *vma;
	struct mm_struct *mm = current->mm;
	unsigned long addr = addr0;
	struct vm_unmapped_area_info info;
	if (len > TASK_SIZE)
		return -ENOMEM;
	if (flags & MAP_FIXED)
		return addr;
	if (!in_32bit_syscall() && (flags & MAP_32BIT))
		goto bottomup;
	if (addr) {
		addr &= PAGE_MASK;
		if (!mmap_address_hint_valid(addr, len))
			goto get_unmapped_area;
		vma = find_vma(mm, addr);
		if (!vma || addr + len <= vm_start_gap(vma))
			return addr;
	}
get_unmapped_area:
	info.flags = VM_UNMAPPED_AREA_TOPDOWN;
	info.length = len;
	info.low_limit = PAGE_SIZE;
	info.high_limit = get_mmap_base(0);
	if (addr > DEFAULT_MAP_WINDOW && !in_32bit_syscall())
		info.high_limit += TASK_SIZE_MAX - DEFAULT_MAP_WINDOW;
	info.align_mask = 0;
	info.align_offset = pgoff << PAGE_SHIFT;
	if (filp) {
		info.align_mask = get_align_mask();
		info.align_offset += get_align_bits();
	}
	addr = vm_unmapped_area(&info);
	if (!(addr & ~PAGE_MASK))
		return addr;
	VM_BUG_ON(addr != -ENOMEM);
bottomup:
	return arch_get_unmapped_area(filp, addr0, len, pgoff, flags);
}
