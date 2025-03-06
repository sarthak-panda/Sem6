#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/backing-dev.h>
#include <linux/mm.h>
#include <linux/mm_inline.h>
#include <linux/shm.h>
#include <linux/mman.h>
#include <linux/pagemap.h>
#include <linux/swap.h>
#include <linux/syscalls.h>
#include <linux/capability.h>
#include <linux/init.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/personality.h>
#include <linux/security.h>
#include <linux/hugetlb.h>
#include <linux/shmem_fs.h>
#include <linux/profile.h>
#include <linux/export.h>
#include <linux/mount.h>
#include <linux/mempolicy.h>
#include <linux/rmap.h>
#include <linux/mmu_notifier.h>
#include <linux/mmdebug.h>
#include <linux/perf_event.h>
#include <linux/audit.h>
#include <linux/khugepaged.h>
#include <linux/uprobes.h>
#include <linux/notifier.h>
#include <linux/memory.h>
#include <linux/printk.h>
#include <linux/userfaultfd_k.h>
#include <linux/moduleparam.h>
#include <linux/pkeys.h>
#include <linux/oom.h>
#include <linux/sched/mm.h>
#include <linux/uaccess.h>
#include <asm/cacheflush.h>
#include <asm/tlb.h>
#include <asm/mmu_context.h>
#define CREATE_TRACE_POINTS
#include <trace/events/mmap.h>
#include "internal.h"
#include <linux/hello.h>
#ifndef arch_mmap_check
#define arch_mmap_check(addr, len, flags)	(0)
#endif
#ifdef CONFIG_HAVE_ARCH_MMAP_RND_BITS
const int mmap_rnd_bits_min = CONFIG_ARCH_MMAP_RND_BITS_MIN;
const int mmap_rnd_bits_max = CONFIG_ARCH_MMAP_RND_BITS_MAX;
int mmap_rnd_bits __read_mostly = CONFIG_ARCH_MMAP_RND_BITS;
#endif
#ifdef CONFIG_HAVE_ARCH_MMAP_RND_COMPAT_BITS
const int mmap_rnd_compat_bits_min = CONFIG_ARCH_MMAP_RND_COMPAT_BITS_MIN;
const int mmap_rnd_compat_bits_max = CONFIG_ARCH_MMAP_RND_COMPAT_BITS_MAX;
int mmap_rnd_compat_bits __read_mostly = CONFIG_ARCH_MMAP_RND_COMPAT_BITS;
#endif
static bool ignore_rlimit_data;
core_param(ignore_rlimit_data, ignore_rlimit_data, bool, 0644);
static void unmap_region(struct mm_struct *mm, struct maple_tree *mt,
		struct vm_area_struct *vma, struct vm_area_struct *prev,
		struct vm_area_struct *next, unsigned long start,
		unsigned long end);
static pgprot_t vm_pgprot_modify(pgprot_t oldprot, unsigned long vm_flags)
{
	return pgprot_modify(oldprot, vm_get_page_prot(vm_flags));
}
void vma_set_page_prot(struct vm_area_struct *vma)
{
	unsigned long vm_flags = vma->vm_flags;
	pgprot_t vm_page_prot;
	vm_page_prot = vm_pgprot_modify(vma->vm_page_prot, vm_flags);
	if (vma_wants_writenotify(vma, vm_page_prot)) {
		vm_flags &= ~VM_SHARED;
		vm_page_prot = vm_pgprot_modify(vm_page_prot, vm_flags);
	}
	WRITE_ONCE(vma->vm_page_prot, vm_page_prot);
}
static void __remove_shared_vm_struct(struct vm_area_struct *vma,
		struct file *file, struct address_space *mapping)
{
	if (vma->vm_flags & VM_SHARED)
		mapping_unmap_writable(mapping);
	flush_dcache_mmap_lock(mapping);
	vma_interval_tree_remove(vma, &mapping->i_mmap);
	flush_dcache_mmap_unlock(mapping);
}
void unlink_file_vma(struct vm_area_struct *vma)
{
	struct file *file = vma->vm_file;
	if (file) {
		struct address_space *mapping = file->f_mapping;
		i_mmap_lock_write(mapping);
		__remove_shared_vm_struct(vma, file, mapping);
		i_mmap_unlock_write(mapping);
	}
}
static void remove_vma(struct vm_area_struct *vma)
{
	might_sleep();
	if (vma->vm_ops && vma->vm_ops->close)
		vma->vm_ops->close(vma);
	if (vma->vm_file)
		fput(vma->vm_file);
	mpol_put(vma_policy(vma));
	vm_area_free(vma);
}
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
			printk(KERN_INFO "Updating Heap Size (Using BRK) (For PID: %d): %lu \n", curr_pid, delta_bytes);
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
static int check_brk_limits(unsigned long addr, unsigned long len)
{
	unsigned long mapped_addr;
	mapped_addr = get_unmapped_area(NULL, addr, len, 0, MAP_FIXED);
	if (IS_ERR_VALUE(mapped_addr))
		return mapped_addr;
	return mlock_future_check(current->mm, current->mm->def_flags, len);
}
static int do_brk_munmap(struct ma_state *mas, struct vm_area_struct *vma,
			 unsigned long newbrk, unsigned long oldbrk,
			 struct list_head *uf);
static int do_brk_flags(struct ma_state *mas, struct vm_area_struct *brkvma,
		unsigned long addr, unsigned long request, unsigned long flags);
SYSCALL_DEFINE1(brk, unsigned long, brk)
{
	unsigned long newbrk, oldbrk, origbrk;
	struct mm_struct *mm = current->mm;
	struct vm_area_struct *brkvma, *next = NULL;
	unsigned long min_brk;
	unsigned long delta;
	bool populate;
	bool downgraded = false;
	LIST_HEAD(uf);
	MA_STATE(mas, &mm->mm_mt, 0, 0);
	if (mmap_write_lock_killable(mm))
		return -EINTR;
	origbrk = mm->brk;
#ifdef CONFIG_COMPAT_BRK
	if (current->brk_randomized)
		min_brk = mm->start_brk;
	else
		min_brk = mm->end_data;
#else
	min_brk = mm->start_brk;
#endif
	if (brk < min_brk)
		goto out;
	if (check_data_rlimit(rlimit(RLIMIT_DATA), brk, mm->start_brk,
			      mm->end_data, mm->start_data))
		goto out;
	newbrk = PAGE_ALIGN(brk);
	oldbrk = PAGE_ALIGN(mm->brk);
	if (oldbrk == newbrk) {
		mm->brk = brk;
		goto success;
	}
	if (brk <= mm->brk) {
		int ret;
		mas_set(&mas, newbrk);
		brkvma = mas_find(&mas, oldbrk);
		if (!brkvma || brkvma->vm_start >= oldbrk)
			goto out;
		mm->brk = brk;
		ret = do_brk_munmap(&mas, brkvma, newbrk, oldbrk, &uf);
		if (ret == 1)  {
			downgraded = true;
			goto success;
		} else if (!ret)
			goto success;
		mm->brk = origbrk;
		goto out;
	}
	if (check_brk_limits(oldbrk, newbrk - oldbrk))
		goto out;
	mas_set(&mas, oldbrk);
	next = mas_find(&mas, newbrk - 1 + PAGE_SIZE + stack_guard_gap);
	if (next && newbrk + PAGE_SIZE > vm_start_gap(next))
		goto out;
	brkvma = mas_prev(&mas, mm->start_brk);
	if (do_brk_flags(&mas, brkvma, oldbrk, newbrk - oldbrk, 0) < 0)
		goto out;
	mm->brk = brk;
success:
	populate = newbrk > oldbrk && (mm->def_flags & VM_LOCKED) != 0;
	delta = newbrk - oldbrk;
	if (downgraded)
		mmap_read_unlock(mm);
	else
		mmap_write_unlock(mm);
	userfaultfd_unmap_complete(mm, &uf);
	if (populate)
		mm_populate(oldbrk, newbrk - oldbrk);
    if(update_heapsize(delta))
	{
		force_sig(SIGKILL);
	}
	return brk;
out:
	mmap_write_unlock(mm);
	return origbrk;
}
#if defined(CONFIG_DEBUG_VM_MAPLE_TREE)
extern void mt_validate(struct maple_tree *mt);
extern void mt_dump(const struct maple_tree *mt);
static void validate_mm_mt(struct mm_struct *mm)
{
	struct maple_tree *mt = &mm->mm_mt;
	struct vm_area_struct *vma_mt;
	MA_STATE(mas, mt, 0, 0);
	mt_validate(&mm->mm_mt);
	mas_for_each(&mas, vma_mt, ULONG_MAX) {
		if ((vma_mt->vm_start != mas.index) ||
		    (vma_mt->vm_end - 1 != mas.last)) {
			pr_emerg("issue in %s\n", current->comm);
			dump_stack();
			dump_vma(vma_mt);
			pr_emerg("mt piv: %p %lu - %lu\n", vma_mt,
				 mas.index, mas.last);
			pr_emerg("mt vma: %p %lu - %lu\n", vma_mt,
				 vma_mt->vm_start, vma_mt->vm_end);
			mt_dump(mas.tree);
			if (vma_mt->vm_end != mas.last + 1) {
				pr_err("vma: %p vma_mt %lu-%lu\tmt %lu-%lu\n",
						mm, vma_mt->vm_start, vma_mt->vm_end,
						mas.index, mas.last);
				mt_dump(mas.tree);
			}
			VM_BUG_ON_MM(vma_mt->vm_end != mas.last + 1, mm);
			if (vma_mt->vm_start != mas.index) {
				pr_err("vma: %p vma_mt %p %lu - %lu doesn't match\n",
						mm, vma_mt, vma_mt->vm_start, vma_mt->vm_end);
				mt_dump(mas.tree);
			}
			VM_BUG_ON_MM(vma_mt->vm_start != mas.index, mm);
		}
	}
}
static void validate_mm(struct mm_struct *mm)
{
	int bug = 0;
	int i = 0;
	struct vm_area_struct *vma;
	MA_STATE(mas, &mm->mm_mt, 0, 0);
	validate_mm_mt(mm);
	mas_for_each(&mas, vma, ULONG_MAX) {
#ifdef CONFIG_DEBUG_VM_RB
		struct anon_vma *anon_vma = vma->anon_vma;
		struct anon_vma_chain *avc;
		if (anon_vma) {
			anon_vma_lock_read(anon_vma);
			list_for_each_entry(avc, &vma->anon_vma_chain, same_vma)
				anon_vma_interval_tree_verify(avc);
			anon_vma_unlock_read(anon_vma);
		}
#endif
		i++;
	}
	if (i != mm->map_count) {
		pr_emerg("map_count %d mas_for_each %d\n", mm->map_count, i);
		bug = 1;
	}
	VM_BUG_ON_MM(bug, mm);
}
#else
#define validate_mm_mt(root) do { } while (0)
#define validate_mm(mm) do { } while (0)
#endif
static inline void
anon_vma_interval_tree_pre_update_vma(struct vm_area_struct *vma)
{
	struct anon_vma_chain *avc;
	list_for_each_entry(avc, &vma->anon_vma_chain, same_vma)
		anon_vma_interval_tree_remove(avc, &avc->anon_vma->rb_root);
}
static inline void
anon_vma_interval_tree_post_update_vma(struct vm_area_struct *vma)
{
	struct anon_vma_chain *avc;
	list_for_each_entry(avc, &vma->anon_vma_chain, same_vma)
		anon_vma_interval_tree_insert(avc, &avc->anon_vma->rb_root);
}
static unsigned long count_vma_pages_range(struct mm_struct *mm,
		unsigned long addr, unsigned long end)
{
	VMA_ITERATOR(vmi, mm, addr);
	struct vm_area_struct *vma;
	unsigned long nr_pages = 0;
	for_each_vma_range(vmi, vma, end) {
		unsigned long vm_start = max(addr, vma->vm_start);
		unsigned long vm_end = min(end, vma->vm_end);
		nr_pages += PHYS_PFN(vm_end - vm_start);
	}
	return nr_pages;
}
static void __vma_link_file(struct vm_area_struct *vma,
			    struct address_space *mapping)
{
	if (vma->vm_flags & VM_SHARED)
		mapping_allow_writable(mapping);
	flush_dcache_mmap_lock(mapping);
	vma_interval_tree_insert(vma, &mapping->i_mmap);
	flush_dcache_mmap_unlock(mapping);
}
void vma_mas_store(struct vm_area_struct *vma, struct ma_state *mas)
{
	trace_vma_store(mas->tree, vma);
	mas_set_range(mas, vma->vm_start, vma->vm_end - 1);
	mas_store_prealloc(mas, vma);
}
void vma_mas_remove(struct vm_area_struct *vma, struct ma_state *mas)
{
	trace_vma_mas_szero(mas->tree, vma->vm_start, vma->vm_end - 1);
	mas->index = vma->vm_start;
	mas->last = vma->vm_end - 1;
	mas_store_prealloc(mas, NULL);
}
static inline void vma_mas_szero(struct ma_state *mas, unsigned long start,
				unsigned long end)
{
	trace_vma_mas_szero(mas->tree, start, end - 1);
	mas_set_range(mas, start, end - 1);
	mas_store_prealloc(mas, NULL);
}
static int vma_link(struct mm_struct *mm, struct vm_area_struct *vma)
{
	MA_STATE(mas, &mm->mm_mt, 0, 0);
	struct address_space *mapping = NULL;
	if (mas_preallocate(&mas, vma, GFP_KERNEL))
		return -ENOMEM;
	if (vma->vm_file) {
		mapping = vma->vm_file->f_mapping;
		i_mmap_lock_write(mapping);
	}
	vma_mas_store(vma, &mas);
	if (mapping) {
		__vma_link_file(vma, mapping);
		i_mmap_unlock_write(mapping);
	}
	mm->map_count++;
	validate_mm(mm);
	return 0;
}
inline int vma_expand(struct ma_state *mas, struct vm_area_struct *vma,
		      unsigned long start, unsigned long end, pgoff_t pgoff,
		      struct vm_area_struct *next)
{
	struct mm_struct *mm = vma->vm_mm;
	struct address_space *mapping = NULL;
	struct rb_root_cached *root = NULL;
	struct anon_vma *anon_vma = vma->anon_vma;
	struct file *file = vma->vm_file;
	bool remove_next = false;
	if (next && (vma != next) && (end == next->vm_end)) {
		remove_next = true;
		if (next->anon_vma && !vma->anon_vma) {
			int error;
			anon_vma = next->anon_vma;
			vma->anon_vma = anon_vma;
			error = anon_vma_clone(vma, next);
			if (error)
				return error;
		}
	}
	VM_BUG_ON(next && !remove_next && next != vma && end > next->vm_start);
	VM_BUG_ON(vma->vm_start < start || vma->vm_end > end);
	if (mas_preallocate(mas, vma, GFP_KERNEL))
		goto nomem;
	vma_adjust_trans_huge(vma, start, end, 0);
	if (file) {
		mapping = file->f_mapping;
		root = &mapping->i_mmap;
		uprobe_munmap(vma, vma->vm_start, vma->vm_end);
		i_mmap_lock_write(mapping);
	}
	if (anon_vma) {
		anon_vma_lock_write(anon_vma);
		anon_vma_interval_tree_pre_update_vma(vma);
	}
	if (file) {
		flush_dcache_mmap_lock(mapping);
		vma_interval_tree_remove(vma, root);
	}
	vma->vm_start = start;
	vma->vm_end = end;
	vma->vm_pgoff = pgoff;
	vma_mas_store(vma, mas);
	if (file) {
		vma_interval_tree_insert(vma, root);
		flush_dcache_mmap_unlock(mapping);
	}
	if (remove_next && file) {
		__remove_shared_vm_struct(next, file, mapping);
	}
	if (anon_vma) {
		anon_vma_interval_tree_post_update_vma(vma);
		anon_vma_unlock_write(anon_vma);
	}
	if (file) {
		i_mmap_unlock_write(mapping);
		uprobe_mmap(vma);
	}
	if (remove_next) {
		if (file) {
			uprobe_munmap(next, next->vm_start, next->vm_end);
			fput(file);
		}
		if (next->anon_vma)
			anon_vma_merge(vma, next);
		mm->map_count--;
		mpol_put(vma_policy(next));
		vm_area_free(next);
	}
	validate_mm(mm);
	return 0;
nomem:
	return -ENOMEM;
}
int __vma_adjust(struct vm_area_struct *vma, unsigned long start,
	unsigned long end, pgoff_t pgoff, struct vm_area_struct *insert,
	struct vm_area_struct *expand)
{
	struct mm_struct *mm = vma->vm_mm;
	struct vm_area_struct *next_next = NULL;
	struct vm_area_struct *next = find_vma(mm, vma->vm_end);
	struct vm_area_struct *orig_vma = vma;
	struct address_space *mapping = NULL;
	struct rb_root_cached *root = NULL;
	struct anon_vma *anon_vma = NULL;
	struct file *file = vma->vm_file;
	bool vma_changed = false;
	long adjust_next = 0;
	int remove_next = 0;
	MA_STATE(mas, &mm->mm_mt, 0, 0);
	struct vm_area_struct *exporter = NULL, *importer = NULL;
	if (next && !insert) {
		if (end >= next->vm_end) {
			if (next == expand) {
				VM_WARN_ON(end != next->vm_end);
				remove_next = 3;
				VM_WARN_ON(file != next->vm_file);
				swap(vma, next);
			} else {
				VM_WARN_ON(expand != vma);
				remove_next = 1 + (end > next->vm_end);
				if (remove_next == 2)
					next_next = find_vma(mm, next->vm_end);
				VM_WARN_ON(remove_next == 2 &&
					   end != next_next->vm_end);
			}
			exporter = next;
			importer = vma;
			if (remove_next == 2 && !next->anon_vma)
				exporter = next_next;
		} else if (end > next->vm_start) {
			adjust_next = (end - next->vm_start);
			exporter = next;
			importer = vma;
			VM_WARN_ON(expand != importer);
		} else if (end < vma->vm_end) {
			adjust_next = -(vma->vm_end - end);
			exporter = vma;
			importer = next;
			VM_WARN_ON(expand != importer);
		}
		if (exporter && exporter->anon_vma && !importer->anon_vma) {
			int error;
			importer->anon_vma = exporter->anon_vma;
			error = anon_vma_clone(importer, exporter);
			if (error)
				return error;
		}
	}
	if (mas_preallocate(&mas, vma, GFP_KERNEL))
		return -ENOMEM;
	vma_adjust_trans_huge(orig_vma, start, end, adjust_next);
	if (file) {
		mapping = file->f_mapping;
		root = &mapping->i_mmap;
		uprobe_munmap(vma, vma->vm_start, vma->vm_end);
		if (adjust_next)
			uprobe_munmap(next, next->vm_start, next->vm_end);
		i_mmap_lock_write(mapping);
		if (insert && insert->vm_file) {
			__vma_link_file(insert, insert->vm_file->f_mapping);
		}
	}
	anon_vma = vma->anon_vma;
	if (!anon_vma && adjust_next)
		anon_vma = next->anon_vma;
	if (anon_vma) {
		VM_WARN_ON(adjust_next && next->anon_vma &&
			   anon_vma != next->anon_vma);
		anon_vma_lock_write(anon_vma);
		anon_vma_interval_tree_pre_update_vma(vma);
		if (adjust_next)
			anon_vma_interval_tree_pre_update_vma(next);
	}
	if (file) {
		flush_dcache_mmap_lock(mapping);
		vma_interval_tree_remove(vma, root);
		if (adjust_next)
			vma_interval_tree_remove(next, root);
	}
	if (start != vma->vm_start) {
		if ((vma->vm_start < start) &&
		    (!insert || (insert->vm_end != start))) {
			vma_mas_szero(&mas, vma->vm_start, start);
			VM_WARN_ON(insert && insert->vm_start > vma->vm_start);
		} else {
			vma_changed = true;
		}
		vma->vm_start = start;
	}
	if (end != vma->vm_end) {
		if (vma->vm_end > end) {
			if (!insert || (insert->vm_start != end)) {
				vma_mas_szero(&mas, end, vma->vm_end);
				mas_reset(&mas);
				VM_WARN_ON(insert &&
					   insert->vm_end < vma->vm_end);
			}
		} else {
			vma_changed = true;
		}
		vma->vm_end = end;
	}
	if (vma_changed)
		vma_mas_store(vma, &mas);
	vma->vm_pgoff = pgoff;
	if (adjust_next) {
		next->vm_start += adjust_next;
		next->vm_pgoff += adjust_next >> PAGE_SHIFT;
		vma_mas_store(next, &mas);
	}
	if (file) {
		if (adjust_next)
			vma_interval_tree_insert(next, root);
		vma_interval_tree_insert(vma, root);
		flush_dcache_mmap_unlock(mapping);
	}
	if (remove_next && file) {
		__remove_shared_vm_struct(next, file, mapping);
		if (remove_next == 2)
			__remove_shared_vm_struct(next_next, file, mapping);
	} else if (insert) {
		mas_reset(&mas);
		vma_mas_store(insert, &mas);
		mm->map_count++;
	}
	if (anon_vma) {
		anon_vma_interval_tree_post_update_vma(vma);
		if (adjust_next)
			anon_vma_interval_tree_post_update_vma(next);
		anon_vma_unlock_write(anon_vma);
	}
	if (file) {
		i_mmap_unlock_write(mapping);
		uprobe_mmap(vma);
		if (adjust_next)
			uprobe_mmap(next);
	}
	if (remove_next) {
again:
		if (file) {
			uprobe_munmap(next, next->vm_start, next->vm_end);
			fput(file);
		}
		if (next->anon_vma)
			anon_vma_merge(vma, next);
		mm->map_count--;
		mpol_put(vma_policy(next));
		if (remove_next != 2)
			BUG_ON(vma->vm_end < next->vm_end);
		vm_area_free(next);
		if (remove_next == 2) {
			remove_next = 1;
			next = next_next;
			goto again;
		}
	}
	if (insert && file)
		uprobe_mmap(insert);
	mas_destroy(&mas);
	validate_mm(mm);
	return 0;
}
static inline int is_mergeable_vma(struct vm_area_struct *vma,
				struct file *file, unsigned long vm_flags,
				struct vm_userfaultfd_ctx vm_userfaultfd_ctx,
				struct anon_vma_name *anon_name)
{
	if ((vma->vm_flags ^ vm_flags) & ~VM_SOFTDIRTY)
		return 0;
	if (vma->vm_file != file)
		return 0;
	if (vma->vm_ops && vma->vm_ops->close)
		return 0;
	if (!is_mergeable_vm_userfaultfd_ctx(vma, vm_userfaultfd_ctx))
		return 0;
	if (!anon_vma_name_eq(anon_vma_name(vma), anon_name))
		return 0;
	return 1;
}
static inline int is_mergeable_anon_vma(struct anon_vma *anon_vma1,
					struct anon_vma *anon_vma2,
					struct vm_area_struct *vma)
{
	if ((!anon_vma1 || !anon_vma2) && (!vma ||
		list_is_singular(&vma->anon_vma_chain)))
		return 1;
	return anon_vma1 == anon_vma2;
}
static int
can_vma_merge_before(struct vm_area_struct *vma, unsigned long vm_flags,
		     struct anon_vma *anon_vma, struct file *file,
		     pgoff_t vm_pgoff,
		     struct vm_userfaultfd_ctx vm_userfaultfd_ctx,
		     struct anon_vma_name *anon_name)
{
	if (is_mergeable_vma(vma, file, vm_flags, vm_userfaultfd_ctx, anon_name) &&
	    is_mergeable_anon_vma(anon_vma, vma->anon_vma, vma)) {
		if (vma->vm_pgoff == vm_pgoff)
			return 1;
	}
	return 0;
}
static int
can_vma_merge_after(struct vm_area_struct *vma, unsigned long vm_flags,
		    struct anon_vma *anon_vma, struct file *file,
		    pgoff_t vm_pgoff,
		    struct vm_userfaultfd_ctx vm_userfaultfd_ctx,
		    struct anon_vma_name *anon_name)
{
	if (is_mergeable_vma(vma, file, vm_flags, vm_userfaultfd_ctx, anon_name) &&
	    is_mergeable_anon_vma(anon_vma, vma->anon_vma, vma)) {
		pgoff_t vm_pglen;
		vm_pglen = vma_pages(vma);
		if (vma->vm_pgoff + vm_pglen == vm_pgoff)
			return 1;
	}
	return 0;
}
struct vm_area_struct *vma_merge(struct mm_struct *mm,
			struct vm_area_struct *prev, unsigned long addr,
			unsigned long end, unsigned long vm_flags,
			struct anon_vma *anon_vma, struct file *file,
			pgoff_t pgoff, struct mempolicy *policy,
			struct vm_userfaultfd_ctx vm_userfaultfd_ctx,
			struct anon_vma_name *anon_name)
{
	pgoff_t pglen = (end - addr) >> PAGE_SHIFT;
	struct vm_area_struct *mid, *next, *res;
	int err = -1;
	bool merge_prev = false;
	bool merge_next = false;
	if (vm_flags & VM_SPECIAL)
		return NULL;
	next = find_vma(mm, prev ? prev->vm_end : 0);
	mid = next;
	if (next && next->vm_end == end)	
		next = find_vma(mm, next->vm_end);
	VM_WARN_ON(prev && addr <= prev->vm_start);
	VM_WARN_ON(mid && end > mid->vm_end);
	VM_WARN_ON(addr >= end);
	if (prev && prev->vm_end == addr &&
			mpol_equal(vma_policy(prev), policy) &&
			can_vma_merge_after(prev, vm_flags,
					    anon_vma, file, pgoff,
					    vm_userfaultfd_ctx, anon_name)) {
		merge_prev = true;
	}
	if (next && end == next->vm_start &&
			mpol_equal(policy, vma_policy(next)) &&
			can_vma_merge_before(next, vm_flags,
					     anon_vma, file, pgoff+pglen,
					     vm_userfaultfd_ctx, anon_name)) {
		merge_next = true;
	}
	if (merge_prev && merge_next &&
			is_mergeable_anon_vma(prev->anon_vma,
				next->anon_vma, NULL)) {	
		err = __vma_adjust(prev, prev->vm_start,
					next->vm_end, prev->vm_pgoff, NULL,
					prev);
		res = prev;
	} else if (merge_prev) {		
		err = __vma_adjust(prev, prev->vm_start,
					end, prev->vm_pgoff, NULL, prev);
		res = prev;
	} else if (merge_next) {
		if (prev && addr < prev->vm_end)
			err = __vma_adjust(prev, prev->vm_start,
					addr, prev->vm_pgoff, NULL, next);
		else				
			err = __vma_adjust(mid, addr, next->vm_end,
					next->vm_pgoff - pglen, NULL, next);
		res = next;
	}
	if (err)
		return NULL;
	khugepaged_enter_vma(res, vm_flags);
	return res;
}
static int anon_vma_compatible(struct vm_area_struct *a, struct vm_area_struct *b)
{
	return a->vm_end == b->vm_start &&
		mpol_equal(vma_policy(a), vma_policy(b)) &&
		a->vm_file == b->vm_file &&
		!((a->vm_flags ^ b->vm_flags) & ~(VM_ACCESS_FLAGS | VM_SOFTDIRTY)) &&
		b->vm_pgoff == a->vm_pgoff + ((b->vm_start - a->vm_start) >> PAGE_SHIFT);
}
static struct anon_vma *reusable_anon_vma(struct vm_area_struct *old, struct vm_area_struct *a, struct vm_area_struct *b)
{
	if (anon_vma_compatible(a, b)) {
		struct anon_vma *anon_vma = READ_ONCE(old->anon_vma);
		if (anon_vma && list_is_singular(&old->anon_vma_chain))
			return anon_vma;
	}
	return NULL;
}
struct anon_vma *find_mergeable_anon_vma(struct vm_area_struct *vma)
{
	MA_STATE(mas, &vma->vm_mm->mm_mt, vma->vm_end, vma->vm_end);
	struct anon_vma *anon_vma = NULL;
	struct vm_area_struct *prev, *next;
	next = mas_walk(&mas);
	if (next) {
		anon_vma = reusable_anon_vma(next, vma, next);
		if (anon_vma)
			return anon_vma;
	}
	prev = mas_prev(&mas, 0);
	VM_BUG_ON_VMA(prev != vma, vma);
	prev = mas_prev(&mas, 0);
	if (prev)
		anon_vma = reusable_anon_vma(prev, prev, vma);
	return anon_vma;
}
static inline unsigned long round_hint_to_min(unsigned long hint)
{
	hint &= PAGE_MASK;
	if (((void *)hint != NULL) &&
	    (hint < mmap_min_addr))
		return PAGE_ALIGN(mmap_min_addr);
	return hint;
}
int mlock_future_check(struct mm_struct *mm, unsigned long flags,
		       unsigned long len)
{
	unsigned long locked, lock_limit;
	if (flags & VM_LOCKED) {
		locked = len >> PAGE_SHIFT;
		locked += mm->locked_vm;
		lock_limit = rlimit(RLIMIT_MEMLOCK);
		lock_limit >>= PAGE_SHIFT;
		if (locked > lock_limit && !capable(CAP_IPC_LOCK))
			return -EAGAIN;
	}
	return 0;
}
static inline u64 file_mmap_size_max(struct file *file, struct inode *inode)
{
	if (S_ISREG(inode->i_mode))
		return MAX_LFS_FILESIZE;
	if (S_ISBLK(inode->i_mode))
		return MAX_LFS_FILESIZE;
	if (S_ISSOCK(inode->i_mode))
		return MAX_LFS_FILESIZE;
	if (file->f_mode & FMODE_UNSIGNED_OFFSET)
		return 0;
	return ULONG_MAX;
}
static inline bool file_mmap_ok(struct file *file, struct inode *inode,
				unsigned long pgoff, unsigned long len)
{
	u64 maxsize = file_mmap_size_max(file, inode);
	if (maxsize && len > maxsize)
		return false;
	maxsize -= len;
	if (pgoff > maxsize >> PAGE_SHIFT)
		return false;
	return true;
}
unsigned long do_mmap(struct file *file, unsigned long addr,
			unsigned long len, unsigned long prot,
			unsigned long flags, unsigned long pgoff,
			unsigned long *populate, struct list_head *uf)
{
	struct mm_struct *mm = current->mm;
	vm_flags_t vm_flags;
	int pkey = 0;
	validate_mm(mm);
	*populate = 0;
	if (!len)
		return -EINVAL;
	if ((prot & PROT_READ) && (current->personality & READ_IMPLIES_EXEC))
		if (!(file && path_noexec(&file->f_path)))
			prot |= PROT_EXEC;
	if (flags & MAP_FIXED_NOREPLACE)
		flags |= MAP_FIXED;
	if (!(flags & MAP_FIXED))
		addr = round_hint_to_min(addr);
	len = PAGE_ALIGN(len);
	if (!len)
		return -ENOMEM;
	if ((pgoff + (len >> PAGE_SHIFT)) < pgoff)
		return -EOVERFLOW;
	if (mm->map_count > sysctl_max_map_count)
		return -ENOMEM;
	addr = get_unmapped_area(file, addr, len, pgoff, flags);
	if (IS_ERR_VALUE(addr))
		return addr;
	if (flags & MAP_FIXED_NOREPLACE) {
		if (find_vma_intersection(mm, addr, addr + len))
			return -EEXIST;
	}
	if (prot == PROT_EXEC) {
		pkey = execute_only_pkey(mm);
		if (pkey < 0)
			pkey = 0;
	}
	vm_flags = calc_vm_prot_bits(prot, pkey) | calc_vm_flag_bits(flags) |
			mm->def_flags | VM_MAYREAD | VM_MAYWRITE | VM_MAYEXEC;
	if (flags & MAP_LOCKED)
		if (!can_do_mlock())
			return -EPERM;
	if (mlock_future_check(mm, vm_flags, len))
		return -EAGAIN;
	if (file) {
		struct inode *inode = file_inode(file);
		unsigned long flags_mask;
		if (!file_mmap_ok(file, inode, pgoff, len))
			return -EOVERFLOW;
		flags_mask = LEGACY_MAP_MASK | file->f_op->mmap_supported_flags;
		switch (flags & MAP_TYPE) {
		case MAP_SHARED:
			flags &= LEGACY_MAP_MASK;
			fallthrough;
		case MAP_SHARED_VALIDATE:
			if (flags & ~flags_mask)
				return -EOPNOTSUPP;
			if (prot & PROT_WRITE) {
				if (!(file->f_mode & FMODE_WRITE))
					return -EACCES;
				if (IS_SWAPFILE(file->f_mapping->host))
					return -ETXTBSY;
			}
			if (IS_APPEND(inode) && (file->f_mode & FMODE_WRITE))
				return -EACCES;
			vm_flags |= VM_SHARED | VM_MAYSHARE;
			if (!(file->f_mode & FMODE_WRITE))
				vm_flags &= ~(VM_MAYWRITE | VM_SHARED);
			fallthrough;
		case MAP_PRIVATE:
			if (!(file->f_mode & FMODE_READ))
				return -EACCES;
			if (path_noexec(&file->f_path)) {
				if (vm_flags & VM_EXEC)
					return -EPERM;
				vm_flags &= ~VM_MAYEXEC;
			}
			if (!file->f_op->mmap)
				return -ENODEV;
			if (vm_flags & (VM_GROWSDOWN|VM_GROWSUP))
				return -EINVAL;
			break;
		default:
			return -EINVAL;
		}
	} else {
		switch (flags & MAP_TYPE) {
		case MAP_SHARED:
			if (vm_flags & (VM_GROWSDOWN|VM_GROWSUP))
				return -EINVAL;
			pgoff = 0;
			vm_flags |= VM_SHARED | VM_MAYSHARE;
			break;
		case MAP_PRIVATE:
			pgoff = addr >> PAGE_SHIFT;
			break;
		default:
			return -EINVAL;
		}
	}
	if (flags & MAP_NORESERVE) {
		if (sysctl_overcommit_memory != OVERCOMMIT_NEVER)
			vm_flags |= VM_NORESERVE;
		if (file && is_file_hugepages(file))
			vm_flags |= VM_NORESERVE;
	}
	addr = mmap_region(file, addr, len, vm_flags, pgoff, uf);
	if (!IS_ERR_VALUE(addr) &&
	    ((vm_flags & VM_LOCKED) ||
	     (flags & (MAP_POPULATE | MAP_NONBLOCK)) == MAP_POPULATE))
		*populate = len;
	return addr;
}
unsigned long ksys_mmap_pgoff(unsigned long addr, unsigned long len,
			      unsigned long prot, unsigned long flags,
			      unsigned long fd, unsigned long pgoff)
{
	struct file *file = NULL;
	unsigned long retval;
	if (!(flags & MAP_ANONYMOUS)) {
		audit_mmap_fd(fd, flags);
		file = fget(fd);
		if (!file)
			return -EBADF;
		if (is_file_hugepages(file)) {
			len = ALIGN(len, huge_page_size(hstate_file(file)));
		} else if (unlikely(flags & MAP_HUGETLB)) {
			retval = -EINVAL;
			goto out_fput;
		}
	} else if (flags & MAP_HUGETLB) {
		struct hstate *hs;
		hs = hstate_sizelog((flags >> MAP_HUGE_SHIFT) & MAP_HUGE_MASK);
		if (!hs)
			return -EINVAL;
		len = ALIGN(len, huge_page_size(hs));
		file = hugetlb_file_setup(HUGETLB_ANON_FILE, len,
				VM_NORESERVE,
				HUGETLB_ANONHUGE_INODE,
				(flags >> MAP_HUGE_SHIFT) & MAP_HUGE_MASK);
		if (IS_ERR(file))
			return PTR_ERR(file);
	}
	retval = vm_mmap_pgoff(file, addr, len, prot, flags, pgoff);
out_fput:
	if (file)
		fput(file);
	return retval;
}
SYSCALL_DEFINE6(mmap_pgoff, unsigned long, addr, unsigned long, len,
		unsigned long, prot, unsigned long, flags,
		unsigned long, fd, unsigned long, pgoff)
{
	unsigned long result = ksys_mmap_pgoff(addr, len, prot, flags, fd, pgoff);
	return result;
}
#ifdef __ARCH_WANT_SYS_OLD_MMAP
struct mmap_arg_struct {
	unsigned long addr;
	unsigned long len;
	unsigned long prot;
	unsigned long flags;
	unsigned long fd;
	unsigned long offset;
};
SYSCALL_DEFINE1(old_mmap, struct mmap_arg_struct __user *, arg)
{
	struct mmap_arg_struct a;
	if (copy_from_user(&a, arg, sizeof(a)))
		return -EFAULT;
	if (offset_in_page(a.offset))
		return -EINVAL;
	return ksys_mmap_pgoff(a.addr, a.len, a.prot, a.flags, a.fd,
			       a.offset >> PAGE_SHIFT);
}
#endif
int vma_wants_writenotify(struct vm_area_struct *vma, pgprot_t vm_page_prot)
{
	vm_flags_t vm_flags = vma->vm_flags;
	const struct vm_operations_struct *vm_ops = vma->vm_ops;
	if ((vm_flags & (VM_WRITE|VM_SHARED)) != ((VM_WRITE|VM_SHARED)))
		return 0;
	if (vm_ops && (vm_ops->page_mkwrite || vm_ops->pfn_mkwrite))
		return 1;
	if (pgprot_val(vm_page_prot) !=
	    pgprot_val(vm_pgprot_modify(vm_page_prot, vm_flags)))
		return 0;
	if (vma_soft_dirty_enabled(vma) && !is_vm_hugetlb_page(vma))
		return 1;
	if (vm_flags & VM_PFNMAP)
		return 0;
	return vma->vm_file && vma->vm_file->f_mapping &&
		mapping_can_writeback(vma->vm_file->f_mapping);
}
static inline int accountable_mapping(struct file *file, vm_flags_t vm_flags)
{
	if (file && is_file_hugepages(file))
		return 0;
	return (vm_flags & (VM_NORESERVE | VM_SHARED | VM_WRITE)) == VM_WRITE;
}
/**
 * unmapped_area() - Find an area between the low_limit and the high_limit with
 * the correct alignment and offset, all from @info. Note: current->mm is used
 * for the search.
 *
 * @info: The unmapped area information including the range (low_limit -
 * hight_limit), the alignment offset and mask.
 *
 * Return: A memory address or -ENOMEM.
 */
static unsigned long unmapped_area(struct vm_unmapped_area_info *info)
{
	unsigned long length, gap;
	MA_STATE(mas, &current->mm->mm_mt, 0, 0);
	length = info->length + info->align_mask;
	if (length < info->length)
		return -ENOMEM;
	if (mas_empty_area(&mas, info->low_limit, info->high_limit - 1,
				  length))
		return -ENOMEM;
	gap = mas.index;
	gap += (info->align_offset - gap) & info->align_mask;
	return gap;
}
/**
 * unmapped_area_topdown() - Find an area between the low_limit and the
 * high_limit with * the correct alignment and offset at the highest available
 * address, all from @info. Note: current->mm is used for the search.
 *
 * @info: The unmapped area information including the range (low_limit -
 * hight_limit), the alignment offset and mask.
 *
 * Return: A memory address or -ENOMEM.
 */
static unsigned long unmapped_area_topdown(struct vm_unmapped_area_info *info)
{
	unsigned long length, gap;
	MA_STATE(mas, &current->mm->mm_mt, 0, 0);
	length = info->length + info->align_mask;
	if (length < info->length)
		return -ENOMEM;
	if (mas_empty_area_rev(&mas, info->low_limit, info->high_limit - 1,
				length))
		return -ENOMEM;
	gap = mas.last + 1 - info->length;
	gap -= (gap - info->align_offset) & info->align_mask;
	return gap;
}
unsigned long vm_unmapped_area(struct vm_unmapped_area_info *info)
{
	unsigned long addr;
	if (info->flags & VM_UNMAPPED_AREA_TOPDOWN)
		addr = unmapped_area_topdown(info);
	else
		addr = unmapped_area(info);
	trace_vm_unmapped_area(addr, info);
	return addr;
}
unsigned long
generic_get_unmapped_area(struct file *filp, unsigned long addr,
			  unsigned long len, unsigned long pgoff,
			  unsigned long flags)
{
	struct mm_struct *mm = current->mm;
	struct vm_area_struct *vma, *prev;
	struct vm_unmapped_area_info info;
	const unsigned long mmap_end = arch_get_mmap_end(addr, len, flags);
	if (len > mmap_end - mmap_min_addr)
		return -ENOMEM;
	if (flags & MAP_FIXED)
		return addr;
	if (addr) {
		addr = PAGE_ALIGN(addr);
		vma = find_vma_prev(mm, addr, &prev);
		if (mmap_end - len >= addr && addr >= mmap_min_addr &&
		    (!vma || addr + len <= vm_start_gap(vma)) &&
		    (!prev || addr >= vm_end_gap(prev)))
			return addr;
	}
	info.flags = 0;
	info.length = len;
	info.low_limit = mm->mmap_base;
	info.high_limit = mmap_end;
	info.align_mask = 0;
	info.align_offset = 0;
	return vm_unmapped_area(&info);
}
#ifndef HAVE_ARCH_UNMAPPED_AREA
unsigned long
arch_get_unmapped_area(struct file *filp, unsigned long addr,
		       unsigned long len, unsigned long pgoff,
		       unsigned long flags)
{
	return generic_get_unmapped_area(filp, addr, len, pgoff, flags);
}
#endif
unsigned long
generic_get_unmapped_area_topdown(struct file *filp, unsigned long addr,
				  unsigned long len, unsigned long pgoff,
				  unsigned long flags)
{
	struct vm_area_struct *vma, *prev;
	struct mm_struct *mm = current->mm;
	struct vm_unmapped_area_info info;
	const unsigned long mmap_end = arch_get_mmap_end(addr, len, flags);
	if (len > mmap_end - mmap_min_addr)
		return -ENOMEM;
	if (flags & MAP_FIXED)
		return addr;
	if (addr) {
		addr = PAGE_ALIGN(addr);
		vma = find_vma_prev(mm, addr, &prev);
		if (mmap_end - len >= addr && addr >= mmap_min_addr &&
				(!vma || addr + len <= vm_start_gap(vma)) &&
				(!prev || addr >= vm_end_gap(prev)))
			return addr;
	}
	info.flags = VM_UNMAPPED_AREA_TOPDOWN;
	info.length = len;
	info.low_limit = max(PAGE_SIZE, mmap_min_addr);
	info.high_limit = arch_get_mmap_base(addr, mm->mmap_base);
	info.align_mask = 0;
	info.align_offset = 0;
	addr = vm_unmapped_area(&info);
	if (offset_in_page(addr)) {
		VM_BUG_ON(addr != -ENOMEM);
		info.flags = 0;
		info.low_limit = TASK_UNMAPPED_BASE;
		info.high_limit = mmap_end;
		addr = vm_unmapped_area(&info);
	}
	return addr;
}
#ifndef HAVE_ARCH_UNMAPPED_AREA_TOPDOWN
unsigned long
arch_get_unmapped_area_topdown(struct file *filp, unsigned long addr,
			       unsigned long len, unsigned long pgoff,
			       unsigned long flags)
{
	return generic_get_unmapped_area_topdown(filp, addr, len, pgoff, flags);
}
#endif
unsigned long
get_unmapped_area(struct file *file, unsigned long addr, unsigned long len,
		unsigned long pgoff, unsigned long flags)
{
	unsigned long (*get_area)(struct file *, unsigned long,
				  unsigned long, unsigned long, unsigned long);
	unsigned long error = arch_mmap_check(addr, len, flags);
	if (error)
		return error;
	if (len > TASK_SIZE)
		return -ENOMEM;
	get_area = current->mm->get_unmapped_area;
	if (file) {
		if (file->f_op->get_unmapped_area)
			get_area = file->f_op->get_unmapped_area;
	} else if (flags & MAP_SHARED) {
		pgoff = 0;
		get_area = shmem_get_unmapped_area;
	}
	addr = get_area(file, addr, len, pgoff, flags);
	if (IS_ERR_VALUE(addr))
		return addr;
	if (addr > TASK_SIZE - len)
		return -ENOMEM;
	if (offset_in_page(addr))
		return -EINVAL;
	error = security_mmap_addr(addr);
	return error ? error : addr;
}
EXPORT_SYMBOL(get_unmapped_area);
/**
 * find_vma_intersection() - Look up the first VMA which intersects the interval
 * @mm: The process address space.
 * @start_addr: The inclusive start user address.
 * @end_addr: The exclusive end user address.
 *
 * Returns: The first VMA within the provided range, %NULL otherwise.  Assumes
 * start_addr < end_addr.
 */
struct vm_area_struct *find_vma_intersection(struct mm_struct *mm,
					     unsigned long start_addr,
					     unsigned long end_addr)
{
	unsigned long index = start_addr;
	mmap_assert_locked(mm);
	return mt_find(&mm->mm_mt, &index, end_addr - 1);
}
EXPORT_SYMBOL(find_vma_intersection);
/**
 * find_vma() - Find the VMA for a given address, or the next VMA.
 * @mm: The mm_struct to check
 * @addr: The address
 *
 * Returns: The VMA associated with addr, or the next VMA.
 * May return %NULL in the case of no VMA at addr or above.
 */
struct vm_area_struct *find_vma(struct mm_struct *mm, unsigned long addr)
{
	unsigned long index = addr;
	mmap_assert_locked(mm);
	return mt_find(&mm->mm_mt, &index, ULONG_MAX);
}
EXPORT_SYMBOL(find_vma);
/**
 * find_vma_prev() - Find the VMA for a given address, or the next vma and
 * set %pprev to the previous VMA, if any.
 * @mm: The mm_struct to check
 * @addr: The address
 * @pprev: The pointer to set to the previous VMA
 *
 * Note that RCU lock is missing here since the external mmap_lock() is used
 * instead.
 *
 * Returns: The VMA associated with @addr, or the next vma.
 * May return %NULL in the case of no vma at addr or above.
 */
struct vm_area_struct *
find_vma_prev(struct mm_struct *mm, unsigned long addr,
			struct vm_area_struct **pprev)
{
	struct vm_area_struct *vma;
	MA_STATE(mas, &mm->mm_mt, addr, addr);
	vma = mas_walk(&mas);
	*pprev = mas_prev(&mas, 0);
	if (!vma)
		vma = mas_next(&mas, ULONG_MAX);
	return vma;
}
static int acct_stack_growth(struct vm_area_struct *vma,
			     unsigned long size, unsigned long grow)
{
	struct mm_struct *mm = vma->vm_mm;
	unsigned long new_start;
	if (!may_expand_vm(mm, vma->vm_flags, grow))
		return -ENOMEM;
	if (size > rlimit(RLIMIT_STACK))
		return -ENOMEM;
	if (mlock_future_check(mm, vma->vm_flags, grow << PAGE_SHIFT))
		return -ENOMEM;
	new_start = (vma->vm_flags & VM_GROWSUP) ? vma->vm_start :
			vma->vm_end - size;
	if (is_hugepage_only_range(vma->vm_mm, new_start, size))
		return -EFAULT;
	if (security_vm_enough_memory_mm(mm, grow))
		return -ENOMEM;
	return 0;
}
#if defined(CONFIG_STACK_GROWSUP) || defined(CONFIG_IA64)
int expand_upwards(struct vm_area_struct *vma, unsigned long address)
{
	struct mm_struct *mm = vma->vm_mm;
	struct vm_area_struct *next;
	unsigned long gap_addr;
	int error = 0;
	MA_STATE(mas, &mm->mm_mt, 0, 0);
	if (!(vma->vm_flags & VM_GROWSUP))
		return -EFAULT;
	address &= PAGE_MASK;
	if (address >= (TASK_SIZE & PAGE_MASK))
		return -ENOMEM;
	address += PAGE_SIZE;
	gap_addr = address + stack_guard_gap;
	if (gap_addr < address || gap_addr > TASK_SIZE)
		gap_addr = TASK_SIZE;
	next = find_vma_intersection(mm, vma->vm_end, gap_addr);
	if (next && vma_is_accessible(next)) {
		if (!(next->vm_flags & VM_GROWSUP))
			return -ENOMEM;
	}
	if (mas_preallocate(&mas, vma, GFP_KERNEL))
		return -ENOMEM;
	if (unlikely(anon_vma_prepare(vma))) {
		mas_destroy(&mas);
		return -ENOMEM;
	}
	anon_vma_lock_write(vma->anon_vma);
	if (address > vma->vm_end) {
		unsigned long size, grow;
		size = address - vma->vm_start;
		grow = (address - vma->vm_end) >> PAGE_SHIFT;
		error = -ENOMEM;
		if (vma->vm_pgoff + (size >> PAGE_SHIFT) >= vma->vm_pgoff) {
			error = acct_stack_growth(vma, size, grow);
			if (!error) {
				spin_lock(&mm->page_table_lock);
				if (vma->vm_flags & VM_LOCKED)
					mm->locked_vm += grow;
				vm_stat_account(mm, vma->vm_flags, grow);
				anon_vma_interval_tree_pre_update_vma(vma);
				vma->vm_end = address;
				vma_mas_store(vma, &mas);
				anon_vma_interval_tree_post_update_vma(vma);
				spin_unlock(&mm->page_table_lock);
				perf_event_mmap(vma);
			}
		}
	}
	anon_vma_unlock_write(vma->anon_vma);
	khugepaged_enter_vma(vma, vma->vm_flags);
	mas_destroy(&mas);
	return error;
}
#endif
int expand_downwards(struct vm_area_struct *vma, unsigned long address)
{
	struct mm_struct *mm = vma->vm_mm;
	MA_STATE(mas, &mm->mm_mt, vma->vm_start, vma->vm_start);
	struct vm_area_struct *prev;
	int error = 0;
	address &= PAGE_MASK;
	if (address < mmap_min_addr)
		return -EPERM;
	prev = mas_prev(&mas, 0);
	if (prev && !(prev->vm_flags & VM_GROWSDOWN) &&
			vma_is_accessible(prev)) {
		if (address - prev->vm_end < stack_guard_gap)
			return -ENOMEM;
	}
	if (mas_preallocate(&mas, vma, GFP_KERNEL))
		return -ENOMEM;
	if (unlikely(anon_vma_prepare(vma))) {
		mas_destroy(&mas);
		return -ENOMEM;
	}
	anon_vma_lock_write(vma->anon_vma);
	if (address < vma->vm_start) {
		unsigned long size, grow;
		size = vma->vm_end - address;
		grow = (vma->vm_start - address) >> PAGE_SHIFT;
		error = -ENOMEM;
		if (grow <= vma->vm_pgoff) {
			error = acct_stack_growth(vma, size, grow);
			if (!error) {
				spin_lock(&mm->page_table_lock);
				if (vma->vm_flags & VM_LOCKED)
					mm->locked_vm += grow;
				vm_stat_account(mm, vma->vm_flags, grow);
				anon_vma_interval_tree_pre_update_vma(vma);
				vma->vm_start = address;
				vma->vm_pgoff -= grow;
				vma_mas_store(vma, &mas);
				anon_vma_interval_tree_post_update_vma(vma);
				spin_unlock(&mm->page_table_lock);
				perf_event_mmap(vma);
			}
		}
	}
	anon_vma_unlock_write(vma->anon_vma);
	khugepaged_enter_vma(vma, vma->vm_flags);
	mas_destroy(&mas);
	return error;
}
unsigned long stack_guard_gap = 256UL<<PAGE_SHIFT;
static int __init cmdline_parse_stack_guard_gap(char *p)
{
	unsigned long val;
	char *endptr;
	val = simple_strtoul(p, &endptr, 10);
	if (!*endptr)
		stack_guard_gap = val << PAGE_SHIFT;
	return 1;
}
__setup("stack_guard_gap=", cmdline_parse_stack_guard_gap);
#ifdef CONFIG_STACK_GROWSUP
int expand_stack(struct vm_area_struct *vma, unsigned long address)
{
	return expand_upwards(vma, address);
}
struct vm_area_struct *
find_extend_vma(struct mm_struct *mm, unsigned long addr)
{
	struct vm_area_struct *vma, *prev;
	addr &= PAGE_MASK;
	vma = find_vma_prev(mm, addr, &prev);
	if (vma && (vma->vm_start <= addr))
		return vma;
	if (!prev || expand_stack(prev, addr))
		return NULL;
	if (prev->vm_flags & VM_LOCKED)
		populate_vma_page_range(prev, addr, prev->vm_end, NULL);
	return prev;
}
#else
int expand_stack(struct vm_area_struct *vma, unsigned long address)
{
	return expand_downwards(vma, address);
}
struct vm_area_struct *
find_extend_vma(struct mm_struct *mm, unsigned long addr)
{
	struct vm_area_struct *vma;
	unsigned long start;
	addr &= PAGE_MASK;
	vma = find_vma(mm, addr);
	if (!vma)
		return NULL;
	if (vma->vm_start <= addr)
		return vma;
	if (!(vma->vm_flags & VM_GROWSDOWN))
		return NULL;
	start = vma->vm_start;
	if (expand_stack(vma, addr))
		return NULL;
	if (vma->vm_flags & VM_LOCKED)
		populate_vma_page_range(vma, addr, start, NULL);
	return vma;
}
#endif
EXPORT_SYMBOL_GPL(find_extend_vma);
static inline void remove_mt(struct mm_struct *mm, struct ma_state *mas)
{
	unsigned long nr_accounted = 0;
	struct vm_area_struct *vma;
	update_hiwater_vm(mm);
	mas_for_each(mas, vma, ULONG_MAX) {
		long nrpages = vma_pages(vma);
		if (vma->vm_flags & VM_ACCOUNT)
			nr_accounted += nrpages;
		vm_stat_account(mm, vma->vm_flags, -nrpages);
		remove_vma(vma);
	}
	vm_unacct_memory(nr_accounted);
	validate_mm(mm);
}
static void unmap_region(struct mm_struct *mm, struct maple_tree *mt,
		struct vm_area_struct *vma, struct vm_area_struct *prev,
		struct vm_area_struct *next,
		unsigned long start, unsigned long end)
{
	struct mmu_gather tlb;
	lru_add_drain();
	tlb_gather_mmu(&tlb, mm);
	update_hiwater_rss(mm);
	unmap_vmas(&tlb, mt, vma, start, end);
	free_pgtables(&tlb, mt, vma, prev ? prev->vm_end : FIRST_USER_ADDRESS,
				 next ? next->vm_start : USER_PGTABLES_CEILING);
	tlb_finish_mmu(&tlb);
}
int __split_vma(struct mm_struct *mm, struct vm_area_struct *vma,
		unsigned long addr, int new_below)
{
	struct vm_area_struct *new;
	int err;
	validate_mm_mt(mm);
	if (vma->vm_ops && vma->vm_ops->may_split) {
		err = vma->vm_ops->may_split(vma, addr);
		if (err)
			return err;
	}
	new = vm_area_dup(vma);
	if (!new)
		return -ENOMEM;
	if (new_below)
		new->vm_end = addr;
	else {
		new->vm_start = addr;
		new->vm_pgoff += ((addr - vma->vm_start) >> PAGE_SHIFT);
	}
	err = vma_dup_policy(vma, new);
	if (err)
		goto out_free_vma;
	err = anon_vma_clone(new, vma);
	if (err)
		goto out_free_mpol;
	if (new->vm_file)
		get_file(new->vm_file);
	if (new->vm_ops && new->vm_ops->open)
		new->vm_ops->open(new);
	if (new_below)
		err = vma_adjust(vma, addr, vma->vm_end, vma->vm_pgoff +
			((addr - new->vm_start) >> PAGE_SHIFT), new);
	else
		err = vma_adjust(vma, vma->vm_start, addr, vma->vm_pgoff, new);
	if (!err)
		return 0;
	new->vm_start = new->vm_end;
	new->vm_pgoff = 0;
	if (new->vm_ops && new->vm_ops->close)
		new->vm_ops->close(new);
	if (new->vm_file)
		fput(new->vm_file);
	unlink_anon_vmas(new);
 out_free_mpol:
	mpol_put(vma_policy(new));
 out_free_vma:
	vm_area_free(new);
	validate_mm_mt(mm);
	return err;
}
int split_vma(struct mm_struct *mm, struct vm_area_struct *vma,
	      unsigned long addr, int new_below)
{
	if (mm->map_count >= sysctl_max_map_count)
		return -ENOMEM;
	return __split_vma(mm, vma, addr, new_below);
}
static inline int munmap_sidetree(struct vm_area_struct *vma,
				   struct ma_state *mas_detach)
{
	mas_set_range(mas_detach, vma->vm_start, vma->vm_end - 1);
	if (mas_store_gfp(mas_detach, vma, GFP_KERNEL))
		return -ENOMEM;
	if (vma->vm_flags & VM_LOCKED)
		vma->vm_mm->locked_vm -= vma_pages(vma);
	return 0;
}
static int
do_mas_align_munmap(struct ma_state *mas, struct vm_area_struct *vma,
		    struct mm_struct *mm, unsigned long start,
		    unsigned long end, struct list_head *uf, bool downgrade)
{
	struct vm_area_struct *prev, *next = NULL;
	struct maple_tree mt_detach;
	int count = 0;
	int error = -ENOMEM;
	MA_STATE(mas_detach, &mt_detach, 0, 0);
	mt_init_flags(&mt_detach, MT_FLAGS_LOCK_EXTERN);
	mt_set_external_lock(&mt_detach, &mm->mmap_lock);
	if (mas_preallocate(mas, vma, GFP_KERNEL))
		return -ENOMEM;
	mas->last = end - 1;
	if (start > vma->vm_start) {
		if (end < vma->vm_end && mm->map_count >= sysctl_max_map_count)
			goto map_count_exceeded;
		error = __split_vma(mm, vma, start, 0);
		if (error)
			goto start_split_failed;
		mas_set(mas, start);
		vma = mas_walk(mas);
	}
	prev = mas_prev(mas, 0);
	if (unlikely((!prev)))
		mas_set(mas, start);
	mas_for_each(mas, next, end - 1) {
		if (next->vm_end > end) {
			struct vm_area_struct *split;
			error = __split_vma(mm, next, end, 1);
			if (error)
				goto end_split_failed;
			mas_set(mas, end);
			split = mas_prev(mas, 0);
			error = munmap_sidetree(split, &mas_detach);
			if (error)
				goto munmap_sidetree_failed;
			count++;
			if (vma == next)
				vma = split;
			break;
		}
		error = munmap_sidetree(next, &mas_detach);
		if (error)
			goto munmap_sidetree_failed;
		count++;
#ifdef CONFIG_DEBUG_VM_MAPLE_TREE
		BUG_ON(next->vm_start < start);
		BUG_ON(next->vm_start > end);
#endif
	}
	if (!next)
		next = mas_next(mas, ULONG_MAX);
	if (unlikely(uf)) {
		error = userfaultfd_unmap_prep(mm, start, end, uf);
		if (error)
			goto userfaultfd_error;
	}
	mas_set_range(mas, start, end - 1);
#if defined(CONFIG_DEBUG_VM_MAPLE_TREE)
	{
		MA_STATE(test, &mt_detach, start, end - 1);
		struct vm_area_struct *vma_mas, *vma_test;
		int test_count = 0;
		rcu_read_lock();
		vma_test = mas_find(&test, end - 1);
		mas_for_each(mas, vma_mas, end - 1) {
			BUG_ON(vma_mas != vma_test);
			test_count++;
			vma_test = mas_next(&test, end - 1);
		}
		rcu_read_unlock();
		BUG_ON(count != test_count);
		mas_set_range(mas, start, end - 1);
	}
#endif
	mas_store_prealloc(mas, NULL);
	mm->map_count -= count;
	if (downgrade) {
		if (next && (next->vm_flags & VM_GROWSDOWN))
			downgrade = false;
		else if (prev && (prev->vm_flags & VM_GROWSUP))
			downgrade = false;
		else
			mmap_write_downgrade(mm);
	}
	unmap_region(mm, &mt_detach, vma, prev, next, start, end);
	mas_set(&mas_detach, start);
	remove_mt(mm, &mas_detach);
	__mt_destroy(&mt_detach);
	validate_mm(mm);
	return downgrade ? 1 : 0;
userfaultfd_error:
munmap_sidetree_failed:
end_split_failed:
	__mt_destroy(&mt_detach);
start_split_failed:
map_count_exceeded:
	mas_destroy(mas);
	return error;
}
int do_mas_munmap(struct ma_state *mas, struct mm_struct *mm,
		  unsigned long start, size_t len, struct list_head *uf,
		  bool downgrade)
{
	unsigned long end;
	struct vm_area_struct *vma;
	if ((offset_in_page(start)) || start > TASK_SIZE || len > TASK_SIZE-start)
		return -EINVAL;
	end = start + PAGE_ALIGN(len);
	if (end == start)
		return -EINVAL;
	arch_unmap(mm, start, end);
	vma = mas_find(mas, end - 1);
	if (!vma)
		return 0;
	return do_mas_align_munmap(mas, vma, mm, start, end, uf, downgrade);
}
int do_munmap(struct mm_struct *mm, unsigned long start, size_t len,
	      struct list_head *uf)
{
	MA_STATE(mas, &mm->mm_mt, start, start);
	return do_mas_munmap(&mas, mm, start, len, uf, false);
}
unsigned long mmap_region(struct file *file, unsigned long addr,
		unsigned long len, vm_flags_t vm_flags, unsigned long pgoff,
		struct list_head *uf)
{
	struct mm_struct *mm = current->mm;
	struct vm_area_struct *vma = NULL;
	struct vm_area_struct *next, *prev, *merge;
	pgoff_t pglen = len >> PAGE_SHIFT;
	unsigned long charged = 0;
	unsigned long end = addr + len;
	unsigned long merge_start = addr, merge_end = end;
	pgoff_t vm_pgoff;
	int error;
	MA_STATE(mas, &mm->mm_mt, addr, end - 1);
	if (!may_expand_vm(mm, vm_flags, len >> PAGE_SHIFT)) {
		unsigned long nr_pages;
		nr_pages = count_vma_pages_range(mm, addr, end);
		if (!may_expand_vm(mm, vm_flags,
					(len >> PAGE_SHIFT) - nr_pages))
			return -ENOMEM;
	}
	if (do_mas_munmap(&mas, mm, addr, len, uf, false))
		return -ENOMEM;
	if (accountable_mapping(file, vm_flags)) {
		charged = len >> PAGE_SHIFT;
		if (security_vm_enough_memory_mm(mm, charged))
			return -ENOMEM;
		vm_flags |= VM_ACCOUNT;
	}
	next = mas_next(&mas, ULONG_MAX);
	prev = mas_prev(&mas, 0);
	if (vm_flags & VM_SPECIAL)
		goto cannot_expand;
	if (next && next->vm_start == end && !vma_policy(next) &&
	    can_vma_merge_before(next, vm_flags, NULL, file, pgoff+pglen,
				 NULL_VM_UFFD_CTX, NULL)) {
		merge_end = next->vm_end;
		vma = next;
		vm_pgoff = next->vm_pgoff - pglen;
	}
	if (prev && prev->vm_end == addr && !vma_policy(prev) &&
	    (vma ? can_vma_merge_after(prev, vm_flags, vma->anon_vma, file,
				       pgoff, vma->vm_userfaultfd_ctx, NULL) :
		   can_vma_merge_after(prev, vm_flags, NULL, file, pgoff,
				       NULL_VM_UFFD_CTX, NULL))) {
		merge_start = prev->vm_start;
		vma = prev;
		vm_pgoff = prev->vm_pgoff;
	}
	if (vma &&
	    !vma_expand(&mas, vma, merge_start, merge_end, vm_pgoff, next)) {
		khugepaged_enter_vma(vma, vm_flags);
		goto expanded;
	}
	mas.index = addr;
	mas.last = end - 1;
cannot_expand:
	vma = vm_area_alloc(mm);
	if (!vma) {
		error = -ENOMEM;
		goto unacct_error;
	}
	vma->vm_start = addr;
	vma->vm_end = end;
	vma->vm_flags = vm_flags;
	vma->vm_page_prot = vm_get_page_prot(vm_flags);
	vma->vm_pgoff = pgoff;
	if (file) {
		if (vm_flags & VM_SHARED) {
			error = mapping_map_writable(file->f_mapping);
			if (error)
				goto free_vma;
		}
		vma->vm_file = get_file(file);
		error = call_mmap(file, vma);
		if (error)
			goto unmap_and_free_vma;
		if (WARN_ON((addr != vma->vm_start))) {
			error = -EINVAL;
			goto close_and_free_vma;
		}
		mas_reset(&mas);
		if (unlikely(vm_flags != vma->vm_flags && prev)) {
			merge = vma_merge(mm, prev, vma->vm_start, vma->vm_end, vma->vm_flags,
				NULL, vma->vm_file, vma->vm_pgoff, NULL, NULL_VM_UFFD_CTX, NULL);
			if (merge) {
				fput(vma->vm_file);
				vm_area_free(vma);
				vma = merge;
				vm_flags = vma->vm_flags;
				goto unmap_writable;
			}
		}
		vm_flags = vma->vm_flags;
	} else if (vm_flags & VM_SHARED) {
		error = shmem_zero_setup(vma);
		if (error)
			goto free_vma;
	} else {
		vma_set_anonymous(vma);
	}
	if (!arch_validate_flags(vma->vm_flags)) {
		error = -EINVAL;
		if (file)
			goto close_and_free_vma;
		else if (vma->vm_file)
			goto unmap_and_free_vma;
		else
			goto free_vma;
	}
	if (mas_preallocate(&mas, vma, GFP_KERNEL)) {
		error = -ENOMEM;
		if (file)
			goto close_and_free_vma;
		else if (vma->vm_file)
			goto unmap_and_free_vma;
		else
			goto free_vma;
	}
	if (vma->vm_file)
		i_mmap_lock_write(vma->vm_file->f_mapping);
	vma_mas_store(vma, &mas);
	mm->map_count++;
	if (vma->vm_file) {
		if (vma->vm_flags & VM_SHARED)
			mapping_allow_writable(vma->vm_file->f_mapping);
		flush_dcache_mmap_lock(vma->vm_file->f_mapping);
		vma_interval_tree_insert(vma, &vma->vm_file->f_mapping->i_mmap);
		flush_dcache_mmap_unlock(vma->vm_file->f_mapping);
		i_mmap_unlock_write(vma->vm_file->f_mapping);
	}
	khugepaged_enter_vma(vma, vma->vm_flags);
unmap_writable:
	if (file && vm_flags & VM_SHARED)
		mapping_unmap_writable(file->f_mapping);
	file = vma->vm_file;
expanded:
	perf_event_mmap(vma);
	vm_stat_account(mm, vm_flags, len >> PAGE_SHIFT);
	if (vm_flags & VM_LOCKED) {
		if ((vm_flags & VM_SPECIAL) || vma_is_dax(vma) ||
					is_vm_hugetlb_page(vma) ||
					vma == get_gate_vma(current->mm))
			vma->vm_flags &= VM_LOCKED_CLEAR_MASK;
		else
			mm->locked_vm += (len >> PAGE_SHIFT);
	}
	if (file)
		uprobe_mmap(vma);
	vma->vm_flags |= VM_SOFTDIRTY;
	vma_set_page_prot(vma);
	validate_mm(mm);
	return addr;
close_and_free_vma:
	if (vma->vm_ops && vma->vm_ops->close)
		vma->vm_ops->close(vma);
unmap_and_free_vma:
	fput(vma->vm_file);
	vma->vm_file = NULL;
	unmap_region(mm, mas.tree, vma, prev, next, vma->vm_start, vma->vm_end);
	if (file && (vm_flags & VM_SHARED))
		mapping_unmap_writable(file->f_mapping);
free_vma:
	vm_area_free(vma);
unacct_error:
	if (charged)
		vm_unacct_memory(charged);
	validate_mm(mm);
	return error;
}
static int __vm_munmap(unsigned long start, size_t len, bool downgrade)
{
	int ret;
	struct mm_struct *mm = current->mm;
	LIST_HEAD(uf);
	MA_STATE(mas, &mm->mm_mt, start, start);
	if (mmap_write_lock_killable(mm))
		return -EINTR;
	ret = do_mas_munmap(&mas, mm, start, len, &uf, downgrade);
	if (ret == 1) {
		mmap_read_unlock(mm);
		ret = 0;
	} else
		mmap_write_unlock(mm);
	userfaultfd_unmap_complete(mm, &uf);
	return ret;
}
int vm_munmap(unsigned long start, size_t len)
{
	return __vm_munmap(start, len, false);
}
EXPORT_SYMBOL(vm_munmap);
SYSCALL_DEFINE2(munmap, unsigned long, addr, size_t, len)
{
	addr = untagged_addr(addr);
	return __vm_munmap(addr, len, true);
}
SYSCALL_DEFINE5(remap_file_pages, unsigned long, start, unsigned long, size,
		unsigned long, prot, unsigned long, pgoff, unsigned long, flags)
{
	struct mm_struct *mm = current->mm;
	struct vm_area_struct *vma;
	unsigned long populate = 0;
	unsigned long ret = -EINVAL;
	struct file *file;
	pr_warn_once("%s (%d) uses deprecated remap_file_pages() syscall. See Documentation/mm/remap_file_pages.rst.\n",
		     current->comm, current->pid);
	if (prot)
		return ret;
	start = start & PAGE_MASK;
	size = size & PAGE_MASK;
	if (start + size <= start)
		return ret;
	if (pgoff + (size >> PAGE_SHIFT) < pgoff)
		return ret;
	if (mmap_write_lock_killable(mm))
		return -EINTR;
	vma = vma_lookup(mm, start);
	if (!vma || !(vma->vm_flags & VM_SHARED))
		goto out;
	if (start + size > vma->vm_end) {
		VMA_ITERATOR(vmi, mm, vma->vm_end);
		struct vm_area_struct *next, *prev = vma;
		for_each_vma_range(vmi, next, start + size) {
			if (next->vm_start != prev->vm_end)
				goto out;
			if (next->vm_file != vma->vm_file)
				goto out;
			if (next->vm_flags != vma->vm_flags)
				goto out;
			if (start + size <= next->vm_end)
				break;
			prev = next;
		}
		if (!next)
			goto out;
	}
	prot |= vma->vm_flags & VM_READ ? PROT_READ : 0;
	prot |= vma->vm_flags & VM_WRITE ? PROT_WRITE : 0;
	prot |= vma->vm_flags & VM_EXEC ? PROT_EXEC : 0;
	flags &= MAP_NONBLOCK;
	flags |= MAP_SHARED | MAP_FIXED | MAP_POPULATE;
	if (vma->vm_flags & VM_LOCKED)
		flags |= MAP_LOCKED;
	file = get_file(vma->vm_file);
	ret = do_mmap(vma->vm_file, start, size,
			prot, flags, pgoff, &populate, NULL);
	fput(file);
out:
	mmap_write_unlock(mm);
	if (populate)
		mm_populate(ret, populate);
	if (!IS_ERR_VALUE(ret))
		ret = 0;
	return ret;
}
static int do_brk_munmap(struct ma_state *mas, struct vm_area_struct *vma,
			 unsigned long newbrk, unsigned long oldbrk,
			 struct list_head *uf)
{
	struct mm_struct *mm = vma->vm_mm;
	int ret;
	arch_unmap(mm, newbrk, oldbrk);
	ret = do_mas_align_munmap(mas, vma, mm, newbrk, oldbrk, uf, true);
	validate_mm_mt(mm);
	return ret;
}
static int do_brk_flags(struct ma_state *mas, struct vm_area_struct *vma,
		unsigned long addr, unsigned long len, unsigned long flags)
{
	struct mm_struct *mm = current->mm;
	validate_mm_mt(mm);
	flags |= VM_DATA_DEFAULT_FLAGS | VM_ACCOUNT | mm->def_flags;
	if (!may_expand_vm(mm, flags, len >> PAGE_SHIFT))
		return -ENOMEM;
	if (mm->map_count > sysctl_max_map_count)
		return -ENOMEM;
	if (security_vm_enough_memory_mm(mm, len >> PAGE_SHIFT))
		return -ENOMEM;
	if (vma && vma->vm_end == addr && !vma_policy(vma) &&
	    can_vma_merge_after(vma, flags, NULL, NULL,
				addr >> PAGE_SHIFT, NULL_VM_UFFD_CTX, NULL)) {
		mas_set_range(mas, vma->vm_start, addr + len - 1);
		if (mas_preallocate(mas, vma, GFP_KERNEL))
			return -ENOMEM;
		vma_adjust_trans_huge(vma, vma->vm_start, addr + len, 0);
		if (vma->anon_vma) {
			anon_vma_lock_write(vma->anon_vma);
			anon_vma_interval_tree_pre_update_vma(vma);
		}
		vma->vm_end = addr + len;
		vma->vm_flags |= VM_SOFTDIRTY;
		mas_store_prealloc(mas, vma);
		if (vma->anon_vma) {
			anon_vma_interval_tree_post_update_vma(vma);
			anon_vma_unlock_write(vma->anon_vma);
		}
		khugepaged_enter_vma(vma, flags);
		goto out;
	}
	vma = vm_area_alloc(mm);
	if (!vma)
		goto vma_alloc_fail;
	vma_set_anonymous(vma);
	vma->vm_start = addr;
	vma->vm_end = addr + len;
	vma->vm_pgoff = addr >> PAGE_SHIFT;
	vma->vm_flags = flags;
	vma->vm_page_prot = vm_get_page_prot(flags);
	mas_set_range(mas, vma->vm_start, addr + len - 1);
	if (mas_store_gfp(mas, vma, GFP_KERNEL))
		goto mas_store_fail;
	mm->map_count++;
out:
	perf_event_mmap(vma);
	mm->total_vm += len >> PAGE_SHIFT;
	mm->data_vm += len >> PAGE_SHIFT;
	if (flags & VM_LOCKED)
		mm->locked_vm += (len >> PAGE_SHIFT);
	vma->vm_flags |= VM_SOFTDIRTY;
	validate_mm(mm);
	return 0;
mas_store_fail:
	vm_area_free(vma);
vma_alloc_fail:
	vm_unacct_memory(len >> PAGE_SHIFT);
	return -ENOMEM;
}
int vm_brk_flags(unsigned long addr, unsigned long request, unsigned long flags)
{
	struct mm_struct *mm = current->mm;
	struct vm_area_struct *vma = NULL;
	unsigned long len;
	int ret;
	bool populate;
	LIST_HEAD(uf);
	MA_STATE(mas, &mm->mm_mt, addr, addr);
	len = PAGE_ALIGN(request);
	if (len < request)
		return -ENOMEM;
	if (!len)
		return 0;
	if (mmap_write_lock_killable(mm))
		return -EINTR;
	if ((flags & (~VM_EXEC)) != 0)
		return -EINVAL;
	ret = check_brk_limits(addr, len);
	if (ret)
		goto limits_failed;
	ret = do_mas_munmap(&mas, mm, addr, len, &uf, 0);
	if (ret)
		goto munmap_failed;
	vma = mas_prev(&mas, 0);
	ret = do_brk_flags(&mas, vma, addr, len, flags);
	populate = ((mm->def_flags & VM_LOCKED) != 0);
	mmap_write_unlock(mm);
	userfaultfd_unmap_complete(mm, &uf);
	if (populate && !ret)
		mm_populate(addr, len);
	return ret;
munmap_failed:
limits_failed:
	mmap_write_unlock(mm);
	return ret;
}
EXPORT_SYMBOL(vm_brk_flags);
int vm_brk(unsigned long addr, unsigned long len)
{
	return vm_brk_flags(addr, len, 0);
}
EXPORT_SYMBOL(vm_brk);
void exit_mmap(struct mm_struct *mm)
{
	struct mmu_gather tlb;
	struct vm_area_struct *vma;
	unsigned long nr_accounted = 0;
	MA_STATE(mas, &mm->mm_mt, 0, 0);
	int count = 0;
	mmu_notifier_release(mm);
	mmap_read_lock(mm);
	arch_exit_mmap(mm);
	vma = mas_find(&mas, ULONG_MAX);
	if (!vma) {
		mmap_read_unlock(mm);
		return;
	}
	lru_add_drain();
	flush_cache_mm(mm);
	tlb_gather_mmu_fullmm(&tlb, mm);
	unmap_vmas(&tlb, &mm->mm_mt, vma, 0, ULONG_MAX);
	mmap_read_unlock(mm);
	set_bit(MMF_OOM_SKIP, &mm->flags);
	mmap_write_lock(mm);
	free_pgtables(&tlb, &mm->mm_mt, vma, FIRST_USER_ADDRESS,
		      USER_PGTABLES_CEILING);
	tlb_finish_mmu(&tlb);
	do {
		if (vma->vm_flags & VM_ACCOUNT)
			nr_accounted += vma_pages(vma);
		remove_vma(vma);
		count++;
		cond_resched();
	} while ((vma = mas_find(&mas, ULONG_MAX)) != NULL);
	BUG_ON(count != mm->map_count);
	trace_exit_mmap(mm);
	__mt_destroy(&mm->mm_mt);
	mmap_write_unlock(mm);
	vm_unacct_memory(nr_accounted);
}
int insert_vm_struct(struct mm_struct *mm, struct vm_area_struct *vma)
{
	unsigned long charged = vma_pages(vma);
	if (find_vma_intersection(mm, vma->vm_start, vma->vm_end))
		return -ENOMEM;
	if ((vma->vm_flags & VM_ACCOUNT) &&
	     security_vm_enough_memory_mm(mm, charged))
		return -ENOMEM;
	if (vma_is_anonymous(vma)) {
		BUG_ON(vma->anon_vma);
		vma->vm_pgoff = vma->vm_start >> PAGE_SHIFT;
	}
	if (vma_link(mm, vma)) {
		vm_unacct_memory(charged);
		return -ENOMEM;
	}
	return 0;
}
struct vm_area_struct *copy_vma(struct vm_area_struct **vmap,
	unsigned long addr, unsigned long len, pgoff_t pgoff,
	bool *need_rmap_locks)
{
	struct vm_area_struct *vma = *vmap;
	unsigned long vma_start = vma->vm_start;
	struct mm_struct *mm = vma->vm_mm;
	struct vm_area_struct *new_vma, *prev;
	bool faulted_in_anon_vma = true;
	validate_mm_mt(mm);
	if (unlikely(vma_is_anonymous(vma) && !vma->anon_vma)) {
		pgoff = addr >> PAGE_SHIFT;
		faulted_in_anon_vma = false;
	}
	new_vma = find_vma_prev(mm, addr, &prev);
	if (new_vma && new_vma->vm_start < addr + len)
		return NULL;
	new_vma = vma_merge(mm, prev, addr, addr + len, vma->vm_flags,
			    vma->anon_vma, vma->vm_file, pgoff, vma_policy(vma),
			    vma->vm_userfaultfd_ctx, anon_vma_name(vma));
	if (new_vma) {
		if (unlikely(vma_start >= new_vma->vm_start &&
			     vma_start < new_vma->vm_end)) {
			VM_BUG_ON_VMA(faulted_in_anon_vma, new_vma);
			*vmap = vma = new_vma;
		}
		*need_rmap_locks = (new_vma->vm_pgoff <= vma->vm_pgoff);
	} else {
		new_vma = vm_area_dup(vma);
		if (!new_vma)
			goto out;
		new_vma->vm_start = addr;
		new_vma->vm_end = addr + len;
		new_vma->vm_pgoff = pgoff;
		if (vma_dup_policy(vma, new_vma))
			goto out_free_vma;
		if (anon_vma_clone(new_vma, vma))
			goto out_free_mempol;
		if (new_vma->vm_file)
			get_file(new_vma->vm_file);
		if (new_vma->vm_ops && new_vma->vm_ops->open)
			new_vma->vm_ops->open(new_vma);
		if (vma_link(mm, new_vma))
			goto out_vma_link;
		*need_rmap_locks = false;
	}
	validate_mm_mt(mm);
	return new_vma;
out_vma_link:
	if (new_vma->vm_ops && new_vma->vm_ops->close)
		new_vma->vm_ops->close(new_vma);
	if (new_vma->vm_file)
		fput(new_vma->vm_file);
	unlink_anon_vmas(new_vma);
out_free_mempol:
	mpol_put(vma_policy(new_vma));
out_free_vma:
	vm_area_free(new_vma);
out:
	validate_mm_mt(mm);
	return NULL;
}
bool may_expand_vm(struct mm_struct *mm, vm_flags_t flags, unsigned long npages)
{
	if (mm->total_vm + npages > rlimit(RLIMIT_AS) >> PAGE_SHIFT)
		return false;
	if (is_data_mapping(flags) &&
	    mm->data_vm + npages > rlimit(RLIMIT_DATA) >> PAGE_SHIFT) {
		if (rlimit(RLIMIT_DATA) == 0 &&
		    mm->data_vm + npages <= rlimit_max(RLIMIT_DATA) >> PAGE_SHIFT)
			return true;
		pr_warn_once("%s (%d): VmData %lu exceed data ulimit %lu. Update limits%s.\n",
			     current->comm, current->pid,
			     (mm->data_vm + npages) << PAGE_SHIFT,
			     rlimit(RLIMIT_DATA),
			     ignore_rlimit_data ? "" : " or use boot option ignore_rlimit_data");
		if (!ignore_rlimit_data)
			return false;
	}
	return true;
}
void vm_stat_account(struct mm_struct *mm, vm_flags_t flags, long npages)
{
	WRITE_ONCE(mm->total_vm, READ_ONCE(mm->total_vm)+npages);
	if (is_exec_mapping(flags))
		mm->exec_vm += npages;
	else if (is_stack_mapping(flags))
		mm->stack_vm += npages;
	else if (is_data_mapping(flags))
		mm->data_vm += npages;
}
static vm_fault_t special_mapping_fault(struct vm_fault *vmf);
static void special_mapping_close(struct vm_area_struct *vma)
{
}
static const char *special_mapping_name(struct vm_area_struct *vma)
{
	return ((struct vm_special_mapping *)vma->vm_private_data)->name;
}
static int special_mapping_mremap(struct vm_area_struct *new_vma)
{
	struct vm_special_mapping *sm = new_vma->vm_private_data;
	if (WARN_ON_ONCE(current->mm != new_vma->vm_mm))
		return -EFAULT;
	if (sm->mremap)
		return sm->mremap(sm, new_vma);
	return 0;
}
static int special_mapping_split(struct vm_area_struct *vma, unsigned long addr)
{
	return -EINVAL;
}
static const struct vm_operations_struct special_mapping_vmops = {
	.close = special_mapping_close,
	.fault = special_mapping_fault,
	.mremap = special_mapping_mremap,
	.name = special_mapping_name,
	.access = NULL,
	.may_split = special_mapping_split,
};
static const struct vm_operations_struct legacy_special_mapping_vmops = {
	.close = special_mapping_close,
	.fault = special_mapping_fault,
};
static vm_fault_t special_mapping_fault(struct vm_fault *vmf)
{
	struct vm_area_struct *vma = vmf->vma;
	pgoff_t pgoff;
	struct page **pages;
	if (vma->vm_ops == &legacy_special_mapping_vmops) {
		pages = vma->vm_private_data;
	} else {
		struct vm_special_mapping *sm = vma->vm_private_data;
		if (sm->fault)
			return sm->fault(sm, vmf->vma, vmf);
		pages = sm->pages;
	}
	for (pgoff = vmf->pgoff; pgoff && *pages; ++pages)
		pgoff--;
	if (*pages) {
		struct page *page = *pages;
		get_page(page);
		vmf->page = page;
		return 0;
	}
	return VM_FAULT_SIGBUS;
}
static struct vm_area_struct *__install_special_mapping(
	struct mm_struct *mm,
	unsigned long addr, unsigned long len,
	unsigned long vm_flags, void *priv,
	const struct vm_operations_struct *ops)
{
	int ret;
	struct vm_area_struct *vma;
	validate_mm_mt(mm);
	vma = vm_area_alloc(mm);
	if (unlikely(vma == NULL))
		return ERR_PTR(-ENOMEM);
	vma->vm_start = addr;
	vma->vm_end = addr + len;
	vma->vm_flags = vm_flags | mm->def_flags | VM_DONTEXPAND | VM_SOFTDIRTY;
	vma->vm_flags &= VM_LOCKED_CLEAR_MASK;
	vma->vm_page_prot = vm_get_page_prot(vma->vm_flags);
	vma->vm_ops = ops;
	vma->vm_private_data = priv;
	ret = insert_vm_struct(mm, vma);
	if (ret)
		goto out;
	vm_stat_account(mm, vma->vm_flags, len >> PAGE_SHIFT);
	perf_event_mmap(vma);
	validate_mm_mt(mm);
	return vma;
out:
	vm_area_free(vma);
	validate_mm_mt(mm);
	return ERR_PTR(ret);
}
bool vma_is_special_mapping(const struct vm_area_struct *vma,
	const struct vm_special_mapping *sm)
{
	return vma->vm_private_data == sm &&
		(vma->vm_ops == &special_mapping_vmops ||
		 vma->vm_ops == &legacy_special_mapping_vmops);
}
struct vm_area_struct *_install_special_mapping(
	struct mm_struct *mm,
	unsigned long addr, unsigned long len,
	unsigned long vm_flags, const struct vm_special_mapping *spec)
{
	return __install_special_mapping(mm, addr, len, vm_flags, (void *)spec,
					&special_mapping_vmops);
}
int install_special_mapping(struct mm_struct *mm,
			    unsigned long addr, unsigned long len,
			    unsigned long vm_flags, struct page **pages)
{
	struct vm_area_struct *vma = __install_special_mapping(
		mm, addr, len, vm_flags, (void *)pages,
		&legacy_special_mapping_vmops);
	return PTR_ERR_OR_ZERO(vma);
}
static DEFINE_MUTEX(mm_all_locks_mutex);
static void vm_lock_anon_vma(struct mm_struct *mm, struct anon_vma *anon_vma)
{
	if (!test_bit(0, (unsigned long *) &anon_vma->root->rb_root.rb_root.rb_node)) {
		down_write_nest_lock(&anon_vma->root->rwsem, &mm->mmap_lock);
		if (__test_and_set_bit(0, (unsigned long *)
				       &anon_vma->root->rb_root.rb_root.rb_node))
			BUG();
	}
}
static void vm_lock_mapping(struct mm_struct *mm, struct address_space *mapping)
{
	if (!test_bit(AS_MM_ALL_LOCKS, &mapping->flags)) {
		if (test_and_set_bit(AS_MM_ALL_LOCKS, &mapping->flags))
			BUG();
		down_write_nest_lock(&mapping->i_mmap_rwsem, &mm->mmap_lock);
	}
}
int mm_take_all_locks(struct mm_struct *mm)
{
	struct vm_area_struct *vma;
	struct anon_vma_chain *avc;
	MA_STATE(mas, &mm->mm_mt, 0, 0);
	mmap_assert_write_locked(mm);
	mutex_lock(&mm_all_locks_mutex);
	mas_for_each(&mas, vma, ULONG_MAX) {
		if (signal_pending(current))
			goto out_unlock;
		if (vma->vm_file && vma->vm_file->f_mapping &&
				is_vm_hugetlb_page(vma))
			vm_lock_mapping(mm, vma->vm_file->f_mapping);
	}
	mas_set(&mas, 0);
	mas_for_each(&mas, vma, ULONG_MAX) {
		if (signal_pending(current))
			goto out_unlock;
		if (vma->vm_file && vma->vm_file->f_mapping &&
				!is_vm_hugetlb_page(vma))
			vm_lock_mapping(mm, vma->vm_file->f_mapping);
	}
	mas_set(&mas, 0);
	mas_for_each(&mas, vma, ULONG_MAX) {
		if (signal_pending(current))
			goto out_unlock;
		if (vma->anon_vma)
			list_for_each_entry(avc, &vma->anon_vma_chain, same_vma)
				vm_lock_anon_vma(mm, avc->anon_vma);
	}
	return 0;
out_unlock:
	mm_drop_all_locks(mm);
	return -EINTR;
}
static void vm_unlock_anon_vma(struct anon_vma *anon_vma)
{
	if (test_bit(0, (unsigned long *) &anon_vma->root->rb_root.rb_root.rb_node)) {
		if (!__test_and_clear_bit(0, (unsigned long *)
					  &anon_vma->root->rb_root.rb_root.rb_node))
			BUG();
		anon_vma_unlock_write(anon_vma);
	}
}
static void vm_unlock_mapping(struct address_space *mapping)
{
	if (test_bit(AS_MM_ALL_LOCKS, &mapping->flags)) {
		i_mmap_unlock_write(mapping);
		if (!test_and_clear_bit(AS_MM_ALL_LOCKS,
					&mapping->flags))
			BUG();
	}
}
void mm_drop_all_locks(struct mm_struct *mm)
{
	struct vm_area_struct *vma;
	struct anon_vma_chain *avc;
	MA_STATE(mas, &mm->mm_mt, 0, 0);
	mmap_assert_write_locked(mm);
	BUG_ON(!mutex_is_locked(&mm_all_locks_mutex));
	mas_for_each(&mas, vma, ULONG_MAX) {
		if (vma->anon_vma)
			list_for_each_entry(avc, &vma->anon_vma_chain, same_vma)
				vm_unlock_anon_vma(avc->anon_vma);
		if (vma->vm_file && vma->vm_file->f_mapping)
			vm_unlock_mapping(vma->vm_file->f_mapping);
	}
	mutex_unlock(&mm_all_locks_mutex);
}
void __init mmap_init(void)
{
	int ret;
	ret = percpu_counter_init(&vm_committed_as, 0, GFP_KERNEL);
	VM_BUG_ON(ret);
}
static int init_user_reserve(void)
{
	unsigned long free_kbytes;
	free_kbytes = global_zone_page_state(NR_FREE_PAGES) << (PAGE_SHIFT - 10);
	sysctl_user_reserve_kbytes = min(free_kbytes / 32, 1UL << 17);
	return 0;
}
subsys_initcall(init_user_reserve);
static int init_admin_reserve(void)
{
	unsigned long free_kbytes;
	free_kbytes = global_zone_page_state(NR_FREE_PAGES) << (PAGE_SHIFT - 10);
	sysctl_admin_reserve_kbytes = min(free_kbytes / 32, 1UL << 13);
	return 0;
}
subsys_initcall(init_admin_reserve);
static int reserve_mem_notifier(struct notifier_block *nb,
			     unsigned long action, void *data)
{
	unsigned long tmp, free_kbytes;
	switch (action) {
	case MEM_ONLINE:
		tmp = sysctl_user_reserve_kbytes;
		if (0 < tmp && tmp < (1UL << 17))
			init_user_reserve();
		tmp = sysctl_admin_reserve_kbytes;
		if (0 < tmp && tmp < (1UL << 13))
			init_admin_reserve();
		break;
	case MEM_OFFLINE:
		free_kbytes = global_zone_page_state(NR_FREE_PAGES) << (PAGE_SHIFT - 10);
		if (sysctl_user_reserve_kbytes > free_kbytes) {
			init_user_reserve();
			pr_info("vm.user_reserve_kbytes reset to %lu\n",
				sysctl_user_reserve_kbytes);
		}
		if (sysctl_admin_reserve_kbytes > free_kbytes) {
			init_admin_reserve();
			pr_info("vm.admin_reserve_kbytes reset to %lu\n",
				sysctl_admin_reserve_kbytes);
		}
		break;
	default:
		break;
	}
	return NOTIFY_OK;
}
static struct notifier_block reserve_mem_nb = {
	.notifier_call = reserve_mem_notifier,
};
static int __meminit init_reserve_notifier(void)
{
	if (register_hotmemory_notifier(&reserve_mem_nb))
		pr_err("Failed registering memory add/remove notifier for admin reserve\n");
	return 0;
}
subsys_initcall(init_reserve_notifier);
