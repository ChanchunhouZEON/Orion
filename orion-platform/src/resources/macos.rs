/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! macOS physical memory and conservative reclaimable-page observations.
use super::MemoryResources;

#[allow(deprecated)] // libc retains the Mach ABI; avoid a separate OS-only dependency.
pub(super) fn observe(result: &mut MemoryResources) {
    unsafe extern "C" {
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
    let mut total = 0u64;
    let mut size = std::mem::size_of::<u64>();
    // SAFETY: the sysctl name is terminated and the output buffer matches size.
    if unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut total as *mut u64).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } == 0
    {
        result.total_bytes = Some(total);
    }
    let mut stats = std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: the Mach call initializes the documented output structure. The
    // host send right is released even if statistics retrieval fails.
    unsafe {
        let host = libc::mach_host_self();
        let status = libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            stats.as_mut_ptr().cast(),
            &mut count,
        );
        mach_port_deallocate(libc::mach_task_self(), host);
        let page = libc::sysconf(libc::_SC_PAGESIZE);
        if status == libc::KERN_SUCCESS && page > 0 {
            let stats = stats.assume_init();
            // Conservative reclaimable estimate. Do not include compressed or
            // speculative pages again (speculative pages are already free).
            result.available_bytes = (u64::from(stats.free_count)
                + u64::from(stats.inactive_count))
            .checked_mul(page as u64);
        }
    }
    // macOS has no VmLck equivalent here. Our registry is accounted for by the
    // planner; external locks may still cause a best-effort mlock refusal.
}
