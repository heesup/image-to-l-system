# Long training runs die at ~21 h to an NCCL watchdog hang, and `--requeue` does not catch it

2026-09-20. Both full-data runs failed within 41 minutes of each other after ~21 hours, at epoch
~172 of 220, with identical symptoms.

    38481342  merged_full      FAILED  1:0  21:36:09
    38486083  merged_fullaug   FAILED  1:0  20:56:10

## The failure

```
[rank0] Exception (either an error or timeout) detected by watchdog at work: 1940009,
        last enqueued NCCL work: 1940014, last completed NCCL work: 1940008.
[rank0] ProcessGroupNCCL's watchdog got stuck for 480 seconds without making progress
        in monitoring enqueued collectives.
torch.distributed.elastic.multiprocessing.errors.ChildFailedError
```

The watchdog, not the training step, is what got stuck. PyTorch's own message attributes this to a
CUDA API call blocking the watchdog thread, typically another thread holding the GIL inside a CUDA
call. These are **single-GPU** jobs that still construct DDP (with `find_unused_parameters=True`,
which the run logs already warn is doing an extra autograd traversal for no benefit here).

## Two things that did not save us

**`--requeue` did not fire.** It covers preemption and node failure; a job whose process exits
non-zero is a normal failure and Slurm does not requeue it. The `--requeue` on these submissions was
doing nothing for this class of fault.

**`AUTO_RESUME=1` only helps on a restart someone has to perform.** It does pick the newest raw
checkpoint out of `OUTPUT_DIR` correctly — both jobs came back at ep170 — but nothing restarts the
job automatically, so the runs simply sat dead until noticed.

The combination is worth stating plainly: the two flags together read like "this run survives
failures", and for this failure mode neither does anything.

## Mitigation applied

Restarted both with `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800` (up from the 480 s default), which is
what the error message itself suggests when the watchdog may be reporting a slow call rather than a
real deadlock. With `SAVE_EVERY=5` the exposure is now at most five epochs (~5 h on full data).

`TORCH_NCCL_ENABLE_MONITORING=0` was considered and rejected: it would convert a genuine deadlock
from a fast failure into a silent hang burning the full 3-day wall limit.

## The real fix, not applied yet

**A single-GPU job should not be building a process group at all.** No DDP means no NCCL, no
watchdog, and no `find_unused_parameters` traversal. That is a change to the launcher's distributed
setup, and three long runs are currently in flight on this exact code path, so it was deliberately
not made mid-flight. Do it when the queue is clear, and pin it with a test that a one-GPU run
constructs no process group.

## What to watch

`relative_fullaug` (38495672) started later and is on the same code path with the old 480 s timeout.
If it dies around the 21-hour mark, that is this bug and not something about the relative layout;
restart it with the same env var rather than reading anything into it.
