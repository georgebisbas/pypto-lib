# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: fixed 2-card run; borrows 2 cards via task-submit --device-num
"""L3 multi-card mesh all-reduce (``@pl.jit`` / ``@pl.jit.host``).

Every rank contributes a row and ends up holding the element-wise sum of all
rows, computed over the L3 distributed stack: HCCL window buffers, notify/wait
barriers, and remote tile loads. Fixed at P=2 (two ranks).

For the same reduction written as a single ``pld.tensor.allreduce`` call, see
``allreduce_composite.py`` next to this file.

Run::

    python examples/advanced/allreduce.py -p a2a3 -d 0,1
    python examples/advanced/allreduce.py -p a2a3 -d 0,1 --size 65536
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld

N_RANKS = 2  # this example runs P=2 only; the window shapes need it statically


def _parse_int_argv(flag: str, default: int) -> int:
    """Read an int flag before argparse runs.

    ``SIZE`` appears in the tensor type annotations, so it has to be known at
    import time — before ``argparse`` gets a chance to run in ``__main__``.

    Args:
        flag: Flag name to look for, e.g. ``"--size"``.
        default: Value to return when the flag is absent.

    Returns:
        The parsed integer, or ``default``.
    """
    for index, arg in enumerate(sys.argv):
        if arg == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith(f"{flag}="):
            return int(arg.split("=", 1)[1])
    return default


# Element-wise reduction length per rank; 256 FP32 = 1 KB.
SIZE = _parse_int_argv("--size", 256)
if SIZE < 1:
    raise ValueError(f"--size must be positive, got {SIZE}")

# Payload chunk, in elements: 4096 FP32 = 16 KiB. Staging and reducing a chunk
# at a time keeps Vec usage at 2*CHUNK*4 bytes instead of 2*SIZE*4, so the
# example still runs at realistic payload sizes — a single [1, SIZE] tile hits
# the Vec limit (188,416 B on a2a3) at 64 KB/rank.
CHUNK = min(SIZE, 4096)
if SIZE % CHUNK:
    raise ValueError(f"--size must be a multiple of {CHUNK} (or <= it), got {SIZE}")


@pl.jit.incore
def reduce_step(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Mesh all-reduce on window-bound ``data`` / ``signal``."""
    # Stage the local input into this rank's window slice, a chunk at a time so
    # no full-size tile is ever live.
    for s0 in pl.range(0, SIZE, CHUNK):
        data = pl.store(pl.load(inp, [0, s0], [1, CHUNK]), [0, s0], data)

    # Barrier: notify every peer, then wait on every peer slot. The window
    # buffer is zero-initialised, so AtomicAdd + Ge(1) is safe.
    for peer in pl.range(N_RANKS):
        if peer != my_rank:
            pld.system.notify(
                signal,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
    for src in pl.range(N_RANKS):
        if src != my_rank:
            pld.system.wait(
                signal=signal,
                offsets=[src, 0],
                expected=1,
                cmp=pld.WaitCmp.Ge,
            )

    # One barrier above, then walk the payload a chunk at a time: load my own
    # chunk, add every peer's, write it out. The barrier count does not grow
    # with the payload — it stays outside this loop.
    for c0 in pl.range(0, SIZE, CHUNK):
        acc = pl.load(data, [0, c0], [1, CHUNK])
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                recv = pld.tile.remote_load(data, peer=peer, offsets=[0, c0], shape=[1, CHUNK])
                acc = pl.add(acc, recv)
        out = pl.store(acc, [0, c0], out)

    return out


@pl.jit
def allreduce(
    inp: pl.Tensor[[1, SIZE], pl.FP32],
    out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
    data: pld.DistributedTensor[[1, SIZE], pl.FP32],
    signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Per-device orchestration wrapper around ``reduce_step``."""
    return reduce_step(inp, out, data, signal, my_rank)


@pl.jit.host
def l3_allreduce(
    inputs: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    """Launch one chip orchestration per rank, sharing the window buffers."""
    data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        allreduce(inputs[r], outputs[r], data, signal, r, device=r)


def build_tensor_specs():
    """Distinct per-rank input rows so the reduced sum is non-trivial."""
    import torch

    from golden import TensorSpec

    def init_inputs():
        rows = [
            torch.arange(r * 100.0, r * 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE)
            for r in range(N_RANKS)
        ]
        return torch.stack(rows)

    return [
        TensorSpec("inputs",  [N_RANKS, 1, SIZE], torch.float32, init_value=init_inputs),
        TensorSpec("outputs", [N_RANKS, 1, SIZE], torch.float32),
    ]


def golden_allreduce(tensors):
    """Every rank ends up holding the element-wise sum of all rank inputs."""
    reduced = tensors["inputs"].sum(dim=0, keepdim=True)  # [1, 1, SIZE]
    tensors["outputs"][:] = reduced.expand_as(tensors["outputs"])


if __name__ == "__main__":
    import argparse

    from golden import run
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=str, default="0,1",
                        help=f"comma-separated device ids (need exactly {N_RANKS})")
    parser.add_argument("--size", type=int, default=256,
                        help="elements per rank; read at import, repeated here for --help")
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) == N_RANKS, f"need exactly {N_RANKS} devices, got {device_ids}"

    result = run(
        fn=l3_allreduce,
        specs=build_tensor_specs(),
        golden_fn=golden_allreduce,
        compile_only=args.compile_only,
        compile_cfg=dict(
            distributed_config=DistributedConfig(
                device_ids=device_ids,
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(platform=args.platform),
        rtol=1e-5,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
