#!/usr/bin/env python3
"""
DDP Initialization Test

Tests the exact sequence that happens during training startup:
1. Import torch
2. Initialize CUDA
3. Import DDP
4. Initialize process group (NCCL)
5. Load model
6. Wrap in DDP
7. Create dataloader with batch_size
8. First forward pass

Logs extensively to detect where crash occurs.
"""

import argparse
import os
import sys
import time
import datetime
from pathlib import Path

class CrashSafeLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.start_time = time.time()
        self.f = open(log_path, 'a', buffering=1)
        self._log(f"{'='*60}")
        self._log(f"DDP INIT TEST STARTED")
        self._log(f"Timestamp: {datetime.datetime.now().isoformat()}")
        self._log(f"PID: {os.getpid()}")
        self._log(f"{'='*60}")

    def _log(self, msg: str):
        elapsed = time.time() - self.start_time
        line = f"[{elapsed:8.3f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def log(self, msg: str):
        self._log(msg)

    def section(self, title: str):
        self._log("")
        self._log(f">>> {title}")
        self._log("-" * 40)

    def close(self):
        self._log("Logger closing normally")
        self.f.close()


def get_gpu_memory_info():
    import torch
    info = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        info.append(f"GPU{i}: {allocated:.2f}GB alloc / {reserved:.2f}GB reserved / {total:.2f}GB total")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--log_dir", type=str, default="runs/ddp_init_test")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{args.log_dir}/ddp_init_bs{args.batch_size}_{timestamp}.log"
    log = CrashSafeLogger(log_path)

    log.log(f"batch_size={args.batch_size}, seq_len={args.seq_len}")

    # Phase 1: Import torch
    log.section("PHASE 1: Import torch")
    import torch
    log.log(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    log.log("Phase 1 PASSED")

    # Phase 2: Initialize CUDA contexts
    log.section("PHASE 2: Initialize CUDA")
    n_gpus = torch.cuda.device_count()
    log.log(f"Found {n_gpus} GPUs")
    for i in range(n_gpus):
        log.log(f"  Initializing GPU{i}...")
        torch.cuda.set_device(i)
        _ = torch.zeros(1, device=f'cuda:{i}')
        log.log(f"  GPU{i} ready")
    for info in get_gpu_memory_info():
        log.log(f"  {info}")
    log.log("Phase 2 PASSED")

    # Phase 3: Import distributed
    log.section("PHASE 3: Import distributed")
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    log.log("Distributed imports complete")
    log.log("Phase 3 PASSED")

    # Phase 4: Initialize process group (NCCL - uses NVLink)
    log.section("PHASE 4: Initialize NCCL process group")
    log.log("Setting environment variables...")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'  # Different port to avoid conflicts
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'

    log.log("Calling dist.init_process_group(backend='nccl')...")
    dist.init_process_group(backend='nccl', world_size=1, rank=0)
    log.log(f"Process group initialized: {dist.is_initialized()}")
    log.log("Phase 4 PASSED")

    # Phase 5: Import model
    log.section("PHASE 5: Import Atlas model")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.model.atlas import Atlas, AtlasConfig
    log.log("Model imported")
    log.log("Phase 5 PASSED")

    # Phase 6: Create model on GPU0
    log.section("PHASE 6: Create model")
    config = AtlasConfig(
        d_model=512,
        n_layers=8,
        n_heads=4,
        d_ff=2048,
        vocab_size=32100,
        max_seq_len=args.seq_len,
    )
    log.log(f"Config: {config}")

    log.log("Instantiating model...")
    model = Atlas(config)
    log.log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    log.log("Moving to GPU0...")
    for info in get_gpu_memory_info():
        log.log(f"  Before: {info}")
    model = model.cuda(0).bfloat16()
    for info in get_gpu_memory_info():
        log.log(f"  After: {info}")
    log.log("Phase 6 PASSED")

    # Phase 7: Wrap in DDP
    log.section("PHASE 7: Wrap in DDP")
    log.log("Calling DistributedDataParallel()...")
    for info in get_gpu_memory_info():
        log.log(f"  Before: {info}")
    model = DDP(model, device_ids=[0])
    for info in get_gpu_memory_info():
        log.log(f"  After: {info}")
    log.log("Phase 7 PASSED")

    # Phase 8: Create synthetic batch
    log.section(f"PHASE 8: Create batch (batch_size={args.batch_size})")
    log.log(f"Allocating input tensor [{args.batch_size}, {args.seq_len}]...")
    for info in get_gpu_memory_info():
        log.log(f"  Before: {info}")
    input_ids = torch.randint(0, 32100, (args.batch_size, args.seq_len), device='cuda:0')
    for info in get_gpu_memory_info():
        log.log(f"  After: {info}")
    log.log(f"Input shape: {input_ids.shape}")
    log.log("Phase 8 PASSED")

    # Phase 9: Forward pass
    log.section("PHASE 9: Forward pass")
    log.log("Running forward pass...")
    for info in get_gpu_memory_info():
        log.log(f"  Before: {info}")

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output = model(input_ids)
    torch.cuda.synchronize()

    if isinstance(output, tuple):
        output = output[0]

    for info in get_gpu_memory_info():
        log.log(f"  After: {info}")
    log.log(f"Output shape: {output.shape}")
    log.log("Phase 9 PASSED")

    # Phase 10: Backward pass
    log.section("PHASE 10: Backward pass")
    loss = output.mean()
    log.log(f"Loss: {loss.item():.4f}")
    log.log("Running backward...")
    for info in get_gpu_memory_info():
        log.log(f"  Before: {info}")
    loss.backward()
    torch.cuda.synchronize()
    for info in get_gpu_memory_info():
        log.log(f"  After: {info}")
    log.log("Phase 10 PASSED")

    # Cleanup
    log.section("TEST COMPLETE")
    log.log(f"All phases passed for batch_size={args.batch_size}")
    dist.destroy_process_group()
    log.close()

    print(f"\nSUCCESS: batch_size={args.batch_size} DDP init test completed")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
