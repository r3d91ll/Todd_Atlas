#!/usr/bin/env python3
"""
GPU/NVLink Stress Test for Atlas Training

This script tests GPU memory allocation and NVLink communication at various
batch sizes to identify the failure point. Logs extensively to file with
flush after every operation since crashes may not reach kernel logs.

Usage:
    python scripts/gpu_stress_test.py --batch_size 8
    python scripts/gpu_stress_test.py --batch_size 12
    python scripts/gpu_stress_test.py --batch_size 16
"""

import argparse
import os
import sys
import time
import datetime
from pathlib import Path

# Flush-on-write logger that survives crashes
class CrashSafeLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.start_time = time.time()
        # Open in line-buffered mode
        self.f = open(log_path, 'a', buffering=1)
        self._log(f"{'='*60}")
        self._log(f"GPU STRESS TEST STARTED")
        self._log(f"Timestamp: {datetime.datetime.now().isoformat()}")
        self._log(f"PID: {os.getpid()}")
        self._log(f"{'='*60}")

    def _log(self, msg: str):
        elapsed = time.time() - self.start_time
        line = f"[{elapsed:8.3f}s] {msg}"
        print(line)
        self.f.write(line + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())  # Force write to disk

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
    """Get current GPU memory usage."""
    import torch
    info = []
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        info.append(f"GPU{i}: {allocated:.2f}GB alloc / {reserved:.2f}GB reserved / {total:.2f}GB total")
    return info


def main():
    parser = argparse.ArgumentParser(description="GPU/NVLink stress test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="runs/stress_test")
    args = parser.parse_args()

    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{args.log_dir}/stress_test_bs{args.batch_size}_{timestamp}.log"
    log = CrashSafeLogger(log_path)

    log.log(f"Arguments: batch_size={args.batch_size}, seq_len={args.seq_len}, d_model={args.d_model}")

    # Phase 1: Basic imports
    log.section("PHASE 1: Importing PyTorch")
    try:
        import torch
        log.log(f"PyTorch version: {torch.__version__}")
        log.log(f"CUDA available: {torch.cuda.is_available()}")
        log.log(f"CUDA version: {torch.version.cuda}")
        log.log("Phase 1 PASSED")
    except Exception as e:
        log.log(f"Phase 1 FAILED: {e}")
        log.close()
        sys.exit(1)

    # Phase 2: GPU detection
    log.section("PHASE 2: Detecting GPUs")
    try:
        n_gpus = torch.cuda.device_count()
        log.log(f"Number of GPUs: {n_gpus}")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            log.log(f"  GPU{i}: {props.name}, {props.total_memory/1024**3:.1f}GB")
        log.log("Phase 2 PASSED")
    except Exception as e:
        log.log(f"Phase 2 FAILED: {e}")
        log.close()
        sys.exit(1)

    # Phase 3: Initialize CUDA contexts
    log.section("PHASE 3: Initializing CUDA contexts")
    try:
        for i in range(n_gpus):
            log.log(f"  Initializing GPU{i}...")
            torch.cuda.set_device(i)
            # Force context creation with small allocation
            _ = torch.zeros(1, device=f'cuda:{i}')
            log.log(f"  GPU{i} context initialized")
        for info in get_gpu_memory_info():
            log.log(f"  {info}")
        log.log("Phase 3 PASSED")
    except Exception as e:
        log.log(f"Phase 3 FAILED: {e}")
        log.close()
        sys.exit(1)

    # Phase 4: Test NVLink with small transfer
    log.section("PHASE 4: Testing NVLink (small transfer)")
    try:
        log.log("  Creating tensor on GPU0...")
        t0 = torch.randn(1000, 1000, device='cuda:0')
        log.log(f"  GPU0 tensor created, shape={t0.shape}")

        log.log("  Copying to GPU1 via NVLink...")
        t1 = t0.to('cuda:1')
        log.log(f"  GPU1 tensor created, shape={t1.shape}")

        log.log("  Verifying data integrity...")
        t1_back = t1.to('cuda:0')
        diff = (t0 - t1_back).abs().max().item()
        log.log(f"  Max difference after round-trip: {diff}")

        del t0, t1, t1_back
        torch.cuda.empty_cache()
        log.log("Phase 4 PASSED")
    except Exception as e:
        log.log(f"Phase 4 FAILED: {e}")
        log.close()
        sys.exit(1)

    # Phase 5: Import DDP
    log.section("PHASE 5: Importing DDP modules")
    try:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        log.log("DDP modules imported")
        log.log("Phase 5 PASSED")
    except Exception as e:
        log.log(f"Phase 5 FAILED: {e}")
        log.close()
        sys.exit(1)

    # Phase 6: Allocate batch-sized tensors on each GPU
    log.section(f"PHASE 6: Allocating batch tensors (batch_size={args.batch_size})")
    try:
        tensors = []
        for i in range(n_gpus):
            log.log(f"  Allocating on GPU{i}...")
            log.log(f"    Shape: [{args.batch_size}, {args.seq_len}, {args.d_model}]")

            # Log memory before
            for info in get_gpu_memory_info():
                log.log(f"    Before: {info}")

            t = torch.randn(args.batch_size, args.seq_len, args.d_model,
                           device=f'cuda:{i}', dtype=torch.bfloat16)
            tensors.append(t)

            # Log memory after
            for info in get_gpu_memory_info():
                log.log(f"    After: {info}")

            log.log(f"  GPU{i} allocation complete")

        log.log("Phase 6 PASSED")
    except Exception as e:
        log.log(f"Phase 6 FAILED: {e}")
        log.close()
        sys.exit(1)

    # Phase 7: Test NVLink with batch-sized transfer
    log.section("PHASE 7: Testing NVLink (batch-sized transfer)")
    try:
        log.log(f"  Transferring {tensors[0].numel() * 2 / 1024**2:.1f}MB GPU0 -> GPU1...")
        t_start = time.time()
        t1_copy = tensors[0].to('cuda:1')
        torch.cuda.synchronize()
        t_elapsed = time.time() - t_start
        log.log(f"  Transfer complete in {t_elapsed*1000:.1f}ms")
        log.log(f"  Bandwidth: {tensors[0].numel() * 2 / t_elapsed / 1024**3:.2f} GB/s")

        del t1_copy
        torch.cuda.empty_cache()
        log.log("Phase 7 PASSED")
    except Exception as e:
        log.log(f"Phase 7 FAILED: {e}")
        log.close()
        sys.exit(1)

    # Phase 8: Import model
    log.section("PHASE 8: Importing Atlas model")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.model.atlas import Atlas, AtlasConfig
        log.log("Atlas model imported")
        log.log("Phase 8 PASSED")
    except Exception as e:
        log.log(f"Phase 8 FAILED: {e}")
        log.close()
        sys.exit(1)

    # Phase 9: Create model on GPU0
    log.section("PHASE 9: Creating model on GPU0")
    try:
        log.log("  Creating config...")
        config = AtlasConfig(
            d_model=args.d_model,
            n_layers=8,
            n_heads=4,
            d_ff=2048,
            vocab_size=32100,
            max_seq_len=args.seq_len,
        )
        log.log(f"  Config created: {config}")

        log.log("  Instantiating model...")
        model = Atlas(config)
        log.log(f"  Model created, params: {sum(p.numel() for p in model.parameters()):,}")

        log.log("  Moving model to GPU0...")
        for info in get_gpu_memory_info():
            log.log(f"    Before: {info}")

        model = model.cuda(0).bfloat16()

        for info in get_gpu_memory_info():
            log.log(f"    After: {info}")

        log.log("Phase 9 PASSED")
    except Exception as e:
        log.log(f"Phase 9 FAILED: {e}")
        import traceback
        log.log(traceback.format_exc())
        log.close()
        sys.exit(1)

    # Phase 10: Create model copy on GPU1
    log.section("PHASE 10: Creating model on GPU1")
    try:
        log.log("  Instantiating second model...")
        model2 = Atlas(config)

        log.log("  Moving model to GPU1...")
        for info in get_gpu_memory_info():
            log.log(f"    Before: {info}")

        model2 = model2.cuda(1).bfloat16()

        for info in get_gpu_memory_info():
            log.log(f"    After: {info}")

        log.log("Phase 10 PASSED")
    except Exception as e:
        log.log(f"Phase 10 FAILED: {e}")
        import traceback
        log.log(traceback.format_exc())
        log.close()
        sys.exit(1)

    # Phase 11: Forward pass on GPU0
    log.section("PHASE 11: Forward pass on GPU0")
    try:
        log.log("  Creating input tensor...")
        input_ids = torch.randint(0, 32100, (args.batch_size, args.seq_len), device='cuda:0')
        log.log(f"  Input shape: {input_ids.shape}")

        log.log("  Running forward pass...")
        for info in get_gpu_memory_info():
            log.log(f"    Before: {info}")

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = model(input_ids)
        torch.cuda.synchronize()

        # Handle tuple output (logits, memory_state)
        if isinstance(output, tuple):
            output = output[0]

        for info in get_gpu_memory_info():
            log.log(f"    After: {info}")

        log.log(f"  Output shape: {output.shape}")
        log.log("Phase 11 PASSED")
    except Exception as e:
        log.log(f"Phase 11 FAILED: {e}")
        import traceback
        log.log(traceback.format_exc())
        log.close()
        sys.exit(1)

    # Phase 12: Forward pass on GPU1
    log.section("PHASE 12: Forward pass on GPU1")
    try:
        log.log("  Creating input tensor...")
        input_ids2 = torch.randint(0, 32100, (args.batch_size, args.seq_len), device='cuda:1')
        log.log(f"  Input shape: {input_ids2.shape}")

        log.log("  Running forward pass...")
        for info in get_gpu_memory_info():
            log.log(f"    Before: {info}")

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output2 = model2(input_ids2)
        torch.cuda.synchronize()

        # Handle tuple output (logits, memory_state)
        if isinstance(output2, tuple):
            output2 = output2[0]

        for info in get_gpu_memory_info():
            log.log(f"    After: {info}")

        log.log(f"  Output shape: {output2.shape}")
        log.log("Phase 12 PASSED")
    except Exception as e:
        log.log(f"Phase 12 FAILED: {e}")
        import traceback
        log.log(traceback.format_exc())
        log.close()
        sys.exit(1)

    # Phase 13: Backward pass on both GPUs
    log.section("PHASE 13: Backward pass on both GPUs")
    try:
        log.log("  Computing loss on GPU0...")
        loss0 = output.mean()
        log.log(f"  Loss0: {loss0.item():.4f}")

        log.log("  Computing loss on GPU1...")
        loss1 = output2.mean()
        log.log(f"  Loss1: {loss1.item():.4f}")

        log.log("  Backward on GPU0...")
        for info in get_gpu_memory_info():
            log.log(f"    Before: {info}")
        loss0.backward()
        torch.cuda.synchronize()
        for info in get_gpu_memory_info():
            log.log(f"    After GPU0 backward: {info}")

        log.log("  Backward on GPU1...")
        loss1.backward()
        torch.cuda.synchronize()
        for info in get_gpu_memory_info():
            log.log(f"    After GPU1 backward: {info}")

        log.log("Phase 13 PASSED")
    except Exception as e:
        log.log(f"Phase 13 FAILED: {e}")
        import traceback
        log.log(traceback.format_exc())
        log.close()
        sys.exit(1)

    # Phase 14: Simulate gradient sync via NVLink
    log.section("PHASE 14: Simulating gradient sync (NVLink stress)")
    try:
        log.log("  Collecting gradients from GPU0...")
        grads0 = [p.grad for p in model.parameters() if p.grad is not None]
        log.log(f"  Found {len(grads0)} gradient tensors")

        total_grad_bytes = sum(g.numel() * 2 for g in grads0)  # bf16 = 2 bytes
        log.log(f"  Total gradient size: {total_grad_bytes / 1024**2:.1f}MB")

        log.log("  Transferring gradients GPU0 -> GPU1...")
        t_start = time.time()
        for i, g in enumerate(grads0):
            g_copy = g.to('cuda:1')
            if i % 20 == 0:
                log.log(f"    Transferred {i+1}/{len(grads0)} tensors")
        torch.cuda.synchronize()
        t_elapsed = time.time() - t_start
        log.log(f"  Transfer complete in {t_elapsed*1000:.1f}ms")
        log.log(f"  Bandwidth: {total_grad_bytes / t_elapsed / 1024**3:.2f} GB/s")

        log.log("Phase 14 PASSED")
    except Exception as e:
        log.log(f"Phase 14 FAILED: {e}")
        import traceback
        log.log(traceback.format_exc())
        log.close()
        sys.exit(1)

    # Cleanup before NVLink stress test
    log.section("CLEANUP: Clearing memory before NVLink stress")
    try:
        del model, model2, output, output2, input_ids, input_ids2, loss0, loss1, grads0
    except:
        pass
    torch.cuda.empty_cache()
    for info in get_gpu_memory_info():
        log.log(f"  {info}")
    log.log("Cleanup complete")

    # Phase 15: Heavy NVLink stress test (without model)
    log.section(f"PHASE 15: Heavy NVLink stress test ({args.n_steps} iterations)")
    try:
        # Allocate large tensors to stress NVLink
        tensor_size_gb = args.batch_size * args.seq_len * args.d_model * 2 / 1024**3  # bf16
        log.log(f"  Tensor size: {tensor_size_gb:.2f}GB per GPU")

        for step in range(args.n_steps):
            log.log(f"  Step {step+1}/{args.n_steps}...")

            # Allocate on GPU0
            log.log(f"    Allocating {tensor_size_gb:.2f}GB on GPU0...")
            t0 = torch.randn(args.batch_size, args.seq_len, args.d_model,
                            device='cuda:0', dtype=torch.bfloat16)
            torch.cuda.synchronize()

            # Allocate on GPU1
            log.log(f"    Allocating {tensor_size_gb:.2f}GB on GPU1...")
            t1 = torch.randn(args.batch_size, args.seq_len, args.d_model,
                            device='cuda:1', dtype=torch.bfloat16)
            torch.cuda.synchronize()

            for info in get_gpu_memory_info():
                log.log(f"    {info}")

            # Transfer GPU0 -> GPU1
            log.log(f"    Transferring GPU0 -> GPU1...")
            t_start = time.time()
            t0_on_1 = t0.to('cuda:1')
            torch.cuda.synchronize()
            t_elapsed = time.time() - t_start
            bw = t0.numel() * 2 / t_elapsed / 1024**3
            log.log(f"    Transfer 0->1: {t_elapsed*1000:.1f}ms, {bw:.1f} GB/s")

            # Transfer GPU1 -> GPU0
            log.log(f"    Transferring GPU1 -> GPU0...")
            t_start = time.time()
            t1_on_0 = t1.to('cuda:0')
            torch.cuda.synchronize()
            t_elapsed = time.time() - t_start
            bw = t1.numel() * 2 / t_elapsed / 1024**3
            log.log(f"    Transfer 1->0: {t_elapsed*1000:.1f}ms, {bw:.1f} GB/s")

            # Bidirectional transfer (stress)
            log.log(f"    Bidirectional transfer...")
            t_start = time.time()
            t0_copy = t0.to('cuda:1')
            t1_copy = t1.to('cuda:0')
            torch.cuda.synchronize()
            t_elapsed = time.time() - t_start
            total_bytes = (t0.numel() + t1.numel()) * 2
            bw = total_bytes / t_elapsed / 1024**3
            log.log(f"    Bidirectional: {t_elapsed*1000:.1f}ms, {bw:.1f} GB/s total")

            # Cleanup
            del t0, t1, t0_on_1, t1_on_0, t0_copy, t1_copy
            torch.cuda.empty_cache()

            log.log(f"    Step {step+1} complete")

        log.log("Phase 15 PASSED")
    except Exception as e:
        log.log(f"Phase 15 FAILED: {e}")
        import traceback
        log.log(traceback.format_exc())
        log.close()
        sys.exit(1)

    # Cleanup
    log.section("TEST COMPLETE")
    log.log(f"All phases passed for batch_size={args.batch_size}")
    for info in get_gpu_memory_info():
        log.log(f"  Final: {info}")
    log.close()

    print(f"\n{'='*60}")
    print(f"SUCCESS: batch_size={args.batch_size} test completed")
    print(f"Log saved to: {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
