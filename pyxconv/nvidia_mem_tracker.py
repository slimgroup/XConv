import os

# Uncommenting this confuses pytorch's internal memory tracker
# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"


import torch

try:
    import pynvml as nvidia_smi

    nvidia_smi.nvmlInit()
except Exception:  # pynvml missing, or no NVIDIA driver present
    nvidia_smi = None


class MemoryTracker:
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        if nvidia_smi is None:
            raise RuntimeError(
                "pynvml required. Install with: pip install pynvml"
            )
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)

    def _get_gpu_mem_mb(self):
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
        return mem.used // 1024**2  # Convert bytes to MB

    def __enter__(self):
        torch.cuda.synchronize()
        self.torch_start = torch.cuda.memory_allocated()
        self.torch_reserved_start = torch.cuda.memory_reserved()
        self.nvidia_start = self._get_gpu_mem_mb()
        torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.torch_end = torch.cuda.memory_allocated()
        self.torch_reserved_end = torch.cuda.memory_reserved()
        self.torch_peak = torch.cuda.max_memory_allocated()
        self.nvidia_end = self._get_gpu_mem_mb()

    @property
    def torch_allocated_mb(self):
        return (self.torch_end - self.torch_start) / 1024**2

    @property
    def torch_reserved_mb(self):
        return (self.torch_reserved_end - self.torch_reserved_start) / 1024**2

    @property
    def torch_peak_mb(self):
        return (self.torch_peak - self.torch_start) / 1024**2

    @property
    def nvidia_used_mb(self):
        return self.nvidia_end - self.nvidia_start

    def print_summary(self):
        print(
            f"PyTorch allocated: {self.torch_allocated_mb:.1f} MB used, {self.torch_peak_mb:.1f} MB peak"
        )
        print(f"PyTorch reserved:  {self.torch_reserved_mb:.1f} MB")
        print(f"Nvidia delta:      {self.nvidia_used_mb:+} MB")
        print(f"Nvidia absolute:   {self.nvidia_start} → {self.nvidia_end} MB")


if __name__ == "__main__":
    with MemoryTracker() as t:
        x = torch.randn(2000, 2000, device="cuda")

    t.print_summary()
