# `mlstm_kernels` benchmarks on AMD MI300X

This document provides tuning instructions and usage guide for benchmarking `mlstm_kernels` on single **AMD MI300X** using **PyTorch's TunableOp**.

The script used to run and reproduce benchmark results is:

* `scripts/run_hf_model_benchmark_amd.py` - used for **xlstm model** MI300X benchmarking with tunable GEMMs.

## 🛠 Environment

These benchmarks were run using:

* **Hardware** : AMD MI300X
* **Software Stack** : ROCm 6.4
* **PyTorch** : with TunableOp support enabled

A Dockerfile and corresponding environment file are included in the repository to help reproduce the environment:

* `envs/hf_model_benchmark_amd.dockerfile`
* `envs/environment_pt240_rocm64.txt`

> 🐳 The Docker image used for benchmarking was built using the above files. While they provide a tested baseline, your results may still vary depending on your specific system configuration and ROCm/PyTorch version.

## ⚡️Tuning for `--ttft` benchmark

Benchmarking and optimization on MI300X demonstrated that, for the `--ttft` benchmark, `tokens per second` and `time to 100th token` can be improved **by up to 2×** (e.g., from 146 to 339 for  `tokens per second`) by leveraging  **PyTorch TunableOp**.

1. **Record untuned**

```
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_RECORD_UNTUNED=1 PYTORCH_TUNABLEOP_TUNING=0 python scripts/run_hf_model_benchmark_amd.py --folder_suffix "mlstm_bench_record_static" --benchmark ttft --use_torch_compile 1 
```

This generates `tunableop_untuned0.csv` with GEMMs to tune.

2. **Tune GEMMs**

```
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_RECORD_UNTUNED=0 PYTORCH_TUNABLEOP_VERBOSE=1 PYTORCH_TUNABLEOP_TUNING=1 python tuning.py
```

Contents of `tuning.py` file:

```
import torch.cuda.tunable as tunable 

tunable.tune_gemm_in_file("tunableop_untuned0.csv") # for tuning 
```

This generates `tunableop_results0.csv`.

3. **Run with Tuned Configs**

```
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_LOAD_RESULTS_FROM="tunableop_results0.csv" python scripts/run_hf_model_benchmark_amd.py --folder_suffix "mlstm_bench_tuned" --benchmark ttft --use_torch_compile 1 
```
