# Spack Package Manager

Spack is a package manager that allows you to install multiple versions of CUDA and other libraries on your cluster.

see: https://csc.uni-frankfurt.de/wiki/doku.php?id=public:usage:spack

## Install Spack Manager

1. Clone GitHub Repository: `git clone https://github.com/spack/spack`
2. Navigate into spack directory: `cd spack`
3. Make Spack command available in the terminal: `. spack/share/spack/setup-env.sh`

Now the `spack` command should be available in the terminal.

## Install CUDA modules

1. Show all available CUDA versions: `spack info cuda`
2. Install a specific CUDA version: `spack install cuda@11.8.0`
3. Run `spack module tcl refresh` to update the available modules
4. Run `module avail` and you should see an output similar to this one:

```
------------ /scratch/dldevel/<your-username>/spack/share/spack/modules/linux-almalinux9-skylake_avx512 -------------
cuda/11.8.0-gcc-11.3.1-szmxwyv
```

Now you can load CUDA 11.8.0 in your scripts using the following line:

```
module load cuda/11.8.0-gcc-11.3.1-szmxwyv
```

There cannot be loaded two CUDA modules at once. If you are unable to load the CUDA 11.8.0 module due to CUDA 12.3.0 already being loaded you may execute the following command first:

```
module unload cuda/11.8.0-gcc-11.3.1-szmxwyv
```

## Use different CUDA versions in your scripts

Create two files:

`nvcc_12.sh`

```bash
#!/bin/bash

#SBATCH --job-name=nvcc-check
#SBATCH --output=output_sbatch/%j.out
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --account=dldevel
#SBATCH --mem=1G
#SBATCH --time=00:01:00

module load nvidia/cuda/12.3.0
nvcc --version
echo "Finished execution"
```

`nvcc.11.sh`

```bash
#!/bin/bash

#SBATCH --job-name=nvcc-check
#SBATCH --output=output_sbatch/%j.out
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --account=dldevel
#SBATCH --mem=1G
#SBATCH --time=00:01:00

module load cuda/11.8.0-gcc-11.3.1-szmxwyv
nvcc --version
echo "Finished execution"
```

Run both scripts via `sbatch nvcc_11.sh` / `sbatch_nvcc_12.sh`

The scripts should produce the following output:

`nvcc_12.sh`:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Sep__8_19:17:24_PDT_2023
Cuda compilation tools, release 12.3, V12.3.52
Build cuda_12.3.r12.3/compiler.33281558_0
Finished execution
```

`nvcc_11.sh`

```
Loading cuda/11.8.0-gcc-11.3.1-szmxwyv
  Loading requirement: gcc-runtime/11.3.1-gcc-11.3.1-5wigvvv
    libiconv/1.17-gcc-11.3.1-szpwjfa xz/5.4.6-gcc-11.3.1-atuphiv
    zlib-ng/2.1.5-gcc-11.3.1-onwqkc5 libxml2/2.10.3-gcc-11.3.1-umgxq4p
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
Finished execution
```

As you can see both scripts have loaded diffferent CUDA versions.
