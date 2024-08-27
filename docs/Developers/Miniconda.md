# Miniconda

Miniconda is a toll which helps you manage multiple python environments.

Website: https://docs.anaconda.com/miniconda/

## Install Miniconda

Here is how you can install Miniconda on your machine.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# On Linux
~/miniconda3/bin/conda init bash

# on Mac
~/miniconda3/bin/conda init zsh
```

Close your terminal after executing these installation commands and reopen it. Now your terminal should show the name of the currently activated conda environment in brackets. The default environment's name is `base`.

## Create a new environment

Run the following command to create and activate a new conda environment. After you have activated the new conda environment all the packages that are installed via the pip package manager are only available in this specific environment. This is really helpful if you want to use multiple PyTorch versions on the same machine.

```bash
conda create -y -n my-env python==3.12 pip
conda activate my-env
```
