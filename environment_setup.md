# Environment Setup Guide

## Copy the data to remote host


## Install Tensorflow with conda
1. Create conda virtual environment  
   reference: https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands

   ```
   conda create --name deeplearning
   conda activate deeplearning
   ```

2. Install ternsorflow cpu-v1.10  
   reference:https://anaconda.org/conda-forge/tensorflow
   ```
   conda install -c conda-forge tensorflow
   ```
3. Setup Jupyter notebook  
   See https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments   
   https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook
   ```
   // in base
   conda install nb_conda

   conda activate deeplearning
   conda install ipykernel
   python -m ipykernel install --user --name deeplearning --display-name "Python (DeepLearning)"
   ```

## conda packages
```
conda install 
matplotlib
matplotlib pillow
tqdm

```
