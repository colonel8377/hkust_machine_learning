FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04:20211012.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/pytorch-1.10.0

# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 \
    pip=21.2.4 \
    pytorch=1.10.0 \
    torchvision=0.11.1 \
    torchaudio=0.10.0 \
    cudatoolkit=11.3 \
    nvidia-apex=0.1.0 \
    -c anaconda -c pytorch -c conda-forge

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Install pip dependencies
RUN HOROVOD_WITH_PYTORCH=1 \
    pip install 'matplotlib>=3.3,<3.4' \
                'psutil>=5.8,<5.9' \
                'tqdm>=4.59,<4.60' \
                'pandas>=1.1,<1.2' \
                'scipy>=1.5,<1.6' \
                'numpy>=1.10,<1.20' \
                'ipykernel~=6.0' \
                'azureml-core==1.35.0' \
                'azureml-defaults==1.35.0' \
                'azureml-mlflow==1.35.0' \
                'azureml-telemetry==1.35.0' \
                'tensorboard==2.4.0' \
                'tensorflow-gpu==2.4.1' \
                'onnxruntime-gpu>=1.7,<1.8' \
                'horovod[pytorch]==0.21.3' \
                'future==0.17.1' \
                'torch-tb-profiler==0.2.1' \
                'concurrent_log_handler' \
                'opencv-python' \
                'pycox==0.2.2' \
                'tqdm==4.59' \
                'sklearn-pandas==2.0.3' \
                'joblib==1.1.0' \
                'scikit-learn==0.23.2' \
                'torchtuples==0.2.2'


RUN pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH
