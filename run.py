import argparse

from azureml.core import Workspace, Environment, ScriptRunConfig, ComputeTarget, Dataset, Experiment
from azureml.tensorboard import Tensorboard
from azureml.core.compute import AmlCompute
from azureml.exceptions import ComputeTargetException

parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')

parser.add_argument('--exp_name',
                    type=str,
                    default='test',
                    metavar='EXPNAME',
                    help='experiment name')
args = parser.parse_args()

ws = Workspace.from_config(path='./config.json')
cluster_name = 'gpu-cluster'
datastore = ws.get_default_datastore()
data_target_path = 'datasets/cifar10'
data_src_path = 'hw/data/cifar10'
datastore.upload(src_dir=data_src_path,
                 target_path=data_target_path,
                 overwrite=True)


def prepare_environment():
    try:
        compute_target = ComputeTarget(workspace=ws, name=cluster_name)
        print('Found existing compute target.')
    except ComputeTargetException:
        print('Creating a new compute target...')
        compute_config = AmlCompute.provisioning_configuration(
            vm_size='STANDARD_NC6',
            max_nodes=4,
            remote_login_port_public_access='Enabled',
            enable_node_public_ip=True)
        # create the cluster
        compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
        compute_target.wait_for_completion(show_output=True)

    # use get_status() to get a detailed status for the current AmlCompute.
    print(compute_target.get_status().serialize())
    pytorch_env = Environment.get(workspace=ws, name='AzureML-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu')\
        .from_pip_requirements(name='pytorch-gpu', file_path='azure-requirements.txt')
    return pytorch_env


def commit_job(_operate_env):
    dataset = Dataset.File.from_files(path=(datastore, data_target_path))
    src = ScriptRunConfig(source_directory='./hw/cnn',
                          script='start.py',
                          arguments=[
                              '--end_epochs', 200, '--data_dir',
                              dataset.as_named_input('input').as_mount()
                          ],
                          compute_target=cluster_name,
                          environment=_operate_env)
    run = experiment.submit(config=src)
    tb = Tensorboard(runs=[run], local_root='./hw/cnn/run')
    tb.start(start_browser=True)
    ml_flow()
    run.wait_for_completion(show_output=True)
    run.download_file(name='checkpoints/ckpt.pth',
                      output_file_path='hw/cnn/outputs/DPN92net.pt')
    tb.stop()


def ml_flow():
    import mlflow
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    mlflow.create_experiment("mlflow-experiment")
    mlflow.set_experiment("mlflow-experiment")
    mlflow_run = mlflow.start_run()
    return mlflow_run


if __name__ == '__main__':
    experiment_name = args.exp_name
    experiment = Experiment(workspace=ws, name=experiment_name)
    operate_env = prepare_environment()
    commit_job(operate_env)
