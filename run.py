import argparse

from azureml.core import Workspace, Environment, ScriptRunConfig, ComputeTarget, Dataset, Experiment
from azureml.core.compute import AmlCompute
from azureml.exceptions import ComputeTargetException
from azureml.tensorboard import Tensorboard

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
    pytorch_env = Environment(name='pytorch', AZUREML_COMPUTE_USE_COMMON_RUNTIME='false')
    pytorch_env = pytorch_env.get(workspace=ws, name='Azure-ml-py38-pyorch1.10-cuda11.3-ust', version='2')
    return pytorch_env


def commit_job(_operate_env, _experiment):
    dataset = Dataset.File.from_files(path=(datastore, data_target_path))
    src = ScriptRunConfig(source_directory='./hw/cnn',
                          script='start.py',
                          arguments=[
                              '--end_epochs', 250, '--data_dir',
                              dataset.as_named_input('input').as_mount()
                          ],
                          compute_target=cluster_name,
                          environment=_operate_env
                          )
    run = _experiment.submit(config=src)
    tb = Tensorboard(runs=[run], port=6000, local_root='./hw/cnn/run')
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
    _experiment = Experiment(workspace=ws, name=experiment_name)
    operate_env = prepare_environment()
    commit_job(operate_env, _experiment)
