import argparse

from azureml.core import Workspace, Environment, ScriptRunConfig, ComputeTarget, Dataset, Experiment, Datastore
from azureml.core.compute import AmlCompute
from azureml.exceptions import ComputeTargetException

parser = argparse.ArgumentParser(description='PyTorch AD Example')

parser.add_argument('--exp_name',
                    type=str,
                    default='test',
                    metavar='EXPNAME',
                    help='experiment name')
args = parser.parse_args()


from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication(tenant_id="6c1d4152-39d0-44ca-88d9-b8d6ddca0708", force=True)

ws = Workspace.from_config(path='config_1.json', auth=interactive_auth)
cluster_name = 'gpu-cluster2'

datastore = ws.get_default_datastore()
# datastore = Datastore.get(ws, 'workspaceartifactstore')

data_target_path = './datasets/ad_trim/'
data_src_path = './data_trim/'
# data_target_path = './datasets/ad/'
# data_src_path = './data/'
data_path = datastore.upload(src_dir=data_src_path,
                             target_path=data_target_path,
                             overwrite=False)
print(data_path)


def prepare_environment():
    try:
        compute_target = ComputeTarget(workspace=ws, name=cluster_name, )
        print('Found existing compute target.')
    except ComputeTargetException:
        print('Creating a new compute target...')
        compute_config = AmlCompute.provisioning_configuration(
            vm_size='STANDARD_NC6',
            max_nodes=4,
            remote_login_port_public_access='Enabled',
            enable_node_public_ip=True,
            location='eastus'
        )
        # create the cluster
        compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
        compute_target.wait_for_completion(show_output=True)

    pytorch_env = Environment(name='pytorch',
                              AZUREML_COMPUTE_USE_COMMON_RUNTIME='false')
    pytorch_env = pytorch_env.get(workspace=ws,
                                  name='churn-prediction',
                                  version='3')
    # pytorch_env = pytorch_env.from_pip_requirements('ad', file_path='./requirements.txt')
    return pytorch_env


def commit_job(_operate_env, _experiment):
    dataset = Dataset.File.from_files(path=(datastore, data_target_path))
    src = ScriptRunConfig(source_directory='./src',
                          script='./run.py',
                          arguments=[
                              '--train_batch_size', '256',
                              '--index', '1',
                              '--eval_steps', '5000',
                              '--kfold', '5',
                              '--max_len_text', '128',
                              '--epoch', '5',
                              '--lr', '1e-4',
                              '--output_path', './output',
                              '--eval_batch_size', '512',
                              '--pretrained_model_path', dataset.as_named_input('input').as_mount(),
                              '--data_path', dataset.as_named_input('input').as_mount(),
                          ],
                          compute_target=cluster_name,
                          environment=_operate_env)
    run = _experiment.submit(config=src)
    # tb = Tensorboard(runs=[run], port=6000, local_root='./hw/cnn/run')
    # tb.start(start_browser=True)
    ml_flow()
    run.wait_for_completion(show_output=True)
    # run.download_file(name='checkpoints/ckpt.pth',
    #                   output_file_path='hw/cnn/outputs/DPN92net.pt')
    # tb.stop()1


def ml_flow():
    import mlflow
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    mlflow.create_experiment("mlflow-experiment")
    mlflow.set_experiment("mlflow-experiment")
    mlflow_run = mlflow.start_run()
    return mlflow_run


if __name__ == '__main__':
    experiment_name = args.exp_name
    _experiment = Experiment(workspace=ws, name='celebrities')
    operate_env = prepare_environment()
    commit_job(operate_env, _experiment)
