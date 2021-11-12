import argparse

from azureml.core import Workspace, Environment, ScriptRunConfig, ComputeTarget, Dataset, Experiment
from azureml.core.compute import AmlCompute
from azureml.exceptions import ComputeTargetException

parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')

parser.add_argument('--exp_name',
                    type=str,
                    default='test',
                    metavar='EXPNAME',
                    help='experiment name')
args = parser.parse_args()

from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication(tenant_id="6c1d4152-39d0-44ca-88d9-b8d6ddca0708", force=True)

ws = Workspace.from_config(path='config.json', auth=interactive_auth)
cluster_name = 'gpu-cluster'
datastore = ws.get_default_datastore()

data_target_path = 'datasets/'
data_src_path = 'data/'
data_path = datastore.upload(src_dir=data_src_path,
                             target_path=data_target_path,
                             overwrite=True)
print(data_path)
# fold_target_define = 'folds/dataset_split'
# fold_src_define = 'data/dataset_split/'
# fold_path = datastore.upload(src_dir=fold_src_define,
#                              target_path=fold_target_define,
#                              overwrite=True)
# print(fold_path)


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

    # use get_status() to get a detailed status for the current AmlCompute.
    # print(compute_target.get_status().serialize())
    pytorch_env = Environment(name='pytorch',
                              AZUREML_COMPUTE_USE_COMMON_RUNTIME='false')
    pytorch_env = pytorch_env.get(workspace=ws,
                                  name='churn-prediction',
                                  version='3')
    return pytorch_env


def commit_job(_operate_env, _experiment):
    dataset = Dataset.File.from_files(path=(datastore, data_target_path))
    # foldset = Dataset.File.from_files(path=(datastore, fold_target_define))
    src = ScriptRunConfig(source_directory='./src',
                          script='./D-Cox-Time/Cox_train.py',
                          arguments=[
                              '--weight_decay', '0.8',
                              '--lr', '0.01',
                              '--distance_level', '10',
                              '--max_session_length', '30',
                              '--earlystop_patience', '10',
                              '--optimizer', 'AdamWR',
                              '--cross_validation', '5',
                              '--one_hot', '1',
                              '--data_path', dataset.as_named_input('input').as_mount(),
                              '--fold_define', dataset.as_named_input('input').as_mount(),
                              '--device', 'cuda',
                              '--model_name', 'D-Cox-Time'
                          ],

                          compute_target=cluster_name,
                          environment=_operate_env)
    run = _experiment.submit(config=src)
    # tb = Tensorboard(runs=[run], port=6000, local_root='./hw/cnn/run')
    # tb.start(start_browser=True)
    # ml_flow()
    run.wait_for_completion(show_output=True)
    # run.download_file(name='checkpoints/ckpt.pth',
    #                   output_file_path='hw/cnn/outputs/DPN92net.pt')
    # tb.stop()


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
