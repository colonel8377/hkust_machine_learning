## Training Steps

### Prerequisite

- [azure client(本机与azure资源交互)](https://docs.microsoft.com/zh-cn/azure/machine-learning/how-to-configure-cli)
- [azure resource group](https://docs.microsoft.com/zh-cn/azure/machine-learning/how-to-train-cli)
- [pytorch (注意下载的版本)](https://pytorch.org/)
- [miniconda (recommend)](https://docs.conda.io/en/latest/miniconda.html)
- [cuda(gpu)](https://developer.nvidia.com/zh-cn/cuda-toolkit)

### Initialization

- 确保弹性计算资源可用
- 创造弹性计算资源(cpu/gpu cluster)

```commandline
az ml compute create -n cpu-cluster --type amlcompute --min-instances 0 --max-instances 10 
az ml compute create -n gpu-cluster --type amlcompute --min-instances 0 --max-instances 4 --size Standard_NC12
```

- az ml job create -f ./cifar.yml --web
  </br>
  此命令将在远端启动一个拥有gpu资源的docker容器服务，需要注意的是学习任务输出路径固定为outputs和logs，logs为实时输出的日志流，参考start.py文件.此外，注意合理设置日志打印/设置concurrent打印服务，不然容易造成服务IO过大。
- API查看任务情况
- [详情参考](https://docs.microsoft.com/zh-cn/azure/machine-learning/how-to-train-cli#prerequisites)

### Model

- [模型来源](https://github.com/kuangliu/pytorch-cifar)

### Structure

```
cnn
├─ START.md
├─ cifar.yml(类docker-compose.yml文件)
├─ data
│    ├─ cifar-10-batches-py
│    └─ cifar-10-python.tar.gz
├─ logs
│    └─ cifar.log
├─ models
│    ├─ __init__.py
│    ├─ cnn.py
│    ├─ densenet.py
│    ├─ dla.py
│    ├─ dla_simple.py
│    ├─ dpn.py
│    ├─ efficientnet.py
│    ├─ googlenet.py
│    ├─ lenet.py
│    ├─ mobilenet.py
│    ├─ mobilenetv2.py
│    ├─ pnasnet.py
│    ├─ preact_resnet.py
│    ├─ regnet.py
│    ├─ resnet.py
│    ├─ resnext.py
│    ├─ senet.py
│    ├─ shufflenet.py
│    ├─ shufflenetv2.py
│    └─ vgg.py
├─ start.py(主启动文件)
├─ train.py(训练文件)
└─ utils.py(训练工具包)
```

### Train

- 调整训练超参数(依据start.py中parser进行调整,需要在yml中命令行格式呈现)
- [自动调整超参数](https://docs.microsoft.com/zh-cn/azure/machine-learning/how-to-train-cli#prerequisites)
- 绘制loss函数/下降图(mathplot)
- 模型重现(./output/ckpt.pth)
```python
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
```
- 开发需求(自动解析loss函数, 自动绘制散点图)