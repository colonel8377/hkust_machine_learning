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
- [参考来源](https://github.com/kuangliu/pytorch-cifar)
- [数据来源](http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

| Model             | Acc.        |
| ---------------------- | ------------------------ |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |
| [CNN](http://cogprints.org/5869/1/cnn_tutorial.pdf)   | 96.14%      |
  

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
### Credit
- [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
