# NeuralSymbolicRegressionThatScales
Pytorch implementation and pretrained models for the paper "Neural Symbolic Regression That Scales" 
For details, see **Emerging Properties in Self-Supervised Vision Transformers**.  
[[`arXiv`](Missing Link] 


## Pretrained models
We offers two models "10M" and "100M". The first, trained on a dataset of 10M and with constant prediction set to false, is the one used for experiements in our paper. The second is trained on 100M of equations and with constant prediction set to true.
For both models, the equations included in data/tests.csv are held out during training

<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>k-nn</th>
    <th>linear</th>
    <th colspan="6">download</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>21M</td>
    <td>74.5%</td>
    <td>77.0%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deits16.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-S/8</td>
    <td>21M</td>
    <td>78.3%</td>
    <td>79.7%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deits8.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>85M</td>
    <td>76.1%</td>
    <td>78.2%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitb16.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/8</td>
    <td>85M</td>
    <td>77.4%</td>
    <td>80.1%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitb8.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>23M</td>
    <td>67.5%</td>
    <td>75.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
</table>

The pretrained models are available on PyTorch Hub.
```python
import torch
vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
vitb8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
```

# How to use it
First you need to generate a dataset. Using the provided makefile is the easiest way to create it.
Define a variable in the console and export it to subprocesses by the command export NUM=${NumberOfEquationsYouWant}. 
NumberOfEquationsYouWant can be defined in two formats with K or M suffix. For instance

## Dataset Generation
Using the makefile is fastest way to create a datest. First, export the number of equations you would like to have with the command "export NUM=100000". Second, call make 