<!-- # SG-Reg -->
<div align="center">
    <h1>SG-Reg: Generalizable and Efficient</br> Scene Graph Registration</h2>
    <strong>Accepted by IEEE T-RO</strong>
    <br>
        <a href="https://glennliu.github.io" target="_blank">Chuhao Liu</a><sup>1</sup>,
        <a href="https://qiaozhijian.github.io/" target="_blank">Zhijian Qiao</a><sup>1</sup>,
        <a href="https://jayceeshi.github.io/" target="_blank">Jieqi Shi</a><sup>2,*</sup>,
        <a href="https://uav.hkust.edu.hk/group/alumni/" target="_blank">Ke Wang</a><sup>3</sup>,
        <a href="" target="https://uav.hkust.edu.hk/current-members/"> Peize Liu </a><sup>1</sup>
        and <a href="https://uav.hkust.edu.hk/group/" target="_blank">Shaojie Shen</a><sup>1</sup>
    <p>
        <h45>
            <sup>1</sup>HKUST Aerial Robotics Group &nbsp;&nbsp;
            <sup>2</sup> NanJing University &nbsp;&nbsp;
            <sup>3</sup>Chang'an University &nbsp;&nbsp;
            <br>
        </h5>
        <sup>*</sup>Corresponding Author
    </p>
    <a href=""> <img src="https://img.shields.io/badge/IEEE-T--RO-004c99"> </a>
    <a href='https://arxiv.org/abs/2504.14440'><img src='https://img.shields.io/badge/arXiv-2504.14440-990000' alt='arxiv'></a>
    <a href="https://youtu.be/IDxAmvpB2T0"><img alt="YouTube" src="https://img.shields.io/badge/YouTube-Video-red"/></a>
</div>

<p align="center">
    <img src="docs/system.001.png" width="800"/>
</p>

### News
* [$19^\text{th}$ Apr 2025] Our paper is accepted by IEEE T-RO as a regular paper.
* [$8^\text{th}$ Oct 2024] Paper submitted to IEEE T-RO.

In this work, we **learn to register two semantic scene graphs**, an essential capability when an autonomous agent needs to register its map against a remote agent, or against a prior map. To acehive a generalizable registration in the real-world experiment, we design a scene graph network to encode multiple modalities of semantic nodes: open-set semantic feature, local topology with spatial awareness, and shape feature. SG-Reg represents a dense indoor scene in **coarse node features** and **dense point features**. In multi-agent SLAM systems, this representation supports both coarse-to-fine localization and bandwidth-efficient communication. 
We generate semantic scene graph using vision foundation models and semantic mapping module [FM-Fusion](). It eliminates the need for ground-truth semantic annotations, enabling **fully self-supervised network training**. 
We evaluate our method using real-world RGB-D sequences: [ScanNet](), [3RScan]() and self-collected data using [Realsense i-435]().
## Install 
### Create environment

```bash
conda create sgreg python=3.9
```

### Install dependencies
Install PyTorch 2.1.2 and other dependencies.
```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia 
```
```bash
pip install -r requirements.txt
python setup.py build develop
```

## Download
#### Network weights
#### Dataset

## Inference 3RScan Scenes
To run an inference program, 
```bash
python sgreg/val.py --cfg_file config/rio.yaml --checkpoint $CHECKPOINT_FOLDER$
```
It will inference all of the downloaded scene pairs in 3RScan. The registration results, including matched nodes, point correspondences and predicted transformation are all saved. You can visualize the registration results,
```bash
python sgreg/visualize.py --dataroot $RIO_DATAROOT$ --viz_mode 1 --find_gt --viz_translation [3.0,5.0,0.0]
``` 
If you are inferencing on a remote server, it also support remote rerun visualization. Check the arguments intruction in [visualize.py](sgreg/visualize.py).

*[Optional]* If you want to evaluate SG-Reg on ScanNet sequences, adjust the running options as below,
```bash
python sgreg/val.py --cfg_file config/scannet.yaml --checkpoint $CHECKPOINT_FOLDER$ 
python sgreg/visualize.py --dataroot $SCANNET_DATAROOT$ --viz_mode 1 --augment_transform --viz_translation [3.0,5.0,0.0]
```

## Develop Log
- [x] Scene graph network code and verify its inference.
- [x] Remove unncessary dependencies.
- [x] Clean the data structure.
- [x] Visualize the results.
- [x] Provide RIO scene graph data for download.
- [x] Provide network weight for download.
- [ ] Registration back-end in python interface.
- [ ] Validation the entire system in a new computer. 
- [ ] A tutorial for running the validation.