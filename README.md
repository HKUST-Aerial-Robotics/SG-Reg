<!-- # SG-Reg -->
<div align="center">
    <h1>SG-Reg: Generalizable and Efficient</br> Scene Graph Registration</h2>
    <strong>Submitted to IEEE T-RO</strong>
    <br>
        <a href="https://uav.hkust.edu.hk/current-members/" target="_blank">Chuhao Liu</a><sup>1</sup>,
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
    <a href=""> <img src="https://img.shields.io/badge/UnderReview-T--RO-004c99"> </a>
    <!-- <a href='https://arxiv.org/abs/2402.04555'><img src='https://img.shields.io/badge/arXiv-2402.04555-990000' alt='arxiv'></a> -->
    <a href="https://youtu.be/Q7qa-6QgG5U"><img alt="YouTube" src="https://img.shields.io/badge/YouTube-Video-red"/></a>
</div>

<p align="center">
    <img src="docs/system.001.png" width="800"/>
</p>

### News
<!-- * [?Dec 2024] Paper accepted by IEEE T-RO. -->
* [Oct 2024] Paper submitted to IEEE T-RO.

We will publish the code once the paper is accepted.

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

## To develop
- [x] Scene graph network code and verify its inference.
- [x] Remove unncessary dependencies.
- [x] Clean the data structure.
- [ ] Provide RIO scene graph data for download.
- [ ] Provide network weight for download.
- [ ] Registration back-end in python interface.
- [ ] Validation the entire system in a new computer. 
- [ ] A tutorial for running the validation.