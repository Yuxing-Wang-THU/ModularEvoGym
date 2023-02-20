# ModularEvoGym
ModularEvoGym is a modified benchmark that provides modular design and state-action spaces for designing and controlling 2D Voxel-based Soft Robots (VSRs).

This work is based on [EvolutionGym [1]](https://github.com/EvolutionGym/evogym)

![image](https://github.com/Yuxing-Wang-THU/ModularEvoGym/blob/main/images/thrower.gif)

## New features

<font color=Blue>Design (optional):</font>

1. A modular design space that can be incorporated into the Reinforcement Learning process.

2. A universal **Designer** (design policy network) based on Neural Cellular Automata (NCA).

<font color=Blue>Control:</font>

1. A modular state-action space for VSRs.

2. A universal Transformer-based **Controller** (control policy network) which can be trained by Proximal Policy Optimization (PPO).

## Observation Space

![image](images/origin_obs.jpg)

**Observation Space of ModularEvoGym**
![image](images/modular_obs.jpg)

The input state of the robot at time step $t$ is represented as $s_{t}^{c}=\lbrace s_{t}^{v},s_{t}^{g}\rbrace$, where $s_{t}^{v}=\lbrace s_{t}^{v_{1}}, s_{t}^{v_{2}},...,s_{t}^{v_N}\rbrace$, $s_{t}^{v_i}$ is composed of each voxel's local information which contains the relative position of its four corners with respect to the center of mass of the robot and its material information (e.g., <b><font color=Gray>soft voxel</font></b>, <b>rigid voxel</b>, <b><font color=Darkorange>horizontal actuator</font></b> and <b><font color=DeepSkyBlue>vertical actuator</font></b>). $s_{t}^{g}$ is the task-related observation such as terrain information of the environment and goal-relevant information. During the simulation, voxels (except empty voxels) only sense locally, and based on the input sensory information, a controller outputs control signals to vary the volume of actuator voxels. The morphology of the robot is unchangeable during the interaction with the environment.

## Installation
### 1. Clone

```shell
git clone --recurse-submodules https://github.com/Yuxing-Wang-THU/ModularEvoGym.git
```
make sure that submodules (glfw, glew and pybind11) are successfully downloaded to "/evogym/simulator/externals"

### 2. Install Evogym

Requirements：
* Python 3.7/3.8
* Linux (Ubuntu)
* [OpenGL](https://www.opengl.org//)
* [CMake](https://cmake.org/download/)
* [PyTorch](http://pytorch.org/)

<!--- (See [installation instructions](#opengl-installation-on-unix-based-systems) on Unix based systems) --->

```shell
sudo apt-get install xorg-dev libglu1-mesa-dev
```

Install Python dependencies:

```shell
conda create -n modularevogym python==3.7.11

conda activate modularevogym

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

pip install Gpy==1.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/

pip install git+https://github.com/yunshengtian/GPyOpt.git

pip install git+https://github.com/yunshengtian/neat-python.git

```
To build the C++ simulation, build all the submodules, and install `evogym` run the following command:

```shell
python setup.py install
``` 
if you meet this error "Could NOT find GLEW (missing: GLEW_INCLUDE_DIRS GLEW_LIBRARIES)", run

```shell
sudo apt install libglew-dev
``` 

### 3. Test Installation

cd to the `examples` folder and run the following script:

```shell
python modularevogym_test.py
```
modularevogym_test.py
```python

import gym
import evogym.envs
from evogym import sample_robot
from modular_envs.wrappers.modular_wrapper import modular_env
import numpy as np

if __name__ == '__main__':
    # Setting
    mode = "modular"
    body_size = (3,3)
    body, connections = sample_robot(body_size)

    # Env
    env = gym.make("Walker-v0", body=body)
    # If you want to use ModularEvoGym
    env = modular_env(env=env, body=body)
    obs = env.reset()

    # Rollout
    while True:
        if mode == 'modular':
            action = np.random.uniform(low=0.6, high=1.6, size=body_size[0]*body_size[1])-1
        else:
            action = env.action_space.sample()-1
        ob, reward, done, info = env.step(action)
        # env.render()
        if done:
            break
    env.close()
```
## Controlling Modular Soft Robots via Transformer
We provide a universal Transformer-based controller [2-3] which can handle the incompatible state-action spaces. This controller can be trained by many popular Reinforcement Learning methods (e.g., SAC, PPO, DDPG).

### **Controller**

![image](images/tf-controller.png)

Self-attention brings better interpretability than multilayer perceptron. We use only one transformer encoder layer, thus, we visualize the attention matrix after the input state passes through the attention layer. The following figure shows attention matrices of 2 control steps produced by the policy network. 

![image](images/attention.png)

The color of each attention score tells the strength of the compatibility between inputs and interprets what is driving the behaviour of the VSR. When the robot’s front foot (voxel 9) or the rear foot (voxel 7) touches the ground, the corresponding voxels are assigned with greater wights, which is consistent with humans’ intuition and common sense.

### **RL Training**

#### 1. Controlling a single VSR 
To train a predefined robot to walk, cd to the `examples` folder and run the following script:

```shell
python run_transformer_ppo.py
```

Logs are stored in "examples/saved_data/Walker-v0" and a trained model is stored in "examples/visual".

To visualize the training process:

```shell
python simple_plotter.py
```
<img src="https://github.com/Yuxing-Wang-THU/ModularEvoGym/blob/main/images/Walker-v0_training_curves.png" div align=middle width = "37%" />

To make a gif:

```shell
python simple_gif.py
```

![image](https://github.com/Yuxing-Wang-THU/ModularEvoGym/blob/main/images/walker.gif)


#### 2. Controlling multiple VSRs using one controller

To train some randomly sampled robots to walk, cd to the `examples` folder and run the following script:

```shell
python run_transformer_ppo_multi.py
```
Morphologies

![image](images/multi.png)

Learning curves

<img src="https://github.com/Yuxing-Wang-THU/ModularEvoGym/blob/main/images/multi_robots.png" div align=middle width = "57%" />

## Citation

```shell

@inproceedings{
wang2023curriculumbased,
title={Curriculum-based Co-design of Morphology and Control of Voxel-based Soft Robots},
author={Yuxing Wang and Shuang Wu and Haobo Fu and QIANG FU and Tiantian Zhang and Yongzhe Chang and Xueqian Wang},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=r9fX833CsuN}
}

```

## References

[1] Jagdeep Bhatia, Holly Jackson, Yunsheng Tian, Jie Xu, and Wojciech Matusik. Evolution gym: A large-scale benchmark for evolving soft robots. In NeurIPS, 2021.

[2] Agrim Gupta, Linxi (Jim) Fan, Surya Ganguli, and Li Fei-Fei. Metamorph: Learning universal controllers with transformers. ArXiv, abs/2203.11931, 2022.

[3] Vitaly Kurin, Maximilian Igl, Tim Rocktaschel, Wendelin Boehmer, and Shimon Whiteson. My body is a cage: the role of morphology in graph-based incompatible control. ArXiv, abs/2010.01856, 2021. aaaa
