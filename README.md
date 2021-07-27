# Super Mario Reference Experiments for Communicating Agents
## Communicating agents

The idea is to use the super Mario game to generate data for a communicating agent problem as described in this paper: H. Poulsen Nautrup, T. Metger, R. Iten, S. Jerbi, L.M. Trenkwalder, H.Wilming, H.J. Briegel, and R. Renner. "Operationally meaningful representations of physical systems in neural networks" (2020).

The code for the Mario game is based on https://github.com/marblexu/PythonSuperMario.git and modified for our purposes. --> Thanks a lot marblexu!


## Some words on the setup
The following sections describe the concepts hidden states, reference
experiments and questions, which are described in the paper in detail.

### Hidden states:
  1. Position of coin ($x_{coin}$) 
  1. Speed of first enemy ($v_{enemy}$)
  2. Position of first pipe ($x_{pipe}$)
  3. Marios speed

<img src="resources/figures/mario1.png" alt="drawing" width="200"/>
<img src="resources/figures/mario2.png" alt="drawing" width="200"/>

### Reference experiment
  - Start game with randomly selected values for hidden states  
  - Let Mario run (walk) at normal and constant speed (speed is a random
    variable in this case)
  - Observations:
    - Some number (in our case 10) of pictures of the game taken at equal $\Delta t$s

### Questions
  - When does Mario need to jump in order to Kill the enemy?
  - When does Mario need to jump in order get the coin from the first question mark?
  - When does Mario need to jump in order overcome the pipe?
  - What is the distance between the coin box and the pipe?

### Neural net implementation
#### Encoding agents
In this scenario we need one decoding agents ($E$).
This Agent ($E$) receives a fixed number of images (a sequence of image/ video) of the game with an unknown experimental (values of the hidden states) setting as input.
The output dimension of $E$ equals the number of hidden states (in this case 3)
The encoder network architecture consists of a Resnet50 which encodes each
individual image of the sequence into a fixed size vector.  
Now we get a time series of these encoding vectors. These are in turn processed
in a RNN. As a last step, the hidden state of the RNN is processed in a fully
connected NN in order to generate the latent space variables.

#### Filter
The filter function passes the activation of E's output nodes on to three decoding agents $D_1$ to $D_4$.
However for each output node the filter function adds a bias term times some Gaussian error. 
This helps in disentangling the underlying factors of variation. Please refer to
the paper for a more detailed explanation.

#### Decoding agents
Simply fully connected neural networks.

## Project orga
The organization of this repo is inspired by the data science coocky cutter
template which can be found [here](https://github.com/drivendata/cookiecutter-data-science).

```
.
├── mario_com_agent.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── mario_game
│   ├── ...
├── notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_out_exploration.ipynb
│   └── 03_selection_bias_evolutoin.ipynb
├── README.md
├── README.md.backup
├── resources
│   ├── 2021-05-07-Note-17-24.xoj
│   └── figures
│       ├── mario1.png
│       ├── mario2.png
│       ├── Presentation1.pptx
│       └── Presentation2.pptx
├── scripts
│   ├── experiment.py
│   ├── __init__.py
│   ├── model_training.py
│   └── question_and_optimal_answer.py
├── setup.py
├── src
│   ├── constants.py
│   ├── __init__.py
│   ├── model
│       ├── agents.py
│       ├── __init__.py
│       ├── lit_module.py
└── utils
    ├── create_experiment_data_debug.sh
    ├── create_experiment_data.sh
    ├── run_training_gpus.sh
    └── run_training_locally.sh
```

## Run the experiment
Install stuff
```shell
git clone <this repo>
cd mario-communicating-agents
conda create -n mario python=3.9
conda activate mario
pip install -e .
```
If you want to do the model training using GPUs (which I highly reccomend), do
this, or check [here](https://pytorch.org/get-started/locally/) for a more up to date version:
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### Data creation
To create the data set (ovservations) we run the mario game with modified
parameter for 10k times. Note two things: first, you will need a machine with a
GUI installed, since this includes taking screenshots. Second, It will probably
occupy that machine for a couple of hours. I you don't want to do this feel free
to reach out to mee, so that I can send you the dataset.

Start the experiments:
```shell
./utils/create_experiment_data.sh
```
Now that we have the obervation data set and a labels file, we still need to
compute the questions and answers:
```shell
python ./scripts/question_and_optimal_answer.py
```

### Model training
```shell
./utils/run_training_gpus.sh
```
or
```shell
./utils/run_training_locally.sh
```










