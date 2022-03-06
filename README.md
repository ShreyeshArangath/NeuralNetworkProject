# Neural Network Project

## How to set up locally? 
```
git clone git@github.com:ShreyeshArangath/NeuralNetworkProject.git
cd NeuralNetworkProject
```
Download the dataset and move it to the `./data`

Datasets can be found at https://sviro.kl.dfki.de/download/

## How to set up on HPCC?

#### Getting Datasets:
```
sh getdatasets.sh
```
#### Set up environment
```
/lustre/work/examples/InstallPython.sh
. $HOME/conda/etc/profile.d/conda.sh
conda activate
```

**You must set up the environment first before installing required packages**

#### Run program
```
sh runProgram.sh
```
#### Run program as a job
```
sbatch <name_of_file>.sh
```
<sub>If anyone needs help with creating headers for jobs, please let Rayven know. It has changed a bit.</sub>

## Requirements 
1. Python 3.x 
2. Tensorflow
3. Keras
4. OpenCV-python

```
pip install -r requirements.txt
```

<sub>*OpenCV is going to have to be it's own pip install command*
