# About
This project is an implementation of a basic multilayer perceptron (MLP) neural network built from scratch, using only the C standard library with no external dependencies.
This was done to gain a greater understanding of the core mathematics and concepts underlying neural networks by implementing everything manually, from matrix operations to backpropagation and gradient descent.

### Key features:
- Custom matrix operations library.
- Full training loop implementation, including forward propagation, loss calculation, backpropagation, and parameter updates via gradient descent.
- Support for various activation and loss functions.
- Support for learning rate scheduling and weight initialisation techniques.
- Configurable network architectures and training hyperparameters via external JSON files.
- Simple command-line interface for training and evaluating models from the terminal.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Limitations](#limitations)

## Installation
### Prerequisites
Before installing, ensure the following are installed and set up:
- [GNU Make](https://www.gnu.org/software/make/), for building the project using the provided Makefile
- [GCC (GNU Compiler Collection)](https://gcc.gnu.org/), for compiling the source code
- A Unix-like terminal, which is needed for both building and running the project:
    * On Linux and macOS, you can use the default terminal
    * On Windows, you'll have to use a terminal from a Unix-like environment such as [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) or [MSYS2](https://www.msys2.org/#installation)

### Clone the repository
```
git clone https://github.com/AlexLomas1/neural-network
```

### Build the project
1. Navigate to the cloned repository:
```
cd neural-network
```

2. Compile the project:
```
make
```

This will do two things:
1. Create a `build/` folder containing object (`.o`) files generated from the `.c` files in the `src/` folder, and
2. Create a `main` executable, which is the entry point of the project.

## Usage
Once you have the project installed, and have navigated to the repository, you can run it using:
```
./main
```

You will then be prompted to enter a name for a dataset. Three datasets are provided with the project by default: `iot_intrusion`, `iris`, and `xor`, with each name matching the name of a subfolder within the `data/` folder. Information about these datasets can be found in [Datasets](#datasets).

From here, the rest is handled automatically:
1. The network architecture and training hyperparameters are loaded from the `net_config.json` and the `train_config.json` files respectively within the relevant `data/` subfolder.
2. The training dataset is loaded from `train.csv`.
3. The neural network is trained on the training dataset using backpropagation and gradient descent. Loss is reported at regular intervals, including before and after training.
4. After training, the network is evaluated on the testing dataset, loaded from `test.csv`. The loss (and accuracy, for classification problems) on this dataset is reported.

An example output may look like this:

<img width="500" alt="Example output of training and evaluation on Iris dataset" src="https://github.com/user-attachments/assets/05e1c4c1-77ca-449b-b6ef-f37b7408241d" />

> Results may vary across runs due to randomness in weight initialisation. This is particularly noticeable with the XOR problem, where the small network size makes it especially sensitive to starting weights.
As such, the neural net can sometimes get stuck at only 50% accuracy on this problem.

## Datasets
There are three datasets which are included in this project by default: 
- [IoT Intrusion Detection and Classification](#iot-intrusion-detection-and-classification)
- [Iris Dataset](#iris-dataset)
- [XOR Problem](#xor-problem)

Each dataset has a corresponding subfolder within the `data/` folder. These subfolders contain:

- `train.csv` and `test.csv` - contains the training and testing datasets respectively
- `net_config.json` - defines the network architecture (number of layers, nodes in each layer, activation functions, and weight initialisation methods)
- `train_config.json` - defines the training hyperparameters (loss function, number of epochs, the base learning rate, and the learning rate schedule)

> Both configuration files are fully editable, allowing experimentation with different network architectures and training parameters.

### IoT Intrusion Detection and Classification
**Problem type**: Multi-class classification (5 classes)

**Training samples**: 7636 (80%)

**Testing samples**: 1909 (20%)

**Input features:** 10

**Output features:** 5 (one-hot encoded)

This dataset consists of samples of IoT network traffic data from a simulated network environment, which is used to classify traffic into one of five classes: Benign, Reconnaissance, DDoS, DoS, or Theft.

It is based on the NF IoT-BoT V1 dataset, one of the [NetFlow V1 datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) introduced by Sarhan, M. et al. (2021). More details are available in the publication: "[Netflow Datasets for Machine Learning-based Network Intrusion Detection Systems](https://doi.org/10.1007/978-3-030-72802-1_9)".

Several changes were made to the original dataset to make it suitable for its use here:

- Balanced the dataset by reducing the number of samples of each class to that of the least represented class. This also had the effect of reducing the dataset from 600,100 samples to only 9545, vastly reducing training time.
- Standardised input features by subtracting the mean and dividing by the standard deviation, both calculated from the training dataset.
- Two input features, the IPv4 address of the sender and of the receiver, were removed as they provided negligible, and perhaps even slightly detrimental, effects on accuracy and increased training time. While there are other ways they could have been encoded to make them useful, for the purpose of simplicity they were just removed entirely.
- The `Label` column, indicating whether a sample was benign or an attack, was removed as this can simply be inferred by the next column, `Attack`, which gives the class.
- The Attack column, indicating each sample's class, was converted from text labels to one-hot vector format.

### Iris Dataset
**Problem type:** Multi-class classification (3 classes)

**Training samples:** 114 (76%)

**Testing samples:** 36 (24%)

**Input features:** 4

**Output features:** 3 (one-hot encoded)

This dataset features measurements taken from different flowers, which is used to classify each sample into one of three species of Iris: Iris setosa, Iris virginica, and Iris versicolor.

The data was taken from the classic [Iris dataset](https://doi.org/10.24432/C56C76), from R. A. Fisher (1936), with the data originally collected by biologist Edgar Anderson.
The dataset was altered slightly to use a one-hot vector format for the output instead of text labels.

### XOR Problem
**Problem type:** Binary classification

**Training samples:** 4

**Testing samples:** 4

**Input features:** 2

**Output features:** 1

This dataset represents the truth table of a two-input XOR logic gate:

| Input 1 | Input 2 | Output |
|:-------:|:-------:|:------:|
|    0    |    0    |   0    |
|    0    |    1    |   1    |
|    1    |    0    |   1    |
|    1    |    1    |   0    |

This is a simple but classic machine learning problem, as it cannot be solved using a single-layer perceptron. This makes it a straightforward and effective way for verifying that the neural network works as intended.

As the XOR gate requires its full truth table to be defined, the training and testing datasets are identical for this problem.

### Adding New Datasets
Alongside the three datasets included by default, others can be added as well.

To do so:
1. Create a subfolder within the `data/` folder. For example, for a dataset named `my_dataset`, create folder `data/my_dataset/`.
2. Add dataset files - `train.csv` and `test.csv`. These should follow a standard format:
    * The first line is reserved for a comment indicating the number of inputs and output features, e.g. `# INPUTS: 2, OUTPUTS: 1`. 
    * The second line is reserved for column headers.
    * Remaining lines contain the data values, with all inputs features listed first, followed by output feature(s).
3. Create `net_config.json` and `train_config.json` - these define the network architecture and training hyperparameters respectively. You can copy them from existing datasets and modify as needed.
  
Once these changes are made and saved, you can now train the neural network on this dataset just like any of the default ones. Simply run the project:
```
./main
```
And when prompted, enter the new dataset name, e.g. `my_dataset`. You will not need to recompile the project.

## Limitations
Since this project was created primarily as a personal learning exercise, it has many limitations compared to widely used machine learning libraries. Some such limitations are listed below:

- Only multilayer perceptron (MLP) architectures are supported.
- The only optimisation method currently implemented is standard gradient descent.
- Saving and loading models is not currently included.
- Only full-batch training is currently supported.
- No regularisation methods have been included.
- This project does not currently utilise parallelism or GPU acceleration.
- Data preprocessing has not been integrated into the project.
