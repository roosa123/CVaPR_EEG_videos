# Emotion Recognition in Videos From EEG Signals Using Deep Neural Networks

## A project for Computer Vision and Pattern Recognition classes at Silesian University of Technology

Lecturer: Krzysztof Kotowski  
Semester: 1st semester of the MSc studies programme

Let's determine some universal rules and project requirements (listed below):

1. Placing the dataset in the repo directory structure:  
    * let's place the dataset in the **DEAP/** directory (already only with .gitkeep file)
2. IDE
    * the suggested IDE is **PyCharm**, but let's make the project able to run from any other - just make sure you have installed proper Python packages
3. Language & libraries
    * the language of the project is **Python 3** (**important:** if you want to use TensorFlow as a backend for Keras, Python version should not be higher than **3.6**!)
    * required packages (please extend this list as you will introduce more packages):
        * Keras
        * TensorFlow
        * NumPy
        * Matplotlib
        * Pandas
        * MNE
        * SciPy
    * if you wish to plot the model of the network, you should install [Graphviz](https://www.graphviz.org) and pydot package
4. GPU usage
    * there is a possibilty to use TensorFlow with computation acceleration on GPU. To utilise this feature, please install GPU-accelerated version of TensorFlow (TensorFlow-GPU) and additional necessary tools (CUDA Toolkit and CuDNN), listed in [TensorFlow website](https://www.tensorflow.org/install/gpu), in section Software Requirements
    * pay attention to the versions - TensorFlow **1.13** will work only with CUDA Toolkit **10.0** (and probably higher)
    * make sure you've added CuDNN path to your PATH variable!

Please extend this file as any new requirements will arise!  
**That's all, folks! :)**
