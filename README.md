# Parallel-Autoencoder-Unsupervised-Feature-Learning

Feature engineering is a fundamental challenge in machine learning: how do we automatically discover good representations of data that capture its underlying structure? In this project, you will implement an Autoencoder-based unsupervised feature learning system for image classification on the CIFAR-10 dataset.

# File structure

- `include`: Contains header files
  - `cpu`: Header files for CPU-related tasks
  - `autoencoder.h`: Interface class representing an autoencoder. CPU version available.
  - `constants.h`: Defines constants for the project.
  - `dataset.h`: Defines a struct that represents a dataset.
- `src`: Source codes for the corresponding header file.

# Todo

- Update read dataset
- Print training process verbosely (training time per epoch, ...)
