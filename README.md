# Animal Image Classification

## Objective
The goal of this project is to develop an image classification system capable of identifying animals in given images. The model should be able to classify 15 different species of animals.
## Dataset
The dataset consists of 15 folders, each containing images of dimensions 224 x 224 x 3. The classes in the dataset are:
- Bear
- Bird
- Cat
- Cow
- Deer
- Dog
- Dolphin
- Elephant
- Giraffe
- Horse
- Kangaroo
- Lion
- Panda
- Tiger
- Zebra

## Approach
To achieve accurate classification, we will explore various machine learning techniques, with a focus on Neural Networks and Transfer Learning. These methods are expected to provide an optimal solution for the required image classification task.

## Repository Structure
- `dataset/`: Contains the dataset with images organized into folders by class.
- `classify.py`: The source code for the image classification system, including a   Gradio-based graphical user interface (GUI).

- `notebooks/`: Contains Jupyter notebook for model development and training.
- `models/`: Output models and model checkpoints.
- `README.md`: This file, providing an overview of the project.

## Getting Started

### Training a New Model (Optional)
A trained model (based on the dataset) via the notebook is already saved in `models/`. To train a new model:

1. Run the Python notebook in `notebooks/`.
2. Feel free to customize hyperparameters, batch size, datasets, and augmentation to create your own model.

### Running the Image Classification System

To run the image classification system, follow these steps:

1. Clone the repository.
2. Install the required dependencies.
3. Run `classify.py`.
4. Upload test images and run the system via the Gradio interface.

## Screenshots

![App Screenshot](app_screenshots/SS1.jpg)

![App Screenshot](app_screenshots/SS2.jpg)

![App Screenshot](app_screenshots/SS3.jpg)

![App Screenshot](app_screenshots/SS4.jpg)

