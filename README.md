# Image_Classification

---

# Distracted Driver Detection using Transfer Learning with MobileNetV2

This repository contains code for building a distracted driver detection model using transfer learning with the MobileNetV2 architecture. The model aims to classify images of drivers into various categories of distraction, such as texting, talking on the phone, etc.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Getting Started

1. Clone this repository:

    ```bash
    git clone https://github.com/your_username/distracted-driver-detection.git
    cd distracted-driver-detection
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset. In this example, we assume the dataset is stored in the directory `/kaggle/input/state-farm-distracted-driver-detection/imgs/train`. You may need to adjust the directory path in `main()` function accordingly.

4. Run the main script:

    ```bash
    python distracted_driver_detection.py
    ```

## Usage

The main script `distracted_driver_detection.py` contains the following components:

- Data preparation: Augments the training data and splits it into training and validation sets using `ImageDataGenerator`.
- Model creation: Utilizes the pre-trained MobileNetV2 model without the top classification layer and adds custom layers for fine-tuning.
- Model compilation: Compiles the model with the Adam optimizer, categorical crossentropy loss, and accuracy metric.
- Training: Trains the model on the augmented data with early stopping to prevent overfitting.
- Evaluation: Computes the confusion matrix and accuracy score on the validation set to assess model performance.

## Code Explanation

### Base Model and Additional Layers
In this project, the MobileNetV2 architecture serves as the base model for transfer learning. MobileNetV2 is a lightweight convolutional neural network (CNN) that is efficient for mobile and embedded vision applications. By utilizing a pre-trained MobileNetV2 model, we benefit from its feature extraction capabilities learned from a large dataset (ImageNet).

On top of the MobileNetV2 base model, additional layers are added to customize the architecture for our specific task of distracted driver detection. These additional layers include:
- **GlobalAveragePooling2D**: This layer reduces the spatial dimensions of the feature maps produced by the base model. It computes the average value of each feature map, effectively summarizing the spatial information.
- **Dropout**: Dropout is used to address overfitting by randomly deactivating a fraction of neurons during training. In this case, a dropout rate of 0.5 is applied, meaning that 50% of the neurons are randomly dropped out during each training iteration. This helps prevent the model from relying too heavily on specific features or correlations in the training data, thus improving generalization performance on unseen data.

### Data Augmentation
To increase the diversity of the training data and improve the model's ability to generalize to new scenarios, data augmentation techniques are employed. These techniques include:
- **Rotation**: Randomly rotates images by up to 20 degrees.
- **Width and Height Shift**: Randomly shifts images horizontally and vertically by up to 20% of the width and height, respectively.
- **Shear Transformation**: Applies a shear transformation with a maximum intensity of 20%.
- **Zoom**: Randomly zooms images by up to 20%.
- **Horizontal Flip**: Randomly flips images horizontally.

### Addressing Overfitting
The original code, without data augmentation and dropout, resulted in only 10% accuracy. To address overfitting and improve performance, several strategies were employed:
- **Data Augmentation**: By augmenting the training data with various transformations, the model learns to be more robust to variations in the input images, reducing the likelihood of overfitting.
- **Dropout**: The addition of dropout layers helps regularize the model by preventing co-adaptation of neurons, thus reducing overfitting.
- **Early Stopping**: Early stopping is implemented as a callback during training. It monitors the validation loss and stops training when the loss stops decreasing, thus preventing overfitting by stopping the training process at the right time.

### Results
With the implementation of data augmentation, dropout, and other strategies to address overfitting, the model's performance significantly improved. The validation accuracy increased to approximately 70%, demonstrating the effectiveness of these techniques in improving model generalization and performance on unseen data.

---
