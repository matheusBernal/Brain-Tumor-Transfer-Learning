# Brain Tumor Classification with Transfer Learning
This project implements a Convolutional Neural Network (CNN) using Transfer Learning to classify brain tumor images into four categories: Glioma, Meningioma, Pituitary, and No Tumor. It leverages a VGG16 model pre-trained on ImageNet, fine-tuned to adapt to the brain tumor classification task.
Key Features:
- Transfer Learning: Utilizes the pre-trained VGG16 model to take advantage of learned features from ImageNet, reducing training time and improving performance.
- Data Augmentation: Implements ImageDataGenerator to preprocess and augment the dataset for better model generalization.
- Custom CNN Architecture: Adds custom layers for brain tumor classification, including fully connected layers and dropout for regularization.
- Multi-class Classification: The model classifies images into four classes (Glioma, Meningioma, Pituitary, No Tumor) using softmax activation.
- Training Visualization: Plots training and validation accuracy and loss to evaluate model performance over time.
Dataset:
- Training Set: Images of brain tumors in various stages (Glioma, Meningioma, Pituitary).
- Validation Set: A separate validation dataset to assess model performance.
Requirements:
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
  
How to Run:
1. Clone this repository or open on google colab.
2. Ensure you have the necessary dependencies installed.
3. Place your dataset in the correct directory structure (`./Train`, `./Val`).
4. Run the training script to fine-tune the model and evaluate its performance.
Results:
- The model is capable of accurately classifying brain tumor images into the predefined categories.
