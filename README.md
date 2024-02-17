# Distracted Driver Behavior Classification with VGG16

### INSAT IIA5

**Technologies:** Python, TensorFlow, NumPy, Keras, Pandas, Seaborn, Scikit-learn, Pickle, IPython, Matplotlib Pyplot, VGG16, CNN

## Introduction

This project aims to tackle the State Farm Distracted Driver Detection challenge on Kaggle. The objective is to classify distracted driver behaviors from dashboard camera images using the VGG16 convolutional neural network architecture.

## Project Scope

Conducted from December 13, 2023, to December 20, 2023, this was a semester project that I did to validate the "Deep Learning" subject in my 5th and final year of Industrial Computing and Automation (IIA5) at INSAT. It involved data preparation, model optimization, and performance evaluation phases. 

## Technical Details

- **Data Preparation**: Constructed train and test dataframes with an 80-20 ratio, implemented categorical label mapping, and one-hot encoding for classification.
- **TensorFlow Integration**: Utilized TensorFlow to convert 2D RGB 64x64 images into 4D tensors, facilitating efficient batch processing during model training.
- **Image Normalization**: Applied pixel value rescaling to mitigate input feature scale discrepancies, enhancing the network's ability to learn discriminative features.
- **VGG16 Model Optimization**: Modified the VGG16 architecture to predict the last layer using the flattened image vector, resulting in a memory-efficient model.
- **Training and Validation**: Trained the VGG16 model with categorical cross-entropy loss function and RMSprop optimizer, achieving a peak validation accuracy of 92.44% after 400 epochs.
- **Model Evaluation**: Analyzed performance using Matplotlib to plot loss and accuracy curves, achieving a precision of 91.88% and a recall of 91.82% on validation data.
- **Checkpoint Mechanism**: Implemented a checkpoint mechanism to save the best-performing model weights based on validation accuracy during training.

### Google Colab Link

Access the Google Colab notebook for this project [here](https://colab.research.google.com/drive/1F8JBWJBtJoCiaHv8ayARRMCsZ8-mHsAT).

## Getting Started

### Installation

1. Ensure compatibility and setup of required libraries including TensorFlow, NumPy, and Keras.
2. Clone the project repository to your local machine.
3. Install necessary Python packages using pip or conda.
4. Execute code cells sequentially.

### Usage

1. Prepare the dataset and ensure proper data preprocessing.
2. Train the VGG16 model using the provided scripts or Jupyter notebooks.
3. Evaluate model performance using provided evaluation scripts and visualizations.

## Contributing

Contributions aimed at enhancing the model's performance and extending its capabilities are welcome.

## License

This project is licensed under the [GPL-3.0 License](LICENSE).

## Contacts

For inquiries or collaboration opportunities, please contact:

- Elyes Khechine: elyeskhechine@gmail.com
