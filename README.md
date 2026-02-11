## Exploring convutional layers through data and experiments

---

# Table of Contents
1. [Project Overview](#1Project-Overview)
2. [Dataset Description](#2Dataset-Description)
3. [Exploratory Data Analysis (EDA)](#3Exploratory-Data-Analysis-EDA)
4. [Baseline Model (Non-Convolutional)](#4Baseline-Model-Non-Convolutional)
5. [Convolutional Neural Network Architecture](#5Convolutional-Neural-Network-Architecture)
6. [Controlled Experiments](#6Controlled-Experiments)
7. [Deployment with sagemaker](#7Deployment-with-sagemaker)
8. [Conclusion](#8Conclusion)
9. [Author](#9Author)

---

# 1. Project Overview

This project delves into the application of convolutional neural networks as architectural components rather than as a black box.
The primary objective of the tutorial is to examine the inductive biases of the convolutional layers, which are typically extended by learning parameters such as kernel size, depth, stride, and pool.

A convolutional model is created from scratch and compared to a non-convolutional baseline with a realistic image dataset. Controlled tests are carried out to evaluate the influence of individual parameters in the convolution process.

The current project is associated with an academic assignment, and the topic of interest is associated with architectural reasoning and hyperparameter tuning.

# 2. Dataset Description
- Dataset: (e.g. CIFAR-10 / Fashion-MNIST / custom dataset — specify here)
- Source: (TensorFlow Datasets / torchvision / Kaggle)
- Data Type: Image data (2D tensors with channels)
- Number of Classes: (e.g. 10)
- Image Shape: (e.g. 32×32×3)

Why this dataset?

- This dataset is appropriate for convolutional architectures because:
- It has strong spatial structure
- Local patterns (edges, textures, shapes) are meaningful
- Translation invariance is desirable
- Fully connected models struggle to scale efficiently

---

# 3. Exploratory Data Analysis (EDA)

The EDA focuses on understanding structure rather than exhaustive statistics:

- Dataset size and class distribution
- Image dimensions and number of channels
- Visualization of sample images per class

Preprocessing steps:

- Normalization
- Optional resizing
- Label encoding

See the notebook for visual examples and analysis.

Here are some sample images from the dataset first of all the class distribution and then the images themselves:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/classDistribution.PNG)

And now some sample images from the dataset:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/photosDataset.PNG)

---

# 4. Baseline Model (Non-Convolutional)

A simple neural network without convolutional layers is implemented as a reference point.

Architecture:

- Flatten layer
- Fully connected (Dense) layers
- ReLU activations
- Softmax output

Observations:

- High number of parameters
- Poor generalization compared to CNN
- No spatial inductive bias
- Sensitive to image dimensionality

This model establishes a baseline for comparison.

Now we will see the architecture of the baseline model with the summary of the model using tensorflow:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/baselineModelSummary.PNG)

After seeing the summary we can see the epochs and the accuracy of the model:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/epoch.PNG)

And last but not least the confusion matrix of the model, that shows the accuracy vs loss of the model:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/accurrancyvsloss.PNG)

---

# 5. Convolutional Neural Network Architecture

The CNN is designed intentionally, not copied from a tutorial.

Design Choices:

- Convolutional layers: (e.g. 2–3 layers)
- Kernel size: (e.g. 3×3)
- Stride & Padding: (e.g. stride=1, same padding)
- Activation function: ReLU
- Pooling: (e.g. MaxPooling 2×2)
- Classifier: Flatten + Dense layers

Each choice is justified based on:

- Spatial locality
- Parameter efficiency
- Feature hierarchy

Here we can see with tensor flow the conventional neural network architecture that we have implemented:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/cnnModel.PNG)

---

# 6. Controlled Experiments

One convolutional parameter is varied while keeping all others fixed.

Here we choose 3x3 kernel size as the baseline and compare it to 5x5 and 7x7. And we can see the depth of 3x3 kernel size with 32 filters and then with 64 filters as we show the results of the experiments in the following images:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/depth.PNG)

Now we can see the loss and accuracy of the model with 3x3 kernel size and 32 filters:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/cnnLossVsAccurancy.PNG)

After that we can see the difference between the baseline model and the convolutional model with 3x3 kernel size and 32 filters:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/baselineVsCnn.PNG)

And we can see the new photos of the model with 3x3 kernel size and 32 filters:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/newPhotos.PNG)

Last we can see the compresive model comparing the baseline model with the convolutional model with 3x3 kernel size and 32 filters:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/cmC.PNG)

---

# 7. Deployment with sagemaker:
For the deployment of the model we have used AWS Sagemaker, which is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning models quickly. We have used the sagemaker python sdk to deploy our model in the cloud and make it available for inference. But because we dont have access to internet we cannot download the dataset and train the model, so we must go to kaggle and download the dataset and upload it to an s3 bucket, and from there we can access the dataset from sagemaker and train the model.

First we go to aws and to sagemaker, after that we open a jupyter notebook instance and create a new notebook.

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/jupitersagemaker.PNG)

After that we create a notebook and we upload the dataset and the notebook created:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/startingjupiterlab3.PNG)
![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/.PNG)

Then we run the notebook in sagemaker and we can see the results:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/results.PNG)

Now we have to create a bucket in s3 to store the model:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/s3bucket.PNG)

When the bucket is created we can save the model in the bucket:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/tarslab3.PNG)

After that we have to go to sagemaker and create a model:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/creatingmodellab3.PNG)

Due to AWS Lab restrictions, real-time endpoint deployment was not permitted:

![image](https://github.com/CamiloFdez/Exploring-Convolutional-Layers-Lab-3/blob/main/images/deployerror.PNG)

Instead, inference was successfully demonstrated locally using the trained model artifact.


---

# 8. Conclusion
This project proves that convolutional neural networks are not just enhanced forms of fully connected models; they actually embed powerful inductive biases appropriate to images.

Compared to the baseline model, the CNN performed better in terms of generalization with many fewer parameters by utilizing the benefits of local connectivity, although it used weight sharing and other spatial hierarchies. The benefits enable the layers of the convolutional network to efficiently learn relevant features in the input data, such as edges and texture.

First of all, the controlled experiments point out that an influence of the architecture’s choice, such as kernel size, depth, or the usage of pooling, on the model’s performance is crucial. It is worth mentioning that small differences in the convolutional architecture’s choice entail considerable trade-offs from the model’s complexity.

Overall, this work reinforces the need to understand the role of architectural reasoning in neural network design. Rather than viewing the system as a box, this helps us understand how efficient and interpretable machine learning systems can be created.

---

# 9. Author
- Camilo Fernandez [GitHub](https://github.com/CamiloFdez)

---