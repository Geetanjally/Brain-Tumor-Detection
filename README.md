# 🧠 Brain Tumor Detection Project Overview
This project focuses on building a Convolutional Neural Network (CNN) from scratch to classify brain MRI images into four categories: Glioma, Meningioma, Pituitary, and No Tumor.

The model was trained using a labeled dataset of MRI images organized into separate folders for each tumor type. TensorFlow’s tf.data pipeline was used to efficiently load, normalize, and batch the images during training. The CNN learns directly from the image data, without using any pre-trained weights or transfer learning.

The trained model can predict the type of tumor present in new MRI images, assisting in early detection and diagnosis.
