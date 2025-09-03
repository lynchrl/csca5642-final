# Project Description

Deepfakes are AI-generated or AI-modified images that represent real subjects. As generator models have become increasingly sophisticated, it is almost impossible for humans to determine whether a good deepfake is real or not. Although the use of deepfakes is typically benign, malicious use has become a concern due to the potential for use of deepfakes to mask identities or otherwise trick people into dangerous or costly situations. The United State Deportment of Homeland Security has also expressed concern in this area, and issued a [whitepaper](https://www.dhs.gov/sites/default/files/publications/increasing_threats_of_deepfake_identities_0.pdf) on the subject.

This project aims to use deep learning models to help flag deepfakes in the domain of human faces. A real-world use case could be online identity fraud detection systems that assess submitted photo identification for deepfakes, or fake news detection for online articles that includes images with human faces.

The data we will use for the project is the [StyleGan-StyleGan2 Deepfake Face Images](https://www.kaggle.com/datasets/kshitizbhargava/deepfake-face-images/data) dataset available on Kaggle. It includes ~6k real and ~7k fake images, labeled by directory.

# Results Summary

The best performing model was found to be the **HP Search** model. The final architecture was 3 levels of convolutional blocks, each consisting of `Conv2D`, `BatchNorm`, `Conv2D`, `BatchNorm`, and `MaxPool2D` layers. Using `Flatten` as the final layer before the fully-connected `Dense` layer contributed to the best resuts, likely by keeping fine-grained signals for the sigmoid function. Finally, the smaller learning rate of `0.0001` performed well by helping the **HP Search** model avoid too much instability during training.

# Future Improvements

Due to time and resource limitations, this project only scratched the surface in the deepfake problem space. Future improvements and areas for exploration could include:

* Larger search space when using the Keras `RandomSearch` functionality to surface even more potential performance improvements.
* Experiment with slightly more complex architectures, such as adding skip-connections between blocks to retain some features discovered from earlier layers.
* Experiment with GANs for training, but in this case use the `discriminator` as the resulting model to test and use for real vs. fake classification.

# Notebook Availability

The notebook for the project is available [in this repository](https://github.com/lynchrl/csca5642-final/blob/main/csca-5642-final-project.ipynb) as well as on [Kaggle](https://www.kaggle.com/code/lynchrl/csca-5642-final-project).

# References

* https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/
* https://keras.io/examples/vision/image_classification_from_scratch/
* https://www.coursera.org/learn/introduction-to-deep-learning-boulder
* https://www.tensorflow.org/tutorials/images/cnn
* https://keras.io/api/layers/preprocessing_layers/image_preprocessing/rescaling/
* https://www.geeksforgeeks.org/deep-learning/what-is-batch-normalization-in-deep-learning/
* https://www.tensorflow.org/guide/data_performance
