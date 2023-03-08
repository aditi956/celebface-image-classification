# CelebFace Image Classification Using Decision Tree Classifier and CNN

In this project, I have performed Gender classification to identify the gender of celeb images using the CelebFaces Attributes Dataset (CelebA) dataset
from kaggle for creating the image classification models.

## Background

Convolutional neural network (CNN) is a state-of-the art architecture
used in image classification. CNN architectures
use a series of convolution and pooling layers followed by
fully connected dense layers for feature extraction. Through
this project, my objective has been to explore the impact of using Decision Tree
Classifiers in the output layer of CNNs to perform binary and multiclass
classification and the impact different CNN hyperparameters
have on model performance. I have also examined the
impact of using pre-trained models in the CNN architecture
instead of training a CNN from scratch. Ans finally, I have drawn comparison
details between those models. CelebFaceA dataset was used
for the analysis.

## Experiment

CelebA is a large-scale face dataset of facial images with
more than 200K celebrity images (RGB). I have used two
other datasets available for CelebA in our project. One contains
40 binary image annotations like wearing glasses, bald,
arched eyebrows, pale skin, no beard etc. while the other dataset
contains the information of data partitioning, providing
the image IDs which can be used for training, validation and
testing.
The primary language used here is python for the analysis using libraries like
pandas, numpy, keras and tensorflow for the implementation.

To identify the gender in the images, two machine learning
algorithms were implemented - Decision trees and CNN.
The results from both were compared using performance
metrics like accuracy, loss, and training time of the models.
The images in the dataset are of good quality with negligible
background noise. So, there was no need to clean the
images. As a part of pre-processing, the images were normalized
by dividing the pixel intensities by 255. As the number
of images in the dataset were very large and due to limitations
of computational resources, the images were sampled
randomly. I sampled 10,000 images for training and
5000 images for validation and testing. I also ensured that
these datasets are balanced by sampling an equal number of
images from each class.
To implement the decision tree, I used the dataset
containing 40 facial attributes for each image. These attributes
give information about facial features in the image like
smiling, bangs, rosy cheeks, oval face, pointy nose and others.
The same dataset also has a ‘male’ column, a binary variable
which was used as target.
By using the images alone as input for the problem, without
providing explicit facial features, I implemented
CNNs, experimenting with different configurations of convolution
layers, pooling layers, number of filters, filter size
etc. 
At first I started with only 2 layers and subsequently increased
the number of layers, filters, and filter size. To understand
the effect of pooling, the pooling layers were removed
in one of the models, increasing the stride. The optimizer
used was Adam, and the loss function used was binary
cross entropy. The activation function used in the layers was
ReLU, and sigmoid function was used in the output layer.
I trained the models for 10 epochs, due to high run times.
The training and test accuracies along with the losses of the
models were compared to understand which model outperformed
the other.

## Performance Table

Gender Classification (Binary Classification)
Table I gives the details about each of the model configurations.
Table II shows the accuracy and loss for each of these
models. Decision tree classifier performed decently with an
accuracy of 85%. Changing the depth of the tree and impurity
indexes didn’t change the performance considerably. As
seen from the values, it is observed that the CNN based
models give satisfactory accuracies (>90%) with layers
more than 2. The best performing model is the one with 5
layers and filter numbers increasing as 32, 64 and 128. After
the pooling layers were removed, and stride was increased,
the accuracy decreased. CNN with only two layers gave the
worst performance with 49% accuracy.

### Table I:

| Model | Parameter   | 
| :-----: | :---: |
| CNN 1 | 2 convolution layers, 2 max pooling layers, 1 dense layer (64 nodes). Number of filters: 32, 64. Filter size: 3 x 3 (convolution), 2 x (pooling)| | 
| CNN 2 | 4 convolution layers, 4 max pooling layers,1 dense layer (64 nodes). Number of filters: 32, 64 (only last layer has 64 filters) Filter size: 3 x 3 (convolution), 2 x 2 (pooling)| 
| CNN 3 | 4 convolution layers, 1 dense layer (64 nodes) Number of filters: 32, 64 (only last layer has 64 filters) Filter size: 3 x 3 (convolution), 2 x 2 (pooling) Stride = 4 | 
| CNN 4 | 5 convolution layers, 1 dense layer (128 nodes). Number of filters: 32, 64, 128 (only last layer has 128 filters, others 2 each). Filter size: 3 x 3 (convolution), 2 x 2 (pooling) | 

### Table II:

| Model | Training accuracy  | Test accuracy | Training loss | Test loss |
| :-----: | :-----: |:-----: |:-----: |:-----: |
| CNN 1 | 49.4%| 54.3%| 0.7|1.5 |
| CNN 2 | 98.4%| 95.4%| 0.04|0.12 |
| CNN 3 | 96%| 92%| 0.1|0.22 |
| CNN 4 | 98.1%| 95.8%| 0.05|0.14 |
| Decision Tree| 96% | 84.6% | NA | NA|

The CNN models were trained for 10 epochs, with a batch
size of 32. The training times per step in an epoch for images
of size 218 x 178 are mentioned in table III (the training
times are averaged across all epochs). As per theory, the
training time increases with increasing number of layers.
The training time decreases when the pooling layers are removed,
and the stride is increased.

### Table III:

| Model | Parameter   | 
| :-----: | :---: |
| CNN 1 | 217 ms | 
| CNN 2 | 644 ms | 
| CNN 3 | 248 ms | 
| CNN 4 | 944 ms | 

Further, the performance, metrics were evaluated using precision,
recall and F1 score for three main architectures

### Table IV

| Model | Precision | Recall |F1 - score |
| :-----: | :-----: |:-----: |:-----: |
| CNN 1 | Man: 0.61 Woman: 0.64| Man: 0.76 Woman: 0.73| Man: 0.66 Woman:0.68 |
| CNN 2 | Man: 0.95 Woman: 0.94| Man: 0.95 Woman: 0.94| Man: 0.95 Woman:0.95 |
| CNN 4 |Man: 0.91 Woman: 0.94| Man: 0.91 Woman: 0.94| Man: 0.91 Woman:0.94 |

## Conclusion
We observed that a vanilla CNN performed better than a decision
tree classifier in the gender classification problem. As
we increase the number of layers, the model trains better but
risks overfitting the data. We also observed that deeper ConvNets
performed better than shallower ConvNets. Moreover,
Dropouts helped in reducing overfitting and improving
model performance. Larger stride values led to quicker
down sampling of the images negatively impacting our
model performance. Since training a single configuration
takes hours, we limited our experiments to a small subset of
configuration choices.
