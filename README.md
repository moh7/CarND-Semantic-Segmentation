# Semantic Segmentation
### Introduction
The goal of this project is to train a fully convolutional neural network (FCNN) to identify the free areas on the road from car dashcam images. The architecture of the FCNN model is based on VGG-16 image classifier with some additional layers for turning a regluar CNN to a FCNN. KITTI data was used for training and testing the network.

### Architecture
A pretrained VGG-16 CNN was turned into a FCNN by replacing the fully connected layer at the end of the network by a 1x1 convolution layer with a depth equal to the number of classes. The number of classes in our case is equal to 2, free road space, not free road space. Replacing the fully connected layer with a 1x1 convolution layer will let us preserve the spacial information in the network. To improve the performance of the network, skip connections were added to the architecture by first adding the 1x1 convolution of layer 4 to upsampled 1x1 convolution of layer 7, and then adding the upsampled of the obtained layer to the 1x1 convolution of layer 3. Upsampling at each step is required in order to be able to add two layers with the same dimension. Lastly, the output layer is upsampled again by the factor of 8 to obtain the original input image shape in the output. The entire procedure is summarized in the following.


```
# 1x1 convolution of layers 3, 4, and 7
conv7 = conv_1x1(layer7)
conv4 = conv_1x1(layer4)   
conv3 = conv_1x1(layer3)

# upsample by 2
out = conv2d_transpose(conv7)

# skip connection
out = add(out, conv4)           

# upsample by 2
out = conv2d_transpose(out)

# skip connection
out = add(out, conv3)

# upsample by 8
out = conv2d_transpose(out)
```

### Optimization and Training
The model was trained with the following parameters.

```
KEEP_PROB = 0.5
LEARNING_RATE = 1e-4
EPOCHS = 15
BATCH_SIZE = 5
```

### Results
Using the parameter values shown above and without data augmentation, the minimum of loss of 0.031 was obtained at epoch 10. No further improvement in the loss was observed after epoch 10. The following pictures show some samples of the performance of the trained semantic segmentation model.



![](/sample_images/um_000015.png)
![](/sample_images/um_000025.png)



### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)

 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow.
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy.
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
