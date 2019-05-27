## Assignment 2

#### Task 1
Making a CNN model from scratch to classify the images of the line dataset into the respective
96 classes and MNIST dataset into 10 classes <br>

There are two models one optimized and one unoptimized one

#### Task 2
Design a non-sequential convolutional neural network for classifying the line dataset. This
network will have 4 outputs based on the 4 kind of variations(length, width, color, angle).
You are required to divide your network architecture into two parts a) Feature network and
b) Classification heads. The feature network will be responsible for extracting the required
features from the input and attached to it would be the four classification heads one for each
variation. The network is shown in figure 1.
The first 3 classification heads are for 2 class problems namely length, width and color
classification. In all these the final layer contains a single neuron with a sigmoid activation
followed by binary crossentropy loss.
The last classification head is a 12 class problem for each 12 angles of variation. In this the
final layer contains 12 neurons with softmax activation and Categorical Cross entropy loss.


#### Task 3
##### Part 1: Visualizing Intermediate Layer Activations
Intermediate layer activations are the outputs of intermediate layers of the neural network. You
are required to plot the intermediate activations of the layers of the neural networks made in
section 1&2 for atleast 6 images.
5
##### Part 2: Visualizing Convnet Filters
In this part you have to visualize the filters in the convolutional neural network. This can be
done by running Gradient Descent on the value of a convnet so as to maximize the response
of a specific filter, starting from a blank input image. You are required to plot the filters of the
layers of the neural networks made in section 1&2.
##### Part 3: Visualizing Heatmaps of class activations
In this part you are required to plot heatmaps of class activations over input images. This will
help you visualize the regions of the input image the convnet is looking at. A class activation
heatmap is a 2D grid of scores associated with a particular output class, computed for every
location for an input image, indicating how important is each location is with respect to that
output class. You are required to plot the heatmaps for atleast 6 images for networks made in
section 1&2.
