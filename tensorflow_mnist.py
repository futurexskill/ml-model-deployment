# coding: utf-8

# ## Handwriting recognition using Tensorflow
# 

# In[53]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

## For Jupyter notebook you can avoid creating multiple loops of creating session variables by declaring this
sess = tf.InteractiveSession()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[55]:


print(mnist.train.num_examples)
print(mnist.test.num_examples)
print(mnist.validation.num_examples)


# ## Print a sample number

# In[61]:


import matplotlib.pyplot as plt

def display_sample(num):
    #Print the one-hot array of this sample's label 
    print(mnist.train.labels[num])  
    #Print the label converted back to a number
    # convert one_hot format to a uman redable digit 
    label = mnist.train.labels[num].argmax(axis=0)
    print (label)
    #Reshape the 768 values to a 28x28 image
    image = mnist.train.images[num].reshape([28,28])
    plt.title('Sample: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

# Pick any value between 0 to 55000    
display_sample(1234)


# In[57]:


mnist.train.images.shape


# In[58]:


mnist.train.images[1].shape


# In[59]:


mnist.train.images[1].reshape(28,28)


# Data is already normalized

# In[63]:


mnist.train.images[1].min()


# In[64]:


mnist.train.images[1].max()


# In[65]:


plt.imshow(mnist.train.images[1].reshape(784,1))


# In[66]:


plt.imshow(mnist.train.images[1].reshape(784,1),cmap='gist_gray',aspect=0.02)


# ## Visualize what neural network is seeing

# In[71]:


import numpy as np

images = mnist.train.images[0].reshape([1,784])
# for 500 training samples
for i in range(1, 500):
    images = np.concatenate((images, mnist.train.images[i].reshape([1,784])))
plt.imshow(images, cmap=plt.get_cmap('gray_r'))
plt.show()
# each row represents a training sample


# This is showing the first 500 training samples, one on each row. Imagine each pixel on each row getting fed into the bottom layer of a neural network 768 neurons (or "units") wide as we train our neural network.
# 
# So let's start setting up that artificial neural network. We'll start by creating "placeholders" for the input images and for the "correct" labels for each. Think of these as parameters - we build up our neural network model without knowledge of the actual data that will be fed into it; we just need to construct it in such a way that our data will fit in.
# 
# So our "input_images" placeholder will be set up to hold an array of values that consist of 784 floats (28x28), and our "target_labels" placeholder will be set up to hold an array of values that consist of 10 floats (one-hot format for 10 digits.)
# 
# While training, we'll assign input_images to the training images and target_labels to the training lables. While testing, we'll use the test images and test labels instead.

# ## Placeholders

# In[44]:


input_images = tf.placeholder(tf.float32, shape=[None, 784])
# Output layer with one_hot label
target_labels = tf.placeholder(tf.float32, shape=[None, 10])


# So let's set up our deep neural network. We'll need an input layer with one node per input pixel per image, or 784 nodes. That will feed into a hidden layer of some arbitrary size - let's pick 512. That hidden layer will output 10 values, corresonding to scores for each classification to be fed into softmax.
# 
# We'll need to reserve variables to keep track of the all the weights and biases for both layers:

# ## Variables

# In[45]:


hidden_nodes = 512

input_weights = tf.Variable(tf.truncated_normal([784, hidden_nodes]))
input_biases = tf.Variable(tf.zeros([hidden_nodes]))

hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes, 10]))
hidden_biases = tf.Variable(tf.zeros([10]))


# Now let's set up the neural network itself. We'll define the input layer and associate it with our placeholder for input data. All this layer does is multiply these inputs by our input_weight tensor which will be learned over time.
# 
# Then we'll feed that into our hidden layer, which applies the ReLU activation function to the weighted inputs with our learned biases added in as well.
# 
# Finally our output layer, called digit_weights, multiplies in the learned weights of the hidden layer and adds in the hidden layer's bias term.

# In[46]:


input_layer = tf.matmul(input_images, input_weights)
hidden_layer = tf.nn.relu(input_layer + input_biases)
digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases


# This sets up a deep neural network like the one we talked about in our slides.
# 
# output layer
# 
# hidden layer
# 
# input layer
# 
# Next we will define our loss function for use in measuring our progress in gradient descent: cross-entropy, which applies a logarithmic scale to penalize incorrect classifications much more than ones that are close. Remember digit_weights is the output of our final layer, and we're comparing that against the target labels used for training.

# In[47]:


loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=digit_weights, labels=target_labels))


# Now we will set up our gradient descent optimizer, initializing it with an aggressive learning rate (0.5) and our loss function defined above.
# 
# That learning rate is an example of a hyperparameter that may be worth experimenting with and tuning.

# In[48]:


optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss_function)


# Next we'll want to train our neural network and measure its accuracy. First let's define some methods for measuring the accuracy of our trained model. 
# 
# correct_prediction will look at the output of our neural network (in digit_weights) and choose the label with the highest value, and see if that agrees with the target label given. During testing, digit_weights will be our prediction based on the test data we give the network, and target_labels is a placeholder that we will assign to our test labels. Ultimately this gives us a 1 for every correct classification, and a 0 for every incorrect classification.
# 
# "accuracy" then takes the average of all the classifications to produce an overall score for our model's accuracy.

# In[49]:


correct_prediction = tf.equal(tf.argmax(digit_weights,1), tf.argmax(target_labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Let's train this thing and see how it works! 
# 
# We'll set up a Tensorflow session, and initialize our variables. Next we will train our network in 2000 steps (or "epochs") with batches of 100 samples from our training data. At each step, we assign the input_images placeholder to the current batch of training images, and the target_labels placeholder to the current batch of training labels.
# 
# Once training is complete, we'll measure the accuracy of our model using the accuracy graph we defined above. While measuring accuracy, we assign the input_images placeholder to our test images, and the target_labels placeholder to our test labels.

# In[51]:


tf.global_variables_initializer().run()

for x in range(2000):
    batch = mnist.train.next_batch(100)
    optimizer.run(feed_dict={input_images: batch[0], target_labels: batch[1]})
    if ((x+1) % 100 == 0):
        print("Training epoch " + str(x+1))
        print("Accuracy: " + str(accuracy.eval(feed_dict={input_images: mnist.test.images, target_labels: mnist.test.labels})))


# You should have about 92% accuracy. 
# 
# Let's take a look at some of the misclassified images and see just how good or bad our model is, compared to what your own brain can do. We'll go through the first 100 test images and look at the ones that are misclassified:

# In[52]:


for x in range(100):
    # Load a single test image and its label
    x_train = mnist.test.images[x,:].reshape(1,784)
    y_train = mnist.test.labels[x,:]
    # Convert the one-hot label to an integer
    label = y_train.argmax()
    # Get the classification from our neural network's digit_weights final layer, and convert it to an integer
    prediction = sess.run(digit_weights, feed_dict={input_images: x_train}).argmax()
    # If the prediction does not match the correct label, display it
    if (prediction != label) :
        plt.title('Prediction: %d Label: %d' % (prediction, label))
        plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
        plt.show()


# To be honest, I'd be a little unsure about some of those myself!
# 
# ## Exercise
# 
# See if you can improve upon the accuracy. Try using more hidden neurons (nodes). Try using fewer! Try a different learning rate. Try adding another hidden layer. Try different batch sizes. What's the best accuracy you can get from this multi-layer perceptron?