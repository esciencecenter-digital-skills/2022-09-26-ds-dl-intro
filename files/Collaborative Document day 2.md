![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document day 2

2022-09-26 Introduction to Deep Learning.

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------


## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## 🖥 Workshop website

[Link to workshop website](https://esciencecenter-digital-skills.github.io/2022-09-26-ds-dl-intro)

🛠 Setup: 
[link](https://esciencecenter-digital-skills.github.io/2022-09-26-ds-dl-intro/#setup)

Download files:
[weather dataset prediction csv](https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1)

[Link to CIFAR10 data](https://www.cs.toronto.edu/~kriz/cifar.html)

#### Certificate of attendance
If you want to receive a certificate of attendance please send an email to training@esciencecenter.nl . The eScience Center will provide a certificate if you attended both days of the training.

## 👩‍🏫👩‍💻🎓 Instructors

Dafne van Kuppevelt, Sven van der Burg

## 🧑‍🙋 Helpers

Pranav Chandramouli, Laura Ootes, Suvayu Ali

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city
'
## 🗓️ Agenda
| Time | Topic |
|--:|:---|
| 9:30 | 	Welcome and recap |
| 9:45 | Monitor the training processs |
| 10:30 | Coffee break |
| 10:40 | Monitor the training processs |
| 11:30 | break |
| 11:40 | Monitor the training processs< |
| 12:30 | Lunch Break |
| 13:30 | Advanced layer types |
| 14:30 | Break |
| 14:40 | Advanced layer types |
| 15:30 | Lunch Break |
| 15:40 | Advanced layer types |
| 16:15 | Post-workshop survey |
| 16:30 | Drinks |

## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building, be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🔧 Exercises

### Exercise 1: Explore the dataset
#### Let’s get a quick idea of the dataset.

- How many data points do we have?
- How many features does the data have (don’t count month and date as a feature)?
- What are the different measured variable types in the data and how many are there (humidity etc.) ?
- (Optional) Plot the distributions of the different features in the dataset. What can you learn from this?

### Exercise 2: Architecture of the Network
#### As we want to design a neural network architecture for a regression task, see if you can first come up with the answers to the following questions:

- What must be the dimension of our input layer?
- We want to output the prediction of a single number. The output layer of the NN hence cannot be the same as for the classification task earlier. This is because the softmax activation being used had a concrete meaning with respect to the class labels which is not needed here. What output layer design would you choose for regression?
- (Optional) If next to the number of sunshine hours, we would also like to predict the precipitation. How would you go about this? 

### Exercise 3: Try to reduce the degree of overfitting by lowering the number of parameters
We can keep the network architecture unchanged (2 dense layers + a one-node output layer) and only play with the number of nodes per layer.

Try to lower the number of nodes in one or both of the two dense layers and observe the changes to the training and validation losses.
- Is it possible to get rid of overfitting this way?
- Does the overall performance suffer or does it mostly stay the same?
- How low can you go with the number of parameters without notable effect on the performance on the validation set?
- (optional) How low can you go with the number of parameters without notable effect on the performance on the validation set?



### Exercise 4: Simplify the model and add data

You may have been wondering why we are including weather observations from
multiple cities to predict sunshine hours only in Basel. The weather is
a complex phenomenon with correlations over large distances and time scales,
but what happens if we limit ourselves to only one city?

1. Since we will be reducing the number of features quite significantly,
we should afford to include more data. Instead of using only 3 years, use
8 or 9 years!
2. Remove all cities from the training data that are not for Basel.
You can use something like:
~~~
cols = [c for c in X_data.columns if c[:5] == 'BASEL']
X_data = X_data[cols]
~~~
3. Now rerun the last model we defined which included the BatchNorm layer.
Recreate the scatter plot comparing your prediction with the baseline
prediction based on yesterday's sunshine hours, and compute also the RMSE.
Note that even though we will use many more observations than previously,
the network should still train quickly because we reduce the number of
features (columns).
Is the prediction better compared to what we had before?
4. (Optional) Try to train a model on all years that are available, and all features from all cities. How does it perform?


### Advanced layer types
#### Exercise 5: Explore the data

Familiarize yourself with the CIFAR10 dataset. To start, consider the following questions:
- What is the dimension of a single data point? What do you think the dimensions mean?
- What is the range of values that your input data takes?
- What is the shape of the labels, and how many labels do we have?
- (Optional) Pre-trained models Step 4 of our 'deep learning workflow' is: 'Choose a pre-trained model or build a new architecture from scratch'. We are going to build a new architecture from scratch to get you familiar with the convolutional neural network basics. But in the real world you wouldn't do that. So the challenge is: Browse the web for (more) existing architectures or pre-trained models that are likely to work well on this type of data. Try to understand why they work well for this type of data.


### Exercise 6: Convolutional neural networks
In groups of 3/4, answer the following questions. Write your answers in the collaborative document.

##### 1. Border pixels
What, do you think, happens to the border pixels when applying a convolution?
##### 2. Number of parameters
Suppose we apply a convolutional layer with 100 kernels of size 3 * 3 * 3 (the last dimension applies to the rgb channels) to our images of 32 * 32 * 3 pixels. How many parameters do we have? Assume, for simplicity, that the kernels do not use bias terms. Compare this to the answer of the previous exercise
##### 3. A small convolutional neural network
So let’s look at a network with a few convolutional layers. We need to finish with a Dense layer to connect the output cells of the convolutional layer to the outputs for our classes.
```python=
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model_small")

model.summary()
```
Inspect the network above:
* What do you think is the function of the `Flatten` layer?
* Which layer has the most parameters? Do you find this intuitive?

#### 4. (optional) A state-of-the-art network
Pick a model from https://paperswithcode.com/sota/image-classification-on-cifar-10 . Try to understand how it works.


### Recap and Q&A Exercise: Recap
Take a few minutes to write down your thoughts on what we learned about deep learning in this course:
* What questions do you still have?
* Whether there are any incremental improvements that can benefit your projects?
* What’s nice that we learnt but is overkill for your current work?

## 🧠 Collaborative Notes

**Recap:** [notes from yesterday](https://hackmd.io/xErjHpAEQjKgUGIAkxd0tg?view).

### Monitoring the training process
Let's look at a weather dataset (linked above).

```python=
import pandas as pd

filename = 'https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1'
data = pd.read_csv(filename)
data.head()
```
|  | DATE|	MONTH|	BASEL_cloud_cover|	BASEL_humidity	| ... |
|--|--|--|--|--|--|
|0	|20000101|	1|	8|	0.89| ... |
|1	|20000102|	1|	8|	0.87| ... |
|2	|20000103|	1|	5|	0.81| ... |	

Let's explore the dataset:
```python=6
data.shape   # rows, columns
data.columns # all column names
```

#### Prepare the data
Let's work with first 3 years worth of data (for conveninence)
```python=
nr_rows = 365*3
# data
X_data = data.loc[:nr_rows].drop(columns=['DATE', 'MONTH'])

# labels (sunshine hours the next day)
y_data = data.loc[1:(nr_rows + 1)]["BASEL_sunshine"]
```

In a typical dataset you might even need to handle missing values, and other irregularities, however here we won't need to do that.

We will split the data into: training, testing, and validation samples.  The validation sample is a new concept, it is going to be used to do our hyperparameter search.  If we used our testing sample for that, we would overfit on it.  We will split 30% of the dataset into test and validation sets, and use the

```python=7
from sklearn.model_selection import train_test_split

X_train, X_holdout, y_train, y_holdout = train_test_split(X_data, y_data, test_size=0.3, random_state=0)

X_val, X_test, y_val, y_test = train_test_split(X_holdout, y_holdout, test_size=0.5, random_state=0)
```
#### Build a network
*Note:* we are going to use the features from all locations, to predict the sunshine/precipitation at Basel.  Weather at different locations maybe correlated, we leave it up to the model to figure it out.

```python=
from tensorflow import keras

def create_nn():
    # Input layer
    inputs = keras.Input(shape=(X_data.shape[1],), name='input')

    # Dense layers
    layers_dense = keras.layers.Dense(100, 'relu')(inputs)
    layers_dense = keras.layers.Dense(50, 'relu')(layers_dense)

    # Output layer
    outputs = keras.layers.Dense(1)(layers_dense)

    return keras.Model(inputs=inputs, outputs=outputs, name="weather_prediction_model")
```

We can then create the model and look at the summary
```python=
model = create_nn()
model.summary()
# Model: "weather_prediction_model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input (InputLayer)          [(None, 89)]              0         
#                                                                 
#  dense (Dense)               (None, 100)               9000      
#                                                                 
#  dense_1 (Dense)             (None, 50)                5050      
#                                                                 
#  dense_2 (Dense)             (None, 1)                 51        
#                                                                 
# =================================================================
# Total params: 14,101
# Trainable params: 14,101
# Non-trainable params: 0
# _________________________________________________________________
```

Question: There are so many parameters when we have only 89 inputs.  How does this not overfit?
Partial answer: In deep learning it has been observed that it is not a problem.

Let's train the model:
```python=20
model.compile(optimizer="adam", loss='mse', metrics=[keras.metrics.RootMeanSquaredError()])
```
The metric is meant for a way to monitor the model during training; note how it is printed along with the loss:
```python=21
history = model.fit(X_train, y_train, batch_size=32, epochs=200)
# ...
# Epoch 3/200
# 24/24 [==============================] - 0s 2ms/step - loss: 12.1449 - root_mean_squared_error: 3.4849
# Epoch 4/200
# 24/24 [==============================] - 0s 2ms/step - loss: 11.8187 - root_mean_squared_error: 3.4378
# ...
```


```python=28
import seaborn as sns
import matplotlib.pyplot as plt

history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df['root_mean_squared_error'])
plt.xlabel("epochs")
plt.ylabel("RMSE")
```
![](https://i.imgur.com/pTH1Uhn.png)

#### Hyperparameter tuning

```python=
y_train_predicted = model.predict(X_train)
y_test_predicted = model.predict(X_test)
```

Let's compare the prediction on our training and testing samples:
```python=
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plt.style.use('ggplot')  # optional, that's only to define a visual style
axes[0].scatter(y_train_predicted, y_train, s=10, alpha=0.5, color="teal")
axes[0].set_title("training set")
axes[0].set_xlabel("predicted sunshine hours")
axes[0].set_ylabel("true sunshine hours")

axes[1].scatter(y_test_predicted, y_test, s=10, alpha=0.5, color="teal")
axes[1].set_title("test set")
axes[1].set_xlabel("predicted sunshine hours")
axes[1].set_ylabel("true sunshine hours")
```
![](https://i.imgur.com/cG87URk.png)

The model isn't generalising well, it does significantly better on the training set compared to the test set.  We can also calculate some metrics.

```python=12
train_metrics = model.evaluate(X_train, y_train, return_dict=True)
test_metrics = model.evaluate(X_test, y_test, return_dict=True)
print('Train RMSE: {:.2f}, Test RMSE: {:.2f}'.format(train_metrics['root_mean_squared_error'], test_metrics['root_mean_squared_error']))
# 24/24 [==============================] - 0s 2ms/step - loss: 0.4201 - root_mean_squared_error: 0.6481
# 6/6 [==============================] - 0s 4ms/step - loss: 15.3703 - root_mean_squared_error: 3.9205
# Train RMSE: 0.65, Test RMSE: 3.92
```

We can establish a baseline, to track if our model is training well.  Say our baseline is we just predict today's weather as same as the previous day.
```python=18
y_baseline_prediction = X_test['BASEL_sunshine']

plt.figure(figsize=(5, 5), dpi=100)
plt.scatter(y_baseline_prediction, y_test, s=10, alpha=0.5)
plt.xlabel("sunshine hours yesterday")
plt.ylabel("true sunshine hours")
```
![](https://i.imgur.com/gXDHfAw.png)

However this is difficult to conclude anything, so let's also calculate the metrics we used before.
```python=
from sklearn.metrics import mean_squared_error
rmse_nn = mean_squared_error(y_test, y_test_predicted, squared=False)
rmse_baseline = mean_squared_error(y_test, y_baseline_prediction, squared=False)
print('NN RMSE: {:.2f}, baseline RMSE: {:.2f}'.format(rmse_nn, rmse_baseline))
# NN RMSE: 3.92, baseline RMSE: 3.88
```

Let's redo our training, with our validation set
```python=
model = create_nn()
model.compile(optimizer='adam',
              loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError()])

history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    validation_data=(X_val, y_val),
                    verbose=2)
```

If we now compare the training error with the validation set, we see the model starts overfitting early on in the training process (see plot below).
```python=11
history_df = pd.DataFrame.from_dict(history.history)
history_df.head()
```
|    |    loss |   root_mean_squared_error |   val_loss |   val_root_mean_squared_error |
|---:|--------:|--------------------------:|-----------:|------------------------------:|
|  0 | 21.3728 |                   4.62307 |    14.2737 |                       3.77805 |
|  1 | 13.1856 |                   3.6312  |    12.1893 |                       3.49131 |
|  2 | 11.7816 |                   3.43244 |    11.7408 |                       3.42649 |
|  3 | 11.5007 |                   3.39126 |    12.4095 |                       3.52271 |
|  4 | 10.9336 |                   3.30661 |    11.7891 |                       3.43352 |


```python=12
sns.lineplot(data=history_df[['root_mean_squared_error', 'val_root_mean_squared_error']])
plt.xlabel("epochs")
plt.ylabel("RMSE")
```
![](https://i.imgur.com/4BNVkGn.png)

#### Early stopping

We can rewrite the model creation function as:
```python=
def create_nn(nodes1=100, nodes2=50):
    # Input layer
    inputs = keras.layers.Input(shape=(X_data.shape[1],), name='input')

    # Dense layers
    layers_dense = keras.layers.Dense(nodes1, 'relu')(inputs)
    layers_dense = keras.layers.Dense(nodes2, 'relu')(layers_dense)

    # Output layer
    outputs = keras.layers.Dense(1)(layers_dense)

    return keras.Model(inputs=inputs, outputs=outputs, name="model_small")
```

We can try different network sizes, let's say we arrive at a network with hidden layers that are 10 & 5.
```python=13
model = create_nn(10, 5)
model.summary()
# Model: "model_small"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input (InputLayer)          [(None, 89)]              0         
#                                                                 
#  dense_9 (Dense)             (None, 10)                900       
#                                                                 
#  dense_10 (Dense)            (None, 5)                 55        
#                                                                 
#  dense_11 (Dense)            (None, 1)                 6         
#                                                                 
# =================================================================
# Total params: 961
# Trainable params: 961
# Non-trainable params: 0
# _________________________________________________________________
```

We can then repeat as before:
```python=32
model.compile(optimizer='adam',
              loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError()])
```

To save on training time, we can also implement early stopping, i.e. stop training further when it starts overfitting.
```python=
from tensorflow.keras.callbacks import EarlyStopping

earlystopper = EarlyStopping(monitor="val_loss", patience=10, verbose=1)


history = model.fit(X_train, y_train,
                    batch_size = 32,
                    epochs = 200,
                    validation_data=(X_val, y_val),
                    callbacks=[earlystopper],
                    verbose = 2)
```

Plotting the errors:
```python=12
history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df[['root_mean_squared_error', 'val_root_mean_squared_error']])
plt.xlabel("epochs")
plt.ylabel("RMSE")
```
![](https://i.imgur.com/nzjHNpB.png)

#### Batch normalisation
```python=
def create_batch_norm_nn():
    # Input layer
    inputs = keras.layers.Input(shape=(X_data.shape[1],), name='input')

    # Dense layers
    batchnorm_layer = keras.layers.BatchNormalization()(inputs)
    hidden_layer1 = keras.layers.Dense(100, 'relu')(batchnorm_layer)
    hidden_layer2 = keras.layers.Dense(50, 'relu')(hidden_layer1)

    # Output layer
    outputs = keras.layers.Dense(1)(hidden_layer2)

    # Defining the model and compiling it
    return keras.Model(inputs=inputs, outputs=outputs, name="model_batchnorm")

model = create_batch_norm_nn()
model.compile(loss='mse', optimizer='adam', metrics=[keras.metrics.RootMeanSquaredError()])
model.summary()
# Model: "model_batchnorm"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input (InputLayer)          [(None, 89)]              0         
#                                                                 
#  batch_normalization (BatchN  (None, 89)               356       
#  ormalization)                                                   
#                                                                 
#  dense_12 (Dense)            (None, 100)               9000      
#                                                                 
#  dense_13 (Dense)            (None, 50)                5050      
#                                                                 
#  dense_14 (Dense)            (None, 1)                 51        
#                                                                 
# =================================================================
# Total params: 14,457
# Trainable params: 14,279
# Non-trainable params: 178
# _________________________________________________________________
```

*Note:* The non-trainable parameters are the batch parameters (statistical parameters like mean and standard deviation).

```python=39
history = model.fit(X_train, y_train,
                    batch_size = 32,
                    epochs = 1000,
                    validation_data=(X_val, y_val),
                    callbacks=[earlystopper],
                    verbose = 2)

history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df[['root_mean_squared_error', 'val_root_mean_squared_error']])
plt.xlabel("epochs")
plt.ylabel("RMSE")
```
![](https://i.imgur.com/PNRrbwm.png)

#### Tensorboard
```python=
from tensorflow.keras.callbacks import TensorBoard
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # You can adjust this to add a more meaningful model name
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(X_train, y_train,
                    batch_size = 32,
                    epochs = 200,
                    validation_data=(X_val, y_val),
                    callbacks=[tensorboard_callback],
                    verbose = 2)
```

```python=
%load_ext tensorboard
%tensorboard --logdir logs/fit
```

### Advanced layer types
Please start a new Jupyter notebook.

Load the data (it is available within the keras package)
```python=
from tensorflow import keras
```

```python=
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
```

Plotting CIFAR-10 images:
```python
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.axis('off')
    plt.title(class_names[train_labels[i,0]])
plt.show()
```

The goal of the model will be to predict what is in the image (an airplain, automobile etc.)
The input will be an image
The output will be one of the 10 classes in the data.

For practical purposes (training time) we will limit our data
```python=
train_images = train_images[:n]
train_labels = train_labels[:n]
```


```python=
train_images.shape
train_images.max(), train_images.min()
train_labels.shape
train_labels.min(), train_labels.max()
```

Prepare the data
The training set consists of 50000 images of 32x32 pixels and 3 channels (RGB values). The RGB values are between 0 and 255. For input of neural networks, it is better to have small input values. So we normalize our data between 0 and 1:
```python=
train_images = train_images / 255.0
test_images = test_images / 255.0
```
#### Finding a pretrained model or starting from scratch?
For this particular data set, there are a lot of state-of-the-art pretrained models available (search for GitHub repos). 

If you want to use an existing model, but you have data with, for example, an additional category you want to detect, you can do transfer learning. The key point to note is, now your output dimensions are different (one additonal category).  We can freeze the weights in the hidden layers of the pretrained model, delete the output layer and update it according to your new categories, and then retrain the weights that were not frozto be able to predict all your categories.

For the purpose of this training, we will try to write a model from scratch
```python=
dim = train_images.shape[1] * train_images.shape[2] * train_images.shape[3]
print(dim)
```

Suppose we apply a convolutional layer with 100 kernels of size 3 * 3 * 3 (the last dimension applies to the rgb channels) to our images of 32 * 32 * 3 pixels. How many parameters do we have? Assume, for simplicity, that the kernels do not use bias terms:
We have 100 matrices with 3 * 3 * 3 = 27 values each so that gives 27 * 100 = 2700 weights.


```python=
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model_small")

model.summary()
```

#### Pooling layer
Often in convolutional neural networks, the convolutional layers are intertwined with Pooling layers. As opposed to the convolutional layer, the pooling layer actually alters the dimensions of the image and reduces it by a scaling factor. It is basically decreasing the resolution of your picture. The rationale behind this is that higher layers of the network should focus on higher-level features of the image. By introducing a pooling layer, the subsequent convolutional layer has a broader 'view' on the original image.
```python=
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x) # added MaxPooling layer
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x) # added MaxPooling layer
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(32, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model_small")

model.summary()
```

#### Dropout layer
Helps to prevent overfitting; the dropout layer randomly shuts down some (a specified fraction) of your neurons during training. The part that is dropped is random for each batch (which neurons are dropped is not constant between batches and neither between epochs of training).

```python=
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = keras.layers.Dropout(0.2)(x) # This is new!
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(32, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)

model_dropout = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model")

model_dropout.summary()
```
## Post workshop survey
https://www.surveymonkey.com/r/KD9XFLG

## 📚 Resources
[Link](https://github.com/esciencecenter-digital-skills/2022-09-26-ds-dl-intro/blob/main/files/02-classification-with-keras.ipynb) to notebook yesterday.

- Interesting project for explainable AI (XAI) with links to methods/research in XAI, [DIANNA](https://github.com/dianna-ai/dianna)
- [3D visualization of Convolutional Neural Network](https://web.archive.org/web/20220126212933/https://www.cs.ryerson.ca/~aharley/vis/conv/)
- [The difference between validation data and test data](https://machinelearningmastery.com/difference-test-validation-datasets/)
- [Underfitting and overfitting](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)
- [Unbalanced data](https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758)
- [Unbalanced data in Keras](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
- [Tensorflow Playground, for visualizing neural networks](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.45454&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
- [Available loss functions in keras (documentation)](https://keras.io/api/losses/)
- [Documentation on keras optimizers](https://keras.io/api/optimizers/)
- [Documentation on Keras EarlyStopping](https://keras.io/api/callbacks/early_stopping/)
- [Documentaion on the Keras batch_normalization layer](https://keras.io/api/layers/normalization_layers/batch_normalization/)
- [Transfer learning in Keras](https://keras.io/guides/transfer_learning/)

## Code for copy-pasting
Plotting CIFAR-10 images:
```python
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.axis('off')
    plt.title(class_names[train_labels[i,0]])
plt.show()
```x