*Behavioral Cloning*

**Project Structure**
My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* basic.0.32x32.tpp.w1.bw.augmentedx.t100k.\* - tensorflow model for basic track
* advanced.0.32x32.tpp.w1.bw.augmentedx.t100k.\* - tensorflow model for advanced track
* basic.learned.mp4 - video of learned policy for basic track
* advanced.learned.mp4 - video of learned policy for advanced track
* writeup_report.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing for basic track:
```sh
python drive.py basic.learned.log ./basic.0.32x32.tpp.w1.bw.augmentedx.t100k 20
```
and for advanced track:
```sh
python drive.py advanced.learned.log ./advanced.0.32x32.tpp.w1.bw.augmentedx.t100k 20
```
Here 20 stands for maximum speed. At least on my Mac it seems that the simulator is lagging and I am receiving three identical consequent frames in a row. It might be able to drive at 30 on a faster computer.

There is also a policy trained for both tracks simultaneously which can (almost) drive both tracks at 15 mph:
```sh
python drive.py joint.learned.log ./joint.0.32x32.tpp.w1.bw.augmentedx.t100k 20
```

Just for fun I also added a PI-controller which can drive at 30mph on basic track:
```sh
python drive.py basic.learned.log ./basic.0.32x32.tpp.w1.bw.augmentedx.t100k 30 1.0
```
I also experimented with predicting steering angle at the next step instead of steering decision, so I could feed this target steering angle to PI controller.

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

**Model Architecture and Training Strategy**

***Model architecture***

My model consists of a convolution neural network with 3x3 filter sizes and depths 16-32-64 (model.py function `learn_build_model()`), followed by 4 fully connected layers of sizes 256-128-64-1. The model includes RELU layers to introduce nonlinearity.

Frames are resized to 32x32 pixels without antialiasing and normalized by converting image to black-and-white, subtracting image mean, and dividing by stddev (see model.py function `learn_load_data()`).

***Attempts to reduce overfitting in the model***

The model contains dropout layers in order to reduce overfitting.

L2 regularization of weights of fully connected layers was implemented to combat overfitting.

The model was trained, validated, and tested on different data sets to ensure that the model was not overfitting (model.py function `learn_split_data()`)

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track at 20mph.

***Model parameter tuning***

The model used an adam optimizer (with rate = 0.001). Training time dropout was empirically selected to be 0.9 to improve convergence speed. L2 regularization weight was set to 0.01.

***Appropriate training data***

To compensate for training data noisiness, model target was set to:
```
(current_steering_decision + 0.75*next_steering_decision)/1.75
```

**Model Architecture and Training Strategy**

***Solution Design Approach***

The general network architecture was inspired by NVIDIA's behavioral cloning paper.

The following data augmentation techniques were used:
* Flipping training data across x-dimension
* Simulating various forms of shades on frames (because all training data was collected in "fastest" mode without shades")

As a final out-of-sample testing step I manually reversed the car direction and let the learned policies drive at 20mph in reverse direction (sucessfully).

***Final Model Architecture***

My model consists of a convolution neural network with 3x3 filter sizes and depths 16-32-64, followed by 4 fully connected layers of sizes 256-128-64-1. The model includes RELU layers to introduce nonlinearity. Dropout and L2 regularization of fully-connected weights is applied.

***Creation of the Training Set & Training Process***

Instead of collecting training data manually, I wrote a handcrafted policy using opencv segmentation to find road on the frames, and then trained behavioral cloning on top of automatically collected training data. For exploration I also added Gaussian noise with stddev=0.1 to handcrafted steering policy. You can see example videos of handcrafted policy in "basic.handcrafted.mp4" and "advanced.handcrafted.mp4". As You can see the handcrafted policy barely drives so my behavioral cloning task is actually harder training than from human generated data. Also the handcrafted policy fails when there are shades on the road so the training data was collected in "fastest" mode without any shades. The demonstration videos "basic.learned.mp4" and "advanced.learned.mp4" were collected in "good" mode with shades.

You can try handcrafted policy this way:
```sh
python drive.py handcrafted.log
```

After the collection process, I used 100K data frames (corresponding to about 1.5h of driving), although the model trained decently well also with 10K frames. Validation and testing set each received 10% of **sequential** training data (we are dealing with time-series after all).

The maximum number of training epochs was set to 100, although after ~15th epoch not much changed any more.

The exact commands used to train the models were:
```python
import model
(label_data1, x_data1, z_data1, y_data1, mask_data1) = model.learn_load_data("basic.0", max_count=100000, data_fraction=1.0, discount=0.75, window=1, history_count=1, flip=True)
(label_data2, x_data2, z_data2, y_data2, mask_data2) = model.learn_load_data("advanced.0", max_count=100000, data_fraction=1.0, discount=0.75, window=1, history_count=1, flip=True)

(label_train1, x_train1, z_train1, y_train1, mask_train1, label_validate1, x_validate1, z_validate1, y_validate1, mask_validate1, label_test1, x_test1, z_test1, y_test1, mask_test1) = model.learn_split_data(label_data1, x_data1, z_data1, y_data1, mask_data1)
(label_train2, x_train2, z_train2, y_train2, mask_train2, label_validate2, x_validate2, z_validate2, y_validate2, mask_validate2, label_test2, x_test2, z_test2, y_test2, mask_test2) = model.learn_split_data(label_data2, x_data2, z_data2, y_data2, mask_data2)

model.learn_run_train(x_train1, z_train1, y_train1, mask_train1, x_validate1, z_validate1, y_validate1, mask_validate1, epochs=100, model_name='basic.0.32x32.tpp.w1.bw.augmentedx.t100k')
model.learn_run_train(x_train2, z_train2, y_train2, mask_train2, x_validate2, z_validate2, y_validate2, mask_validate2, epochs=100, model_name='advanced.0.32x32.tpp.w1.bw.augmentedx.t100k')
model.learn_run_train(x_train1+x_train2, z_train1+z_train2, y_train1+y_train2, mask_train1+mask_train2, x_validate1+x_validate2, z_validate1+z_validate2, y_validate1+y_validate2, mask_validate1+mask_validate2, epochs=100, model_name='joint.0.32x32.tpp.w1.bw.augmentedx.t100k')

model.learn_run_test(label_test1, x_test1, z_test1, y_test1, mask_test1, model_name='./basic.0.32x32.tpp.w1.bw.augmentedx.t100k')
model.learn_run_test(label_test2, x_test2, z_test2, y_test2, mask_test2, model_name='./advanced.0.32x32.tpp.w1.bw.augmentedx.t100k')
model.learn_run_test(label_test1+label_test2, x_test1+x_test2, z_test1+z_test2, y_test1+y_test2, mask_test1+mask_test2, model_name='./joint.0.32x32.tpp.w1.bw.augmentedx.t100k')
```
Which could be then tested using:
```sh
python drive.py basic.learned.log ./basic.0.32x32.tpp.w1.bw.augmentedx.t100k 20
python drive.py advanced.learned.log ./advanced.0.32x32.tpp.w1.bw.augmentedx.t100k 20
python drive.py joint.learned.log ./joint.0.32x32.tpp.w1.bw.augmentedx.t100k 20
```
