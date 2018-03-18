import sys
import os
import csv
import random
import numpy as np
import random
import math
from PIL import Image
from sklearn.utils import shuffle
import cv2
import tensorflow as tf
import tensorflow.contrib.layers

### Image normalization routines, using normalize_image_bw() in practice

def normalize_image_simple(image):
    return image/256

def normalize_image_color(image):
    image_mean = np.mean(image)
    image_std = np.std(image-image_mean)
    return (image - image_mean)/image_std

def normalize_image_bw(image):
    image = np.mean(image, 2)
    image_mean = np.mean(image)
    image_std = np.std(image-image_mean)
    return np.expand_dims((image - image_mean)/image_std, 2)

### Image augmentation routines (adding shades of various shapes)

def learn_augment_image_periodic(image):
    y_size = image.shape[0]
    x_size = image.shape[1]
    image = image * random.uniform(0,1)
    y_scale = math.pi / random.uniform(1,y_size)
    x_scale = math.pi / random.uniform(1,x_size)
    def fill_function(y,x,c):
        return 0.5 + 0.25*np.sin(y * y_scale+x * x_scale)
    image_mask = np.fromfunction(function=fill_function, shape=image.shape)
    return image * image_mask

def learn_augment_image_polygons(image):
    y_size = image.shape[0]
    x_size = image.shape[1]
    image = image * random.uniform(0,1)
    image_mask = np.full((y_size, x_size), 1.0)

    polygon1 = [[(x_size*random.random()*0.25,y_size*(0.5+random.random()*0.5)), (x_size*random.random(),y_size*0.5*random.random()), (x_size*(0.5+0.5*random.random()),y_size*(0.5 + random.random()*0.5))]]
    cv2.fillPoly(image_mask, np.int32(polygon1), random.uniform(0,1.0))

    x1 = random.uniform(-1,2) * x_size
    x2 = random.uniform(-1,2) * x_size
    if random.random() > 0.5:
        x3 = min(0,min(x1,x2))
    else:
        x3 = max(x_size, max(x1, x2))

    polygon2 = [[(x1, 0), (x2, y_size), (x3, y_size), (x3, 0)]]
    cv2.fillPoly(image_mask, np.int32(polygon2), random.uniform(0,1.0))

    image = image.astype(np.float64)
    for channel in range(0,image.shape[2]):
        image[:,:,channel] *= image_mask
    return image

def learn_augment_image(image):
    return learn_augment_image_polygons(learn_augment_image_periodic(image))

###  Training data preparation routines

def learn_load_data(log_dir, max_count=None, data_fraction = 0.1, flip=False, history_count=1, window=1, discount=0.99, predict_step = None, active_counter_threshold = 100, image_dimension = 32):
    """ Loading training data

    Args:
        log_dir - folder where to collect training data from
        max_count - maximum number of samples to load
        data_fraction - data sampling fraction
        flip - whether to randomly flip samples along x-axis
        history_count - how many consequent frames to use as prediction input
        window - window across which to average future steering decisions (in addition to current steering decision)
        discount - future steering decisions have weight proportional to discount**(window_idx - current_idx)
        predict_step - my failed (so far) attempt at Q-learning (disabled when None)
        active_counter_threshold - threshold filtering out samples where the car was probably stuck
        image_dimension - resizing frames to (image_dimension)x(image_dimension)

    Returns:
        Tuple (labels_train, x_train, z_train, y_train, mask_train), where
            labels_train - list of labels of training samples (image pathes)
            x_train - list sample input frames
            z_train - list of side inputs to prediction (current speed, current steering anglem, etc)
            y_train - list of prediction targets: weighted average of current steering angle and discounted future steering angles (discounted)
            mask_train - list of one-hot masks need for Q-learning, equal to [1] when predict_step=None 
    """

    episodes = []
    for log_file in filter(lambda s:s.endswith(".log"), os.listdir(log_dir)):
        print(log_dir + "/" + log_file)
        lines = csv.reader(open(log_dir + "/" + log_file))
        try:
            next(lines)
        except StopIteration:
            continue

        episode = []
        last_speed = None

        for line in lines:
            [current_timestamp, current_speed, current_steering_angle, current_throttle, active_counter, stuck_counter, recovery_counter, slowdown_counter, steering_decision, throttle_decision] = line
            active_counter = int(active_counter)
            slowdown_counter = int(slowdown_counter)
            recovery_counter = float(recovery_counter)
            if active_counter < active_counter_threshold or recovery_counter > 0:# or slowdown_counter > 0
                if len(episode) > 0:
                    episodes.append(episode)
                episode = []
                last_speed = None
            else:
                current_label = image_file = log_dir + "/" + log_file.replace(".log", ".img") + "/" + current_timestamp + ".png"
                current_speed = float(current_speed)
                current_steering_angle = float(current_steering_angle)
                current_throttle = float(current_throttle)
                steering_decision = float(steering_decision)
                throttle_decision = float(throttle_decision)
                if current_speed == last_speed:
                    continue
                else:
                    last_speed = current_speed
                episode.append([current_label, current_speed, current_steering_angle, current_throttle, steering_decision, throttle_decision])
        if len(episode) > 0:
            episodes.append(episode)

    labels_train = []
    x_train = []
    z_train = []    
    y_train = []
    mask_train = []

    for episode in episodes:
        print("EPISODE LENGTH:",len(episode))
        max_speed = max(map(lambda i:i[1], episode))
        print("EPISODE MAX SPEED:", max_speed)

        history_cache = {}
        return_cache = {}

        for current_idx in range(0, len(episode)):
            if random.random() > data_fraction: continue

            data_count = len(labels_train)

            if max_count is not None and data_count >= max_count:
                break

            if data_count % 100 == 0:
                print("DATA COUNT:", data_count)

            [current_label, current_speed, current_steering_angle, current_throttle, current_steering_decision, current_throttle_decision] = episode[current_idx]
            
            def load_history(history_idx):
                history_label = episode[history_idx][0]
                if not history_label in history_cache:
                    history_cache[history_label] = np.asarray(Image.open(history_label).resize((image_dimension,image_dimension)))
                return history_cache[history_label]

            current_x = np.dstack([load_history(max(0,current_idx + history_offset + 1)) for history_offset in range(-history_count,0)])
            
            current_target = 0
            current_target_window = min(window, len(episode) - current_idx - 1)
            if current_target_window <= window/2:
                continue
            #print("CURRENT", current_idx, episode[current_idx])
            for window_episode_offset in range(1, current_target_window+1):
                [window_label, window_speed, window_steering_angle, window_throttle, window_steering_decision, window_throttle_decision] = episode[current_idx + window_episode_offset]
                if not window_label in return_cache:
                    #window_image = np.asarray(Image.open(window_label).resize((image_dimension,image_dimension)))
                    #return_cache[window_label] = window_steering_angle/25
                    return_cache[window_label] = discount*window_steering_decision

                window_episode_return = return_cache[window_label]

                discount_weight = discount**(window_episode_offset-1) * (1-discount)
                current_target += window_episode_return*discount_weight
                #print("WINDOW", window_label, window_episode_offset, window_episode_return, discount_weight)
            #print("RETURN", current_idx, episode[current_idx], current_target)
            current_target /= (1-discount**current_target_window)
            current_target = (current_steering_decision + discount*current_target)/(1+discount)
            #print("RETURN_FULL", current_idx, episode[current_idx], current_target)

            if flip and random.random() > 0.5:
                current_x = np.asarray(current_x)[:,::-1,:]
                current_steering_angle = -current_steering_angle
                current_steering_decision = -current_steering_decision
                current_target = -current_target

            labels_train.append(current_label)
            x_train.append(current_x)
            z_train.append(np.asarray([current_speed/30.0, current_steering_angle/25]))# + ([current_steering_decision] if not predict_step else [])))
            y_train.append(np.asarray([current_target]))

            if predict_step:
                steering_min_idx = int(math.floor(-1/predict_step))
                steering_max_idx = int(math.ceil(1/predict_step))
                steering_idx = round(min(1,max(-1, current_steering_decision))/predict_step) - steering_min_idx
                steering_one_hot = [1 if i == steering_idx else 0  for i in range(0,steering_max_idx - steering_min_idx + 1)]
                assert(sum(steering_one_hot)==1)
                mask_train.append(np.asarray(steering_one_hot))
            else:
                mask_train.append(np.asarray([1.0]))
    return (labels_train, x_train, z_train, y_train, mask_train)

### Routines for building model and setting up evaluation and training operation

def learn_split_data(label_data, x_data, z_data, y_data, mask_data, train_fraction=0.8, validate_fraction=0.1):
    ### Splits data into training, validation, and test

    train_len = int(len(x_data)*train_fraction)
    validate_len = int(len(x_data)*(train_fraction+validate_fraction))
    label_train = label_data[0:train_len]
    x_train = x_data[0:train_len]
    z_train = z_data[0:train_len]
    y_train = y_data[0:train_len]
    mask_train = mask_data[0:train_len]
    label_validate = label_data[train_len:validate_len]
    x_validate = x_data[train_len:validate_len]
    z_validate = z_data[train_len:validate_len]
    y_validate = y_data[train_len:validate_len]
    mask_validate = mask_data[train_len:validate_len]
    label_test = label_data[validate_len:]
    x_test = x_data[validate_len:]
    z_test = z_data[validate_len:]
    y_test = y_data[validate_len:]
    mask_test = mask_data[validate_len:]
    return (label_train, x_train, z_train, y_train, mask_train, label_validate, x_validate, z_validate, y_validate, mask_validate, label_test, x_test, z_test, y_test, mask_test)

def learn_build_model(x_placeholder, z_placeholder, dropout_placeholder, mu = 0, sigma = 0.1, z_count=1, output_count=1, image_dimension=32, image_channels=1):
    conv1_f = 16
    conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, image_channels, conv1_f), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(conv1_f))
    conv1   = tf.nn.conv2d(x_placeholder, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    conv1   = tf.nn.relu(conv1)
    conv1   = tf.nn.dropout(conv1, dropout_placeholder)

    #conv2_f = 32
    #conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, conv1_f, conv2_f), mean = mu, stddev = sigma))
    #conv2_b = tf.Variable(tf.zeros(conv2_f))
    #conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
    #conv2   = tf.nn.relu(conv2)
    #conv2   = tf.nn.dropout(conv2, dropout_placeholder)

    #conv3_f = 32
    #conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, conv2_f, conv3_f), mean = mu, stddev = sigma))
    #conv3_b = tf.Variable(tf.zeros(conv3_f))
    #conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
    #conv3   = tf.nn.relu(conv3)
    #conv3   = tf.nn.dropout(conv3, dropout_placeholder)

    conv3_f = conv1_f
    conv3 = conv1

    conv4_f = 32
    conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, conv3_f, conv4_f), mean = mu, stddev = sigma))
    conv4_b = tf.Variable(tf.zeros(conv4_f))
    conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME') + conv4_b
    conv4   = tf.nn.relu(conv4)
    conv4   = tf.nn.dropout(conv4, dropout_placeholder)
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    conv5_f = 64
    conv5_W = tf.Variable(tf.truncated_normal(shape=(3, 3, conv4_f, conv5_f), mean = mu, stddev = sigma))
    conv5_b = tf.Variable(tf.zeros(conv5_f))
    conv5   = tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='SAME') + conv5_b
    conv5   = tf.nn.relu(conv5)
    conv5   = tf.nn.dropout(conv5, dropout_placeholder)
    conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    fc0   = tf.contrib.layers.flatten(conv5)
        
    fc1_f = 256
    fc1_W = tf.Variable(tf.truncated_normal(shape=(int(image_dimension/4)**2*conv5_f, fc1_f), mean = mu, stddev = sigma))
    fc1_Wz = tf.Variable(tf.truncated_normal(shape=(z_count, fc1_f), mean = mu, stddev = sigma))    
    fc1_b = tf.Variable(tf.zeros(fc1_f))
    fc1   = tf.matmul(fc0, fc1_W) + tf.matmul(z_placeholder, fc1_Wz) + fc1_b
    fc1   = tf.nn.relu(fc1)
    fc1   = tf.nn.dropout(fc1, dropout_placeholder)

    fc2_f = 128
    fc2_W = tf.Variable(tf.truncated_normal(shape=(fc1_f, fc2_f), mean = mu, stddev = sigma))
    fc2_Wz = tf.Variable(tf.truncated_normal(shape=(z_count, fc2_f), mean = mu, stddev = sigma))    
    fc2_b = tf.Variable(tf.zeros(fc2_f))
    fc2   = tf.matmul(fc1, fc2_W) + tf.matmul(z_placeholder, fc2_Wz) + fc2_b
    fc2    = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout_placeholder)

    fc3_f = 64
    fc3_W = tf.Variable(tf.truncated_normal(shape=(fc2_f, fc3_f), mean = mu, stddev = sigma))
    fc3_Wz = tf.Variable(tf.truncated_normal(shape=(z_count, fc3_f), mean = mu, stddev = sigma))    
    fc3_b = tf.Variable(tf.zeros(fc3_f))
    fc3   = tf.matmul(fc2, fc3_W) + tf.matmul(z_placeholder, fc3_Wz) + fc3_b
    fc3    = tf.nn.relu(fc3)
    fc3 = tf.nn.dropout(fc3, dropout_placeholder)
        
    fc4_W  = tf.Variable(tf.truncated_normal(shape=(fc3_f, output_count), mean = mu, stddev = sigma))
    fc4_Wz = tf.Variable(tf.truncated_normal(shape=(z_count, output_count), mean = mu, stddev = sigma))
    fc4_b  = tf.Variable(tf.zeros(output_count))
    output_result = tf.matmul(fc3, fc4_W) + tf.matmul(z_placeholder, fc4_Wz) + fc4_b
    
    return [output_result, [fc1_W, fc1_Wz, fc2_W, fc2_Wz, fc3_W, fc3_Wz, fc4_W, fc4_Wz]]

def learn_build_model_operations(regularization=0.01, mu = 0, sigma = 0.1, rate = 0.001, z_count=1, output_count = 1, image_dimension = 32, image_channels=1):
    x_placeholder = tf.placeholder(tf.float32, (None, image_dimension, image_dimension, image_channels))
    z_placeholder = tf.placeholder(tf.float32, (None, z_count))
    y_placeholder = tf.placeholder(tf.float32, (None, 1))
    mask_placeholder = tf.placeholder(tf.float32, (None, output_count))
    dropout_placeholder = tf.placeholder_with_default(1.0, None)
    [output_result, model_weights] = learn_build_model(x_placeholder, z_placeholder, dropout_placeholder, mu=mu, sigma=sigma, z_count=z_count, output_count=output_count, image_dimension=image_dimension, image_channels=image_channels)
    regularizer = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(regularization), model_weights)

    loss_operation = tf.reduce_mean(tf.squared_difference(tf.reduce_sum(tf.multiply(output_result, mask_placeholder),1,keep_dims=True), y_placeholder)) + regularizer
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    evaluation_result = tf.reduce_mean(tf.squared_difference(tf.reduce_sum(tf.multiply(output_result, mask_placeholder),1,keep_dims=True), y_placeholder))
    training_operation = optimizer.minimize(loss_operation)

    return (x_placeholder, z_placeholder, y_placeholder, mask_placeholder, dropout_placeholder, output_result, evaluation_result, training_operation)

### Model evaluation routines

def learn_evaluate_model(x_data, z_data, y_data, mask_data, x_placeholder, z_placeholder, y_placeholder, mask_placeholder, evaluation_result, batch_size, sess):
    num_examples = len(x_data)
    total_loss = 0
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_z, batch_y, batch_mask = x_data[offset:offset+batch_size], z_data[offset:offset+batch_size], y_data[offset:offset+batch_size], mask_data[offset:offset+batch_size]
        batch_x = list(map(normalize_image_bw, batch_x))
        loss = sess.run(evaluation_result, feed_dict={x_placeholder: batch_x, z_placeholder: batch_z, y_placeholder: batch_y, mask_placeholder: batch_mask})
        total_loss += (loss * len(batch_x))
    return math.sqrt(total_loss / num_examples)

### Model training/testing routines

def learn_train_model(x_train, z_train, y_train, mask_train, x_validate, z_validate, y_validate, mask_validate, x_placeholder, z_placeholder, y_placeholder, mask_placeholder, dropout_placeholder, training_operation, evaluation_result, epochs, batch_size, sess, saver, model_name):
    output_shape = mask_train[0].shape
    sqr_sum = np.zeros(output_shape)
    mean_sum = np.zeros(output_shape)
    train_count = np.zeros(output_shape)
    for train_idx in range(0, len(x_train)):
        mask = mask_train[train_idx]
        y_masked = mask * y_train[train_idx]
        sqr_sum += y_masked**2
        mean_sum += y_masked
        train_count += mask
    train_mean = mean_sum / train_count
    train_stddev = np.sqrt(sqr_sum / train_count - (train_mean)**2)
    print("Count:", list(train_count))
    print("Mean:", list(train_mean))
    print("Stddev:", list(train_stddev))

    print("Training...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))        
        x_train, z_train, y_train, mask_train = shuffle(x_train, z_train, y_train, mask_train)
        for offset in range(0, len(x_train), batch_size):
            end = offset + batch_size
            batch_x, batch_z, batch_y, batch_mask = x_train[offset:offset+batch_size], z_train[offset:offset+batch_size], y_train[offset:offset+batch_size], mask_train[offset:offset+batch_size]
            batch_x = list(map(normalize_image_bw, map(learn_augment_image, batch_x)))
            sess.run(training_operation, feed_dict={x_placeholder: batch_x, z_placeholder: batch_z, y_placeholder: batch_y, mask_placeholder: batch_mask, dropout_placeholder:0.9})

        training_loss = learn_evaluate_model(x_train, z_train, y_train, mask_train, x_placeholder, z_placeholder, y_placeholder, mask_placeholder, evaluation_result, batch_size, sess)
        print("Training Loss = {:.3f}".format(training_loss))
        validation_loss = learn_evaluate_model(x_validate, z_validate, y_validate, mask_validate, x_placeholder, z_placeholder, y_placeholder, mask_placeholder, evaluation_result, batch_size, sess)
        print("Validation Loss = {:.3f}".format(validation_loss))        

        print("Saving model:", model_name)
        saver.save(sess, model_name)

def learn_test_model(labels_data, x_data, z_data, y_data, mask_data, x_placeholder, z_placeholder, output_result, batch_size, sess):
    output_shape = mask_data[0].shape
    total_count = np.zeros(output_shape)
    total_loss = np.zeros(output_shape)

    num_examples = len(x_data)
    for offset in range(0, num_examples, batch_size):
        batch_labels, batch_x, batch_z, batch_y, batch_mask = labels_data[offset:offset+batch_size], x_data[offset:offset+batch_size], z_data[offset:offset+batch_size], y_data[offset:offset+batch_size], mask_data[offset:offset+batch_size]
        batch_x = list(map(normalize_image_bw, batch_x))
        batch_output_result = sess.run(output_result, feed_dict={x_placeholder: batch_x, z_placeholder: batch_z})
        for idx in range(0, len(batch_labels)):
            total_loss += (batch_y[idx]*batch_mask[idx] - batch_output_result[idx]*batch_mask[idx])**2
            total_count += batch_mask[idx]
            print("Test ", batch_labels[idx], batch_z[idx], batch_y[idx], list(batch_mask[idx]), np.argmax(batch_mask[idx]), list(batch_output_result[idx]), np.argmax(batch_output_result[idx]))
    print("Total loss per mask:", list(np.sqrt(total_loss/total_count)))
    print("Total loss:", np.sqrt(np.sum(total_loss)/np.sum(total_count)))

### Wrapper rountines for model training / testing

def learn_run_train(x_train, z_train, y_train, mask_train, x_validate, z_validate, y_validate, mask_validate, image_channels=1, epochs = 10, batch_size = 32, model_name = "clone"):
    image_dimension = x_train[0].shape[0]
    z_count = z_train[0].shape[0]
    output_count = mask_train[0].shape[0]

    g = tf.Graph()
    with g.as_default():
        (x_placeholder, z_placeholder, y_placeholder, mask_placeholder, dropout_placeholder, output_result, evaluation_result, training_operation) = learn_build_model_operations(image_dimension = image_dimension, image_channels=image_channels, z_count = z_count, output_count=output_count)

        saver = tf.train.Saver()
        #saver = tf.train.import_meta_graph(model_name + '.meta')

        tf.add_to_collection('x_placeholder', x_placeholder)
        tf.add_to_collection('z_placeholder', z_placeholder)        
        tf.add_to_collection('y_placeholder', y_placeholder)
        tf.add_to_collection('mask_placeholder', mask_placeholder)
        tf.add_to_collection('dropout_placeholder', dropout_placeholder)
        tf.add_to_collection('output_result', output_result)
        tf.add_to_collection('evaluation_result', evaluation_result)
        tf.add_to_collection('training_operation', training_operation)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allocator_type = 'BFC'
        tf_config.gpu_options.allow_growth = True    
        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())#, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))
            learn_train_model(x_train, z_train, y_train, mask_train, x_validate, z_validate, y_validate, mask_validate, x_placeholder, z_placeholder, y_placeholder, mask_placeholder, dropout_placeholder, training_operation, evaluation_result, epochs=epochs, batch_size=batch_size, sess=sess, saver=saver, model_name=model_name)

def learn_run_test(labels_test, x_test, z_test, y_test, mask_test, image_channels=1, batch_size = 32, model_name = "clone"):
    image_dimension = x_test[0].shape[0]
    g = tf.Graph()
    with g.as_default():    
        saver = tf.train.import_meta_graph(model_name + '.meta')

        x_placeholder = g.get_collection("x_placeholder")[0]
        z_placeholder = g.get_collection("z_placeholder")[0]
        output_result = g.get_collection("output_result")[0]

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allocator_type = 'BFC'
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            saver.restore(sess, model_name)
            learn_test_model(labels_test, x_test, z_test, y_test, mask_test, x_placeholder, z_placeholder, output_result, batch_size=batch_size, sess=sess)

