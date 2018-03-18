import sys
import os
import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
import datetime
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import random
import time
import os
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import tensorflow as tf

### This part is a handcrafted policy based on opencv image segmentation that I used to collect training data, instead of driving the car myself ###

def segmentation_prepare_mask(x_size,y_size):
    mask = np.zeros((y_size,x_size),np.uint8)
    mask[:,:] = cv2.GC_PR_BGD
    cv2.fillPoly(mask, np.int32([[(x_size*0.2,y_size*0.8), (x_size*0.5,y_size*0.6), (x_size*0.8,y_size*0.8)]]), cv2.GC_FGD)    
    return mask

def segmentation_image(image, segmentation_mask):
    mask = np.copy(segmentation_mask)
    bgdModel = np.zeros((1,65),np.float64) 
    fgdModel = np.zeros((1,65),np.float64)
    _ = cv2.grabCut(image,mask,None,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8') 
    return mask2

def segmentation_analyze(image_mask, upper, lower):
    y_size = image_mask.shape[0]
    x_size = image_mask.shape[1]
    x_middle = int(x_size/2)
    y_upper = int(y_size * upper)
    y_lower = int(y_size * lower)
    return ((int(np.sum(image_mask[y_upper:y_lower,0:x_middle]))-int(np.sum(image_mask[y_upper:y_lower,x_middle:x_size])))/float(x_size*y_size))

segmentation_mask = None
def policy_steering_segmentation(image_array, current_speed, current_steering_angle, current_throttle):
    global segmentation_mask
    if segmentation_mask is None:
        segmentation_mask = segmentation_prepare_mask(image_array.shape[1], image_array.shape[0])
    image_mask = segmentation_image(image_array, segmentation_mask)
    range_buckets = 10
    range_start = 0.0
    range_step = 1/range_buckets
    range_weights = [0,0,0,0,0,10,20,20,10,0]
    steering_decision = 0
    for i in range(0,range_buckets):
        range_center = range_start + i*range_step - range_step/2
        range_weight = range_weights[i]
        range_centrality = segmentation_analyze(image_mask, range_center - range_step/2, range_center + range_step/2)
        range_decision = -range_centrality * range_weight
        steering_decision += range_decision 
    return steering_decision

def policy_throttle_adaptive(current_speed, current_steering_angle, current_throttle, steering_decision, speed_factor=1.0):
    if math.fabs(steering_decision) > 0.5:
      speed_factor *= 1.5
    elif math.fabs(steering_decision) > 0.25:
      speed_factor *= 1.25
    else:
      speed_factor *= 1.0
    if current_speed*speed_factor < 10:
        throttle_decision = 1.0
    elif current_speed*speed_factor < 15:
        throttle_decision = 0.5
    elif current_speed*speed_factor < 20:
        throttle_decision = 0
    elif current_speed*speed_factor < 25:
        throttle_decision = -0.125
    else:
        throttle_decision = -0.25
    return throttle_decision
    
def policy_crafted(image_array, current_speed, current_steering_angle, current_throttle):
    steering_decision = policy_steering_segmentation(image_array, current_speed, current_steering_angle, current_throttle)
    throttle_decision = policy_throttle_adaptive(current_speed, current_steering_angle, current_throttle, steering_decision)
    return (steering_decision, throttle_decision)

### This policy is learned by behavioral cloning of training data ###

def policy_normalize_image(image):
    image = np.mean(image, 2)
    image_mean = np.mean(image)
    image_std = np.std(image-image_mean)
    return np.expand_dims((image - image_mean)/image_std, 2)

learned_fun = None
def policy_steering_learned(image_array, current_speed, current_steering_angle, current_throttle, model_name):
    global learned_fun
    if learned_fun is None:
        saver = tf.train.import_meta_graph(model_name + '.meta')

        x_placeholder = tf.get_collection("x_placeholder")[0]
        z_placeholder = tf.get_collection("z_placeholder")[0]
        output_result = tf.get_collection("output_result")[0]

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allocator_type = 'BFC'
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)
        saver.restore(sess, model_name)

        learned_fun = lambda image_array: sess.run(output_result, feed_dict={x_placeholder: [policy_normalize_image(image_array)], z_placeholder: [np.asarray([current_speed/30.0, current_steering_angle/25])]})[0][0]
    return learned_fun(image_array)

def policy_throttle_simple(current_speed, current_steering_angle, current_throttle, steering_decision, max_speed):
    if current_speed < max_speed:
        throttle_decision = 1
    else:
        throttle_decision = -0.5
    return throttle_decision

def policy_learned(image_array, current_speed, current_steering_angle, current_throttle, max_speed, model_name):
    steering_decision = policy_steering_learned(image_array, current_speed, current_steering_angle, current_throttle, model_name=model_name)
    throttle_decision = policy_throttle_simple(current_speed, current_steering_angle, current_throttle, steering_decision, max_speed = max_speed)
    return (steering_decision, throttle_decision)

### I also implemented a simple PI-controller on top of learned policy, just for fun ###
learned_steering_error_integral = 0
def policy_learned_pid(image_array, current_speed, current_steering_angle, current_throttle, max_speed, model_name):
    global learned_steeing_error_integral
    global pid_coeff
    steering_error = pid_coeff*policy_steering_learned(image_array, current_speed, current_steering_angle, current_throttle, model_name=model_name) - current_steering_angle/25
    steeing_error_integral = 0.1 * steering_error + 0.9 * learned_steering_error_integral
    steering_decision = 0.8*steering_error + 0.1*steeing_error_integral
    throttle_decision = policy_throttle_simple(current_speed, current_steering_angle, current_throttle, steering_decision, max_speed = max_speed)
    return (steering_decision, throttle_decision)

### Policy executor makes decisions using: (steering_decision, throttle_decision) = policy_fun(image_array, current_speed, current_steering_angle, current_throttle)
### Besides running the policy is can detect being stuck and has a simple recovery mechanism (used for collecting training data automatically).
### It also has some mechanisms for adding exploration noise to steering_decision & throttle_decision and can simulate car slowing down and speeding up (disabled by default)
### Finally, it also saves logs of driving sessions into log_path

sio = socketio.Server()
app = Flask(__name__)

last_log_file = None
last_img_dir = None
last_timestamp = None

@sio.on('telemetry')
def telemetry(sid, data):
    global policy_fun
    global last_log_file
    global last_img_dir
    global last_timestamp
    global last_throttle_decision
    global active_counter
    global stuck_counter
    global recovery_counter 
    global slowdown_counter

    if data == None:
        last_timestamp = None
        raise Exception("crafted override")

    start_time = time.time()
    # The current steering angle of the car
    current_steering_angle = float(data["steering_angle"])
    # The current throttle of the car
    current_throttle = float(data["throttle"])
    # The current speed of the car
    current_speed = float(data["speed"])
    # The current image from the center camera of the car
    imgString = data["image"]
  
    current_timestamp = datetime.datetime.now()
    print("Last timestmap:", last_timestamp)
    print("Current timestmap:", current_timestamp)
    print("Current speed:", current_speed)
    print("Current steering angle:", current_steering_angle)
    print("Current throttle:", current_throttle)

    if last_timestamp == None or (current_timestamp - last_timestamp).total_seconds() > 1.0:
      last_timestamp = None
      last_throttle_decision = 0
      active_counter = 0
      stuck_counter = 0
      recovery_counter = 0
      slowdown_counter = 0

      print("====================" + str(current_timestamp) + "====================")
      last_log_file = open(log_path + "/" + str(current_timestamp) + ".log", "w")
      last_log_file.write("current_timestamp, current_speed, current_steering_angle, current_throttle, active_counter, stuck_counter, recovery_counter, slowdown_counter, steering_decision, throttle_decision\n")
      last_img_dir = log_path + "/" + str(current_timestamp) + ".img"
      os.makedirs(last_img_dir,exist_ok=True)

    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(Image.fromarray(np.asarray(image)).resize((32,32)))
    (steering_decision, throttle_decision) = policy_fun(image_array, current_speed, current_steering_angle, current_throttle)

    if recovery_counter == 0:
      if last_throttle_decision > 0 and current_speed < 1:
        stuck_counter = min(50, stuck_counter+1)
      else:
        stuck_counter = 0
      if stuck_counter == 50:
        recovery_counter = 100
        active_counter = 0
        stuck_counter = 0
        slowdown_counter = 0

    if recovery_counter > 0:
      recovery_counter = max(0, recovery_counter - min(1,current_speed))
      if current_speed >= 5:
        recovery_counter = 0
    else:
      if current_speed >= 1:
        active_counter += 1
      if slowdown_counter > 0:
        if current_speed >= 1:
            slowdown_counter += 1
        else:
            slowdown_counter = 0
      else:
        #if random.random() > 0.999:
        #  slowdown_counter = 1
        pass

    print("Active counter:", active_counter)
    print("Stuck counter:", stuck_counter)
    print("Recovery counter:", recovery_counter)
    print("Slowdown counter:", slowdown_counter)

    if recovery_counter == 0:
      steering_decision_noise = random.uniform(-exploration_noise,exploration_noise)
      steering_decision += steering_decision_noise 
      print("Steering decision noise:", steering_decision_noise)
      #throttle_decision_noise = random.gauss(0,0.25)
      #throttle_decision += throttle_decision_noise 
      #print("Throttle decision noise:", throttle_decision_noise)
      #if slowdown_counter > 0:
      #  throttle_decision = random.uniform(-1,0)
      pass
    else:
      steering_decision = -steering_decision
      throttle_decision = -1

    print("Steering decision:", steering_decision)
    print("Throttle decision:", throttle_decision)

    last_log_file.write("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" % (current_timestamp, current_speed, current_steering_angle, current_throttle, active_counter, stuck_counter, recovery_counter, slowdown_counter, steering_decision, throttle_decision))
    last_log_file.flush()

    with open(last_img_dir + "/" + str(current_timestamp) + ".png", "wb") as img_file:
        img_file.write(base64.b64decode(imgString))

    last_timestamp = current_timestamp
    last_throttle_decision = throttle_decision

    send_control(steering_decision, throttle_decision)
    end_time = time.time()
    print("Runtime:", end_time-start_time)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

if __name__ == '__main__':
    log_path = sys.argv[1]
    os.makedirs(log_path,exist_ok=True)
    exploration_noise = float(os.environ.get("EXPLORATION_NOISE", "0.0"))
    if len(sys.argv) >= 4:
        model_name = sys.argv[2]
        max_speed = float(sys.argv[3])
        if len(sys.argv) >= 5:
            pid_coeff = float(sys.argv[4])
            policy_fun = lambda image_array, current_speed, current_steering_angle, current_throttle: policy_learned_pid(image_array, current_speed, current_steering_angle, current_throttle, max_speed=max_speed, model_name=model_name)
        else:
            policy_fun = lambda image_array, current_speed, current_steering_angle, current_throttle: policy_learned(image_array, current_speed, current_steering_angle, current_throttle, max_speed=max_speed, model_name=model_name)
    else:
        policy_fun = lambda image_array, current_speed, current_steering_angle, current_throttle: policy_crafted(image_array, current_speed, current_steering_angle, current_throttle)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
