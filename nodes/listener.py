#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
from std_msgs.msg import Int16MultiArray
import tensorflow as tf
import numpy as np
import random
from collections import deque
import os.path
import roslib
from sensor_msgs.msg import Image
import sys, time
import pickle

####### description
#### http://discourse.ros.org/t/robotic-humanoid-hand/188
########


NUM_STATES = 264+264+1024+1024+1024+1024  #264 degree angle_goal, 264 possible degrees the joint could move, 1024 force values, two times
NUM_ACTIONS = 9  #3^2=9      ,one stop-state, one different speed left, one diff.speed right, two servos
STATE_FRAMES = 4
GAMMA = 0.5
RESIZED_DATA_X = 68
RESIZED_DATA_Y = 68   # = NUM_STATES

force_reward_max = 15  #where should the max/middle point be, we get force values from 0.0 - 1023.0 (float),
force_reward_length = 10  #how long/big the area around max
force_max_value = 1024     #how much force values possible
degree_goal = 105		#to which angle should it go, get reward  #100 is the middle position
force_1_goal = 15
force_2_goal = 15
degree_possible_max = 264  #how many degrees the angle can go max
current_degree = 0     #will be filled periodically from  callback-function
current_force_1 = 0    #will be filled periodically from  callback-function
current_force_2 = 0    #will be filled periodically from  callback-function

#lists we use for build reward list
angle = []
f1 = []
f2 = []
states = []

#variables for bad-mapping approach, s1=servo1 , s2=servo2,.... 
s1_stop = 377
s1_fwd_1 = 382
s1_bwd_1 = 372
#.... normaly there are many more fwd or bwd speeds, but i dont know how to map so many mathematically
s2_stop = 399
s2_fwd_1 = 406
s2_bwd_1 = 394
sx0 = 1050  #do nothing value for not-used servos

observations = deque()


MINI_BATCH_SIZE = 100 
probability_of_random_action = 1



# we get the current state, that means, current degree and current forces. 
# We build a list, like the states-list, so we can compute reward
#
#
#
#    degree      force1      force2
#           |
#           |         /\         /\
# ----------|--------/  \-------/  \---------
#
#   degree_goal    force_1_goal   force_2_goal
#    
# 68x68 = 4624 => 264_degree *2 + 4*1024

def get_current_state():
	global current_degree
	global current_force_1
	global current_force_2
	global degree_goal
	a1 = [(x==current_degree) for x in range(degree_possible_max)]
	a1 = np.array(a1) * 255
	a2 = [(x==current_force_1) for x in range(force_max_value)]
	a2 = np.array(a2) * 255
	a3 = [(x==current_force_2) for x in range(force_max_value)]
	a3 = np.array(a3) * 255        

	b1 = np.linspace(0, 255.0, num=5)
	b2 = np.linspace(255.0, 0, num=5)

	b3 = np.linspace(0,0, num=(degree_goal - 5))
	b4 = np.linspace(0,0, num=(degree_possible_max - (degree_goal + 5)))
	b5 = np.linspace(0,0, num=(force_1_goal - 5))
	b6 = np.linspace(0,0, num=(force_max_value - (force_1_goal + 5)))


	d = []
	d.extend(a1)
	d.extend(a2)
	d.extend(a3)
	d.extend(b3)
	d.extend(b1)
	d.extend(b2)
	d.extend(b4)
	d.extend(b5)
	d.extend(b1)
	d.extend(b2)
	d.extend(b6)
	d.extend(b5)
	d.extend(b1)
	d.extend(b2)
	d.extend(b6)
	#rospy.loginfo("get_current_state: len_4624 state >%d<", len(d))
	return d

#the reward for reaching the degree_goal and force_1/2_goal
#
def get_reward(state):
	s = np.asarray(state)
	s = s.reshape(2, 2312)

	s1 = (s[0] / 255) * s[1]
	s2 = sum(s1)

	return s2

#we publish an image of the state, to look what the network is seeing
#
def publish_state_image(state_from_env1, current_state_image_pub):
	current_state_image_msg = Image()
        current_state_image_msg.encoding = "mono8"
        current_state_image_msg.header.stamp = rospy.Time.now()
        current_state_image_msg.height = 68
        current_state_image_msg.width = 34
        current_state_image_msg.step = 68
        x = np.reshape(state_from_env1, (2,2312))
        idx_x = np.argwhere(x[0] == np.amax(x[0]))
        lx = idx_x.flatten().tolist()
        x[1][lx[0]] = 255
        x[1][lx[1]] = 255
        x[1][lx[2]] = 255
        y = x[1].tolist()
        current_state_image_msg.data = y
	current_state_image_pub.publish(current_state_image_msg)


# callback which delivers us periodically the adc values of the force sensors
# adc values are floats from 0.0 to 5.0.  we convert them to int from 0.0-1023.0  (float)
def adc_callback(data):
    #rospy.loginfo(rospy.get_caller_id() + "adc heard %s", data.data)
    #rospy.loginfo("adc-val0: %f", (1023/5.0)*data.data[0])
    #rospy.loginfo("adc-val1: %f", (1023/5.0)*data.data[1])
    global current_force_1
    global current_force_2
    current_force_1 = int((1023/5.0)*data.data[0])
    current_force_2 = int((1023/5.0)*data.data[1])
    
#callback which delivers us periodically the degree, from 0.0-200.0 degree (float)
def degree_callback(data):
    #rospy.loginfo(rospy.get_caller_id() + "degree heard %f", data.data)
    # 132, because 0 degree is the middle position
    global current_degree
    current_degree = int(data.data + 132)

def probability_callback(data):
    rospy.loginfo("probability heard %f", data.data)
    global probability_of_random_action
    probability_of_random_action = data.data

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#the main thread/program
#it runs in a loop, todo something and learn to reach the angle_goal (in degree)
def listener():

    rospy.init_node('listener', anonymous=True)

    session = tf.Session()

    state = tf.placeholder("float", [None, NUM_STATES])
    action = tf.placeholder("float", [None, NUM_ACTIONS])
    target = tf.placeholder("float", [None])

    with tf.name_scope("conv1") as conv1:
	conv_weights_1 = weight_variable([8,8,4,32], "conv1_weights")
    	conv_biases_1 = bias_variable([32], "conv1_biases")
	cw1_hist = tf.histogram_summary("conv1/weights", conv_weights_1)
	cb1_hist = tf.histogram_summary("conv1/biases", conv_biases_1)
	c1 = tf.reshape(conv_weights_1, [32, 8,8, 4])
	cw1_image_hist = tf.image_summary("conv1_w", c1)

    with tf.name_scope("conv2") as conv2:
    	conv_weights_2 = weight_variable([4,4,32,64], "conv2_weights")
    	conv_biases_2 = bias_variable([64], "conv2_biases")
	cw2_hist = tf.histogram_summary("conv2/weights", conv_weights_2)
	cb2_hist = tf.histogram_summary("conv2/biases", conv_biases_2)
	c2 = tf.reshape(conv_weights_2, [32,64,4,4])
	cw2_image_hist = tf.image_summary("conv2_w", c2)

    with tf.name_scope("conv3") as conv3:
    	conv_weights_3 = weight_variable([3,3,64,64], "conv3_weights")
    	conv_biases_3 = bias_variable([64], "conv3_biases")
	cw3_hist = tf.histogram_summary("conv3/weights", conv_weights_3)
	cb3_hist = tf.histogram_summary("conv3/biases", conv_biases_3)
	c3 = tf.reshape(conv_weights_3, [64,64,3,3])
	cw3_image_hist = tf.image_summary("conv3_w", c3)

    with tf.name_scope("fc_1") as fc_1:
    	fc1_weights = weight_variable([2*2*64, 4624], "fc1_weights")
    	fc1_biases = bias_variable([4624], "fc1_biases")
        fc1_b_hist = tf.histogram_summary("fc_1/biases", fc1_biases)
        fc1_w_hist = tf.histogram_summary("fc_1/weights", fc1_weights)

    with tf.name_scope("fc_2") as fc_2:
    	fc2_weights = weight_variable([4624, NUM_ACTIONS], "fc2_weights")
    	fc2_biases = bias_variable([NUM_ACTIONS], "fc2_biases")
	fc2_w_hist = tf.histogram_summary("fc_2/weights", fc2_weights)
    	fc2_b_hist = tf.histogram_summary("fc_2/biases", fc2_biases)

    input_layer = tf.placeholder("float", [None, RESIZED_DATA_X, RESIZED_DATA_Y, STATE_FRAMES])

    h_conv1 = tf.nn.relu(tf.nn.conv2d(input_layer, conv_weights_1, strides=[1, 4, 4, 1], padding="SAME") + conv_biases_1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, conv_weights_2, strides=[1, 2, 2, 1], padding="SAME") + conv_biases_2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, conv_weights_3, strides=[1,1,1,1], padding="SAME") + conv_biases_3)
    h_pool3 = max_pool_2x2(h_conv3)

    h_pool3_flat = tf.reshape(h_pool3, [-1,2*2*64])
    final_hidden_activation = tf.nn.relu(tf.matmul(h_pool3_flat, fc1_weights, name='final_hidden_activation') + fc1_biases)

    output_layer = tf.matmul(final_hidden_activation, fc2_weights) + fc2_biases
    ol_hist = tf.histogram_summary("output_layer", output_layer)

 
    #we feed in the action the NN would do and targets=rewards ???
    readout_action = tf.reduce_sum(tf.mul(output_layer, action), reduction_indices=1)
    r_hist = tf.histogram_summary("readout_action", readout_action)

    with tf.name_scope("loss_summary"):
    	#loss = tf.reduce_mean(tf.square(output - target))
	loss = tf.reduce_mean(tf.square(target - readout_action))
	#loss = tf.reduce_mean(tf.square(output_layer - target))
    	tf.scalar_summary("loss", loss)

    merged = tf.merge_all_summaries()

    sum_writer = tf.train.SummaryWriter('/tmp/train', session.graph)

    train_operation = tf.train.AdamOptimizer(0.1).minimize(loss)

    session.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    if os.path.isfile("/home/ros/tensorflow-models/model-mini.ckpt"):
	saver.restore(session, "/home/ros/tensorflow-models/model-mini.ckpt")
	rospy.loginfo("model restored")
	

    #connect callbacks, so that we periodically get our values, degree and force
    rospy.Subscriber("adc_pi_plus_pub", Float32MultiArray, adc_callback)
    rospy.Subscriber("degree", Float32, degree_callback)
    rospy.Subscriber("probability", Float32, probability_callback)
    servo_pub = rospy.Publisher('servo_pwm_pi_sub', Int16MultiArray, queue_size=1)
    current_state_image_pub = rospy.Publisher('current_state_image', Image, queue_size=1)

    #the loop runs at 1hz
    rate = rospy.Rate(1)

    a=0
    #probability_of_random_action = 1
    servo_pub_values = Int16MultiArray()
    servo_pub_values.data = []

    last_action = np.zeros([NUM_ACTIONS])
    last_action[0] = 1   #stop action
    last_state = None
    sum_writer_index = 0
    MEMORY_SIZE = 100000
    OBSERVATION_STEPS = 1000 
    FUTURE_REWARD_DISCOUNT = 0.9
    observations = None
    if os.path.isfile("/home/ros/pickle-dump/save.p"):
	observations = pickle.load(open("/home/ros/pickle-dump/save.p", "rb"))
	rospy.loginfo("loaded observations, length is >%d<", len(observations))

    STEPPER = 0	
    punish = 0
    global current_degree
    global current_force_1
    global current_force_2
    
    while not rospy.is_shutdown():

	if a==0:
		rospy.loginfo("build last_state and use stop-action, or load checkpoint file and go on from there, should only run once")
		state_from_env = get_current_state()
		state_from_env = np.reshape(state_from_env, (RESIZED_DATA_X, RESIZED_DATA_Y))

		#for the first time we run, we build last_state with 4-last-states
		last_state = np.stack(tuple(state_from_env for _ in range(STATE_FRAMES)), axis=2)
		a=2
	elif a==1:
		#rospy.loginfo("a1 publish random or learned action")
		
		#badly simple decreasing probability
		global probability_of_random_action
		probability_of_random_action -= 0.001
		rospy.loginfo("probability_of_random_action >%f<", probability_of_random_action)

		current_action = np.zeros([NUM_ACTIONS])

		#build random action
		if random.random() <= probability_of_random_action :
		#if STEPPER == 0:
			rospy.loginfo("random action")
			#current_action = np.zeros([NUM_ACTIONS])
			rand = random.randrange(NUM_ACTIONS)
			current_action[rand] = 1
			STEPPER = 1		
			
		else :
			rospy.loginfo("learned action")
			#or we readout learned action
			current_action1 = session.run(output_layer, feed_dict={input_layer: [last_state]})
			current_action = np.zeros([NUM_ACTIONS])
			#current_action1 is not a array ?? build it new
			current_action[np.argmax(current_action1)] = 1
			STEPPER = 0

		#get the index of the max value to map this value to an original-action
		max_idx = np.argmax(current_action)
		#rospy.loginfo("action we publish >%d<", max_idx)
		#how do i map 32 or even more values to the appropriate action?
		
		#rospy.loginfo("blocker: current_force_1 >%f<  2 >%f<", current_force_1, current_force_2)
		if current_force_1 < 5:
			if current_force_2 < 5:
				if max_idx==2 or max_idx==5 or max_idx==6 or max_idx==7 or max_idx==8:
					idx_array = [1,3,4]
					idx_rand = random.randrange(3)
					#max_idx=1
					current_action = np.zeros([NUM_ACTIONS])
					current_action[idx_array[idx_rand]] = 1
					#punish = 1
			else:
				if max_idx==2 or max_idx==8:
					idx_array = [1,3,4,5,6,7]
					idx_rand = random.randrange(6)
					#max_idx=3
					current_action = np.zeros([NUM_ACTIONS])
					current_action[idx_array[idx_rand]] = 1
					#punish = 1
		elif current_force_2 < 5:
			if max_idx==6 or max_idx==7 or max_idx==8:
				idx_array = [1,2,3,4,5]
				idx_rand = random.randrange(5)
				#max_idx=5
				current_action = np.zeros([NUM_ACTIONS])
				current_action[idx_array[idx_rand]] = 1
				#punish = 1

		if current_force_1 > 35:
			if current_force_2 > 35:
				if max_idx==1 or max_idx==3 or max_idx==4 or max_idx==5 or max_idx==7:
					idx_array = [2,6,8]
					idx_rand = random.randrange(3)
					#max_idx=8
					current_action = np.zeros([NUM_ACTIONS])
					current_action[idx_array[idx_rand]] = 1
					#punish = 1
			else:
				if max_idx==1 or max_idx==3 or max_idx==4 or max_idx==7:
					idx_array = [2,5,6,8]
					idx_rand = random.randrange(4)
					#max_idx=5
					current_action = np.zeros([NUM_ACTIONS])
					current_action[idx_array[idx_rand]] = 1
					#punish = 1

		if current_force_2 > 35:
			if max_idx==1 or max_idx==3 or max_idx==4:
				idx_array = [2,5,6,7,8]
				idx_rand = random.randrange(5)
				#max_idx=6
				current_action = np.zeros([NUM_ACTIONS])
				current_action[idx_array[idx_rand]] = 1
				#punish = 1


		if current_degree < -80:
			if max_idx==1 or max_idx==4 or max_idx==7:
				idx_array = [2,3,5,6,8]
				idx_rand = random.randrange(5)
				#max_idx=5
				current_action = np.zeros([NUM_ACTIONS])
				current_action[idx_array[idx_rand]] = 1
		elif current_degree > 80:
			if max_idx==3 or max_idx==4 or max_idx==5:
				idx_array = [1,2,6,7,8]
				idx_rand = random.randrange(5)
				#max_idx=7
				current_action = np.zeros([NUM_ACTIONS])
				current_action[idx_array[idx_rand]] = 1
		

		rospy.loginfo("action we publish >%d<", max_idx)
		if max_idx==0:
			#2 servos stop
			servo_pub_values.data = [s1_stop,s2_stop, sx0, sx0, sx0, sx0, sx0, sx0]
		elif max_idx==1:
			servo_pub_values.data = [s1_fwd_1, s2_stop, sx0, sx0, sx0, sx0, sx0, sx0]
		elif max_idx==2:
                        servo_pub_values.data = [s1_bwd_1, s2_stop, sx0, sx0, sx0, sx0, sx0, sx0]
		elif max_idx==3:
                        servo_pub_values.data = [s1_stop, s2_fwd_1, sx0, sx0, sx0, sx0, sx0, sx0]
		elif max_idx==4:
                        servo_pub_values.data = [s1_fwd_1, s2_fwd_1, sx0, sx0, sx0, sx0, sx0, sx0]
		elif max_idx==5:
                        servo_pub_values.data = [s1_bwd_1, s2_fwd_1, sx0, sx0, sx0, sx0, sx0, sx0]
		elif max_idx==6:
                        servo_pub_values.data = [s1_stop, s2_bwd_1, sx0, sx0, sx0, sx0, sx0, sx0]
		elif max_idx==7:
                        servo_pub_values.data = [s1_fwd_1, s2_bwd_1, sx0, sx0, sx0, sx0, sx0, sx0]
		elif max_idx==8:
                        servo_pub_values.data = [s1_bwd_1, s2_bwd_1, sx0, sx0, sx0, sx0, sx0, sx0]

		last_action = current_action

		servo_pub.publish(servo_pub_values)
		# after publishing we publish stop servo values, so we are not continous, thats why i use this if-elif-elif construct

		a=2	

	elif a==2:
		rospy.loginfo("a2 publish stop values")
		#publish stop servo values, and let one ros-rate-cycle run, to settle the servos
		servo_pub_values.data = [s1_stop,s2_stop, sx0, sx0, sx0, sx0, sx0, sx0]
		servo_pub.publish(servo_pub_values)
		
		#we dont store a new state and action value
		a=3
	
	elif a==3:
		rospy.loginfo("a3 train")

		#we get our state
		state_from_env = get_current_state()
		reward = get_reward(state_from_env)

		state_from_env1 = np.reshape(state_from_env, (RESIZED_DATA_X, RESIZED_DATA_Y,1))
		current_state = np.append(last_state[:,:,1:], state_from_env1, axis=2)

		publish_state_image(state_from_env1, current_state_image_pub)

		if punish==1:
			punish=0
			reward = 0

		rospy.loginfo("reward %f", reward)


		#we append it to our observations
		observations.append((last_state, last_action, reward, current_state))
		if len(observations) > MEMORY_SIZE:
			observations.popleft()
		
		if len(observations) % OBSERVATION_STEPS == 0:
			#train
			rospy.loginfo("train network-----------------------------------------")

			mini_batch = random.sample(observations, MINI_BATCH_SIZE)
        		previous_states = [d[0] for d in mini_batch]
        		actions = [d[1] for d in mini_batch]
			rewards = [d[2] for d in mini_batch]
			current_states = [d[3] for d in mini_batch]
			previous_states = np.reshape(previous_states, (MINI_BATCH_SIZE, RESIZED_DATA_X, RESIZED_DATA_Y, STATE_FRAMES))

			agents_expected_reward = []
			
			agents_reward_per_action = session.run(output_layer, feed_dict={input_layer: current_states})
			for i in range(len(mini_batch)):
				agents_expected_reward.append(rewards[i] + FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))

			agents_expected_reward = np.reshape(agents_expected_reward, (MINI_BATCH_SIZE))
			
			actions = np.asarray(actions)	
        		_, result = session.run([train_operation, merged], feed_dict={input_layer: previous_states, action : actions, target: agents_expected_reward})


			sum_writer.add_summary(result, sum_writer_index)
			sum_writer_index += 1
		
		last_state = current_state
		#last_action = choose_next_action(),  that is a1 in this loop

	
		a=1

	rate.sleep()



    save_path = saver.save(session, "/home/ros/tensorflow-models/model-mini.ckpt")
    rospy.loginfo("model saved---------")
    session.close()
    rospy.loginfo("saving pickle, takes some time, perhaps minutes")
    pickle.dump(observations, open("/home/ros/pickle-dump/save.p", "wb"))
    rospy.loginfo("pickle saved")

    # spin() simply keeps python from exiting until this node is stopped
    #rospy.spin()

if __name__ == '__main__':
    listener()

