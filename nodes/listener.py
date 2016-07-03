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

####### description
#### http://discourse.ros.org/t/robotic-humanoid-hand/188
########


NUM_STATES = 264+264+1024+1024+1024+1024  #264 degree angle_goal, 264 possible degrees the joint could move, 1024 force values, two times
NUM_ACTIONS = 9  #3^2=9      ,one stop-state, one different speed left, one diff.speed right, two servos
STATE_FRAMES = 1
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
s1_fwd_1 = 380
s1_bwd_1 = 373
#.... normaly there are many more fwd or bwd speeds, but i dont know how to map so many mathematically
s2_stop = 400
s2_fwd_1 = 406
s2_bwd_1 = 394
sx0 = 1050  #do nothing value for not-used servos

observations = deque()


MINI_BATCH_SIZE = 10 
probability_of_random_action = 1



# we get the current state, that means, current degree and current forces. 
# We build a list, like the states-list, so we can compute reward
#
#
#
# angle_goal   degree      force1      force2
#    |               |
#    |               |         /\         /\
# ---|---------------|--------/  \-------/  \---------
# 68x68 = 4624 => 264_degree *2 + 4*1024

def get_current_state():
	global current_degree
	global current_force_1
	global current_force_2
	global degree_goal
	a1 = [(x==current_degree) for x in range(degree_possible_max)]
	a2 = [(x==current_force_1) for x in range(force_max_value)]
	a3 = [(x==current_force_2) for x in range(force_max_value)]
        
	b1 = np.linspace(0, 20, num=degree_goal)
	b2 = np.linspace(19.9, 0, num=degree_possible_max-degree_goal)

	b3 = np.linspace(0, 1, num=force_reward_max)
	b4 = np.linspace(0.99, 0, num=force_max_value-force_reward_max)

	b5 = np.linspace(0, 1, num=force_reward_max)
	b6 = np.linspace(0.99, 0, num=force_max_value-force_reward_max)

	d = []
	d.extend(a1)
	d.extend(a2)
	d.extend(a3)
	d.extend(b1)
	d.extend(b2)
	d.extend(b3)
	d.extend(b4)
	d.extend(b5)
	d.extend(b6)
	rospy.loginfo("get_current_state: len_4624 state >%d<", len(d))
	return d

def get_reward(state):
	s = np.asarray(state)
	s = s.reshape(2, 2312)
	print("get_reward: s.shape", s.shape)

	s1 = s[0] * s[1]
	print("get_reward: s1 shape", s1.shape)
	print("get_reward: s1", s1)
	print("get_reward: s0", s[0])

	s2 = sum(s1)
	print("get_reward: s2 sum", s2)

	return s2


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
    #minus 100, because 0 degree is the middle position
    global current_degree
    current_degree = int(data.data + 100)

def probability_callback(data):
    rospy.loginfo("probability heard %f", data.data)
    global probability_of_random_action
    probability_of_random_action = data.data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
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
    #build_reward_state()

    state = tf.placeholder("float", [None, NUM_STATES*STATE_FRAMES])
    action = tf.placeholder("float", [None, NUM_ACTIONS])
    target = tf.placeholder("float", [None])

    #hidden_weights = tf.Variable(tf.constant(0., shape=[NUM_STATES, NUM_ACTIONS]))
    Weights = tf.Variable(tf.truncated_normal([NUM_STATES*STATE_FRAMES, NUM_ACTIONS], mean=0.1, stddev=0.02, dtype=tf.float32, seed=1), name="Weights")
    h_w_hist = tf.histogram_summary("Weights", Weights)

    #bias
    biases = tf.Variable(tf.zeros([NUM_ACTIONS]), name="biases")
    b_hist = tf.histogram_summary("biases", biases) 

    conv_weights_1 = weight_variable([5,5,1,32])
    conv_biases_1 = bias_variable([32])

    #
    fc1_weights = weight_variable([25*25*32, 2450])
    fc1_biases = bias_variable([2450])
    fc1_b_hist = tf.histogram_summary("fc1_biases", fc1_biases)
    fc1_w_hist = tf.histogram_summary("fc1_weights", fc1_weights)

    fc2_weights = weight_variable([2450, 9])
    fc2_biases = bias_variable([9])
    fc2_w_hist = tf.histogram_summary("fc2_weights", fc2_weights)
    fc2_b_hist = tf.histogram_summary("fc2_biases", fc2_biases)

    #reshape state from 1,2450  to 50,49
    conv_state_1 = tf.reshape(state, [-1, 50, 49, 1])

    h_conv1 = tf.nn.relu(conv2d(conv_state_1, conv_weights_1) + conv_biases_1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_pool1_flat = tf.reshape(h_pool1, [-1, 25*25*32])
    final_hidden_activation = tf.nn.relu(tf.matmul(h_pool1_flat, fc1_weights, name='final_hidden_activation') + fc1_biases)

    output_layer = tf.matmul(final_hidden_activation, fc2_weights) + fc2_biases
    ol_hist = tf.histogram_summary("output_layer", output_layer)

    #we feed in our state and fetch (1,9) array with values. Highest value is the action the NN would do
    #output = tf.matmul(state, Weights) + biases
    #o_hist = tf.histogram_summary("output", output)

    #output1 = tf.nn.relu(output)
    #output1 = tf.nn.softmax(output)
    #o1_hist = tf.histogram_summary("output1", output1)
 
 
    #we feed in the action the NN would do and targets=rewards ???
    readout_action = tf.reduce_sum(tf.mul(output_layer, action), reduction_indices=1)
    r_hist = tf.histogram_summary("readout_action", readout_action)

    #r2 = tf.nn.softmax(readout_action)

    #relu_action = tf.nn.relu(readout_action)
    #relu_hist = tf.histogram_summary("relu_action", relu_action)
    #sig_action = tf.nn.sigmoid(readout_action)
    #sig_hist = tf.histogram_summary("sig_action", sig_action)

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

    #the loop runs at 1hz
    rate = rospy.Rate(1)

    a=0
    #probability_of_random_action = 1
    servo_pub_values = Int16MultiArray()
    servo_pub_values.data = []

    last_action = np.zeros([NUM_ACTIONS])
    last_action[0] = 1   #stop action
    sum_writer_index = 0
    MEMORY_SIZE = 10000
    OBSERVATION_STEPS = 20 
    FUTURE_REWARD_DISCOUNT = 0.9

    STEPPER = 0	
    punish = 0
    global current_degree
    global current_force_1
    global current_force_2
    
    while not rospy.is_shutdown():

	if a==0:
		rospy.loginfo("build last_state and use stop-action, or load checkpoint file and go on from there, should only run once")
		#get current state, that means degree, force1 and force2 in one list
		state_from_env = get_current_state()
		state_from_env = np.reshape(state_from_env,(NUM_STATES,1))

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
			last_state_array = np.reshape(last_state, (NUM_STATES*STATE_FRAMES))
			current_action1 = session.run(output_layer, feed_dict={state: [last_state_array]})
			current_action = np.zeros([NUM_ACTIONS])
			#current_action1 is not a array ?? build it new
			current_action[np.argmax(current_action1)] = 1
			STEPPER = 0

		#get the index of the max value to map this value to an original-action
		max_idx = np.argmax(current_action)
		rospy.loginfo("action we publish >%d<", max_idx)
		#how do i map 32 or even more values to the appropriate action?
		
		#rospy.loginfo("blocker: current_force_1 >%f<  2 >%f<", current_force_1, current_force_2)
		if current_force_1 == 0:
			if current_force_2==0:
				if max_idx==2 or max_idx==5 or max_idx==6 or max_idx==7 or max_idx==8:
					max_idx=0
					punish = 1
			else:
				if max_idx==2 or max_idx==8:
					max_idx=0
					punish = 1
		elif current_force_2==0:
			if max_idx==6 or max_idx==7 or max_idx==8:
				max_idx=0
				punish = 1

		if current_force_1 > 30:
			if current_force_2 > 30:
				if max_idx==1 or max_idx==3 or max_idx==4 or max_idx==5 or max_idx==7:
					max_idx=0
					punish = 1
			else:
				if max_idx==1 or max_idx==3 or max_idx==4 or max_idx==7:
					max_idx=0
					punish = 1

		if current_force_2 > 30:
			if max_idx==1 or max_idx==3 or max_idx==4:
				max_idx=0
				punish = 1


		if current_degree < 30:
			if max_idx==1 or max_idx==4 or max_idx==7:
				max_idx=5
				punish = 1
				#current_action = np.zeros([NUM_ACTIONS])
				#current_action[5] = 1
		elif current_degree > 170:
			if max_idx==3 or max_idx==4 or max_idx==5:
				max_idx=7
				punish = 1
				#current_action = np.zeros([NUM_ACTIONS])
				#current_action[7] = 1
		
	
		if max_idx==0:
			#current_action = np.zeros([NUM_ACTIONS])
			#current_action[0] = 1
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

		state_from_env1 = np.reshape(state_from_env, (NUM_STATES, 1,1))
		current_state = np.append(last_state[:,:,1:], state_from_env1, axis=2)


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
			current_states = np.reshape(current_states, (MINI_BATCH_SIZE, NUM_STATES*STATE_FRAMES))
			previous_states = np.reshape(previous_states, (MINI_BATCH_SIZE, NUM_STATES*STATE_FRAMES))

			agents_expected_reward = []
			
			agents_reward_per_action = session.run(output_layer, feed_dict={state: current_states})
			print("tt-rewards", rewards)
			print("tt-agents_reward_per_action", agents_reward_per_action)
			for i in range(len(mini_batch)):
				agents_expected_reward.append(rewards[i] + FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))

			agents_expected_reward = np.reshape(agents_expected_reward, (MINI_BATCH_SIZE))
			
			actions = np.asarray(actions)	
			print("tt-actions", actions)
			print("tt-target", agents_expected_reward)
			print("tt-pre-stat", previous_states)
        		_, result = session.run([train_operation, merged], feed_dict={state: previous_states, action : actions, target: agents_expected_reward})


			sum_writer.add_summary(result, sum_writer_index)
			sum_writer_index += 1
		
		last_state = current_state
		#last_action = choose_next_action(),  that is a1 in this loop

	
		a=1

	rate.sleep()



    save_path = saver.save(session, "/home/ros/tensorflow-models/model-mini.ckpt")
    rospy.loginfo("model saved---------")
    session.close()

    # spin() simply keeps python from exiting until this node is stopped
    #rospy.spin()

if __name__ == '__main__':
    listener()

