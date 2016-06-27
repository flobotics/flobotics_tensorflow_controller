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


NUM_STATES = 200+200+1024+1024  #200 degree angle_goal, 200 possible degrees the joint could move, 1024 force values, two times
NUM_ACTIONS = 9  #3^2=9      ,one stop-state, one different speed left, one diff.speed right, two servos
STATE_FRAMES = 4
GAMMA = 0.5
RESIZED_DATA_X = 12
RESIZED_DATA_Y = 204   #12*204 = 2448 = NUM_STATES

force_reward_max = 15  #where should the max/middle point be, we get force values from 0.0 - 1023.0 (float),
force_reward_length = 10  #how long/big the area around max
force_max_value = 1024     #how much force values possible
angle_goal = 105		#to which angle should it go, get reward  #100 is the middle position
angle_possible_max = 200  #how many degrees the angle can go max
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
s1_bwd_1 = 374
#.... normaly there are many more fwd or bwd speeds, but i dont know how to map so many mathematically
s2_stop = 400
s2_fwd_1 = 404
s2_bwd_1 = 396
sx0 = 1050  #do nothing value for not-used servos

observations = deque()
#state_batch = []
#rewards_batch = []
#actions_batch = []


MINI_BATCH_SIZE = 5 
probability_of_random_action = 1

#   degree      force1      force2
#         |
#         |         /\         /\
# --------|--------/  \-------/  \---------
#
# for degree we reward only the direct reaching of the angle_goal, 
# for force1 and force2, we reward with a little pyramid, so that 
# it does not need to be exactly there (is that right??)
#



def build_reward_state():
	f_list1_length = force_reward_max - (force_reward_length/2)
	f_list1 = [(x==1050) for x in range(f_list1_length)]
	print "length f1 >%d<" %len(f_list1)

	f_list_pos = np.linspace(0,1, num=force_reward_length/2)
	print "length f-pos >%d<" %len(f_list_pos)


	f_list_neg = np.linspace(0.99,0,num=force_reward_length/2)
	print "length f-neg >%d<" %len(f_list_neg)


	f_list2 = [(x==1050) for x in range((1024 - (len(f_list1) + len(f_list_pos) + len(f_list_neg) ) ))]
	print "length f_list2 >%d<" %len(f_list2)

	force_l_1 = np.linspace(0,1, num=force_reward_max)
	print "length force_l_1 >%d<" %len(force_l_1)
	force_l_2 = np.linspace(0.99,0, num=force_max_value-force_reward_max)
	print "length force_l_2 >%d<" %len(force_l_2)
	f1.extend(force_l_1)
	f1.extend(force_l_2)
	f2.extend(force_l_1)
	f2.extend(force_l_2)
	
	#c = []
	#f1.extend(f_list1)
	#f1.extend(f_list_pos)
	#f1.extend(f_list_neg)
	#f1.extend(f_list2)
	#f1[0] = -1
	#f1[1] = -1
	#f1[1023] = -1
	#print(f1)
	#copy the same into f2
        #f2.extend(f_list1)
        #f2.extend(f_list_pos)
        #f2.extend(f_list_neg)
        #f2.extend(f_list2)
	#f2[0] = -1
	#f2[1] = -1
	#f2[1023] = -1
        #print(f2)

	#angle = [(x==angle_goal) for x in range(angle_possible_max)]
	angle1 = np.linspace(0,3, num=angle_goal)
	print "length angle1 >%d<" %len(angle1)
	angle2 = np.linspace(2.99,0, num=angle_possible_max-angle_goal)
	print "length angle2 >%d<" %len(angle2)
	angle.extend(angle1)
	angle.extend(angle2)
	#print(angle)	

	states.extend(angle)
	states.extend(f1)
	states.extend(f2)
	print "length of states >%d>" %len(states)


# we get the current state, that means, current degree and current forces. 
# We build a list, like the states-list, so we can compute reward
#
#
#
# angle_goal   degree      force1      force2
#    |               |
#    |               |         /\         /\
# ---|---------------|--------/  \-------/  \---------


def get_current_state():
	global current_degree
	global current_force_1
	global current_force_2
	global angle_goal
	a1 = [(x==angle_goal) for x in range(angle_possible_max)]
	a = [(x==current_degree) for x in range(angle_possible_max)]
	b = [(x==current_force_1) for x in range(force_max_value)]
	c = [(x==current_force_2) for x in range(force_max_value)]
	d = []
	d.extend(a1)
	d.extend(a)
	d.extend(b)
	d.extend(c)
	return d

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

#the main thread/program
#it runs in a loop, todo something and learn to reach the angle_goal (in degree)
def listener():

    rospy.init_node('listener', anonymous=True)

    session = tf.Session()
    build_reward_state()

    state = tf.placeholder("float", [None, NUM_STATES*STATE_FRAMES])
    action = tf.placeholder("float", [None, NUM_ACTIONS])
    target = tf.placeholder("float", [None])

    #hidden_weights = tf.Variable(tf.constant(0., shape=[NUM_STATES, NUM_ACTIONS]))
    Weights = tf.Variable(tf.truncated_normal([NUM_STATES*STATE_FRAMES, NUM_ACTIONS], mean=0.0, stddev=0.02, dtype=tf.float32, seed=1), name="Weights")
    h_w_hist = tf.histogram_summary("Weights", Weights)

    #bias
    biases = tf.Variable(tf.zeros([NUM_ACTIONS]), name="biases")
    b_hist = tf.histogram_summary("biases", biases) 

    #we feed in our state and fetch (1,9) array with values. Highest value is the action the NN would do
    output = tf.matmul(state, Weights) + biases
    o_hist = tf.histogram_summary("output", output)

    #we feed in the action the NN would do and targets=rewards ???
    readout_action = tf.reduce_sum(tf.mul(output, action), reduction_indices=1)

    with tf.name_scope("loss_summary"):
    	#loss = tf.reduce_mean(tf.square(output - target))
	loss = tf.reduce_mean(tf.square(target - readout_action))
    	tf.scalar_summary("loss", loss)

    merged = tf.merge_all_summaries()

    sum_writer = tf.train.SummaryWriter('/tmp/train', session.graph)

    train_operation = tf.train.AdamOptimizer(0.1).minimize(loss)

    session.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    if os.path.isfile("/home/ros/tensorflow-models/model.ckpt"):
	    saver.restore(session, "/home/ros/tensorflow-models/model.ckpt")

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
    OBSERVATION_STEPS = 5
    FUTURE_REWARD_DISCOUNT = 0.9

    STEPPER = 0	
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
		last_state = np.stack(tuple(state_from_env for _ in range(4)), axis=2)
		
		a=2
	elif a==1:
		rospy.loginfo("a1 publish random or learned action")
		
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
			last_state_array = np.reshape(last_state, (NUM_STATES*4))
			current_action1 = session.run(output, feed_dict={state: [last_state_array]})
			#current_action1 is not a array ?? build it new
			current_action[np.argmax(current_action1)] = 1
			STEPPER = 0

		#get the index of the max value to map this value to an original-action
		max_idx = np.argmax(current_action)
		rospy.loginfo("action we publish >%d<", max_idx)
		#how do i map 32 or even more values to the appropriate action?
		#global current_force_1
		#global current_force_2
		rospy.loginfo("blocker: current_force_1 >%f<  2 >%f<", current_force_1, current_force_2)
		if current_force_1 == 0:
			if current_force_2==0:
				if max_idx==2 or max_idx==5 or max_idx==6 or max_idx==7 or max_idx==8:
					max_idx=0
			else:
				if max_idx==2 or max_idx==5 or max_idx==8:
					max_idx=0
		elif current_force_2==0:
			if max_idx==6 or max_idx==7 or max_idx==8:
				max_idx=0

	
		if max_idx==0:
			current_action = np.zeros([NUM_ACTIONS])
			current_action[0] = 1
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

		state_from_env1 = np.reshape(state_from_env, (NUM_STATES, 1,1))
		current_state = np.append(last_state[:,:,1:], state_from_env1, axis=2)

		#we calculate the reward, for that we use states[] from build_reward_state()
                #the reward for reaching the degree/angle_goal
		#global current_degree
		#global current_force_1
		#global current_force_2
                rospy.loginfo("current_degree %d",  current_degree)
		rospy.loginfo("current_force_1 >%f<  2 >%f<", current_force_1, current_force_2)
		print("states len", len(states))


		r1 = state_from_env[angle_possible_max-1 + current_degree] + states[current_degree]
                #the reward for holding a specified force on wire-1
                r2 = state_from_env[angle_possible_max-1 + angle_possible_max + current_force_1] + states[angle_possible_max-1 + current_force_1]
                #the reward for holding a specified force on wire-2
                r3 = state_from_env[angle_possible_max-1 + angle_possible_max + force_max_value + current_force_2] + states[angle_possible_max-1 + force_max_value + current_force_2]
                #add them
                reward = r1 + r2 + r3

		rospy.loginfo("reward %f", reward)


		#we append it to our observations
		observations.append((last_state, last_action, reward, current_state))
		if len(observations) > MEMORY_SIZE:
			observations.popleft()
		
		if len(observations) > OBSERVATION_STEPS:
			#train
			rospy.loginfo("train network-----------------------------------------")

			mini_batch = random.sample(observations, MINI_BATCH_SIZE)
        		previous_states = [d[0] for d in mini_batch]
        		actions = [d[1] for d in mini_batch]
			rewards = [d[2] for d in mini_batch]
			current_states = [d[3] for d in mini_batch]
			current_states = np.reshape(current_states, (MINI_BATCH_SIZE, NUM_STATES*4))
			previous_states = np.reshape(previous_states, (MINI_BATCH_SIZE, NUM_STATES*4))

			agents_expected_reward = []
			
			agents_reward_per_action = session.run(output, feed_dict={state: current_states})

			for i in range(len(mini_batch)):
				agents_expected_reward.append(rewards[i] + FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))

			agents_expected_reward = np.reshape(agents_expected_reward, (MINI_BATCH_SIZE))
			
	
        		_, result = session.run([train_operation, merged], feed_dict={state: previous_states, action : actions, target: agents_expected_reward})


			sum_writer.add_summary(result, sum_writer_index)
			sum_writer_index += 1
		
		last_state = current_state
		#last_action = choose_next_action(),  that is a1 in this loop

	
		a=1

	rate.sleep()



    save_path = saver.save(session, "/home/ros/tensorflow-models/model.ckpt")
    rospy.loginfo("model saved---------")

    # spin() simply keeps python from exiting until this node is stopped
    #rospy.spin()

if __name__ == '__main__':
    listener()

