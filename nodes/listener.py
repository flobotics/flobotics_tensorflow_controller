#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
from std_msgs.msg import Int16MultiArray
import tensorflow as tf
import numpy as np
import random

####### description
#### http://discourse.ros.org/t/robotic-humanoid-hand/188
########


NUM_STATES = 200+1024+1024  #possible degrees the joint could move, 1024 force values, two times
NUM_ACTIONS = 9  #3^2=9      ,one stop-state, two different speed left, two diff.speed right, two servos
GAMMA = 0.5

force_reward_max = 150  #where should the max point be
force_reward_length = 100  #how long/big the area around max
force_max_value = 1024     #how much force values possible
angle_goal = 90
angle_possible_max = 200  #how many degrees the angle can go max
current_degree = 0
current_force_1 = 0
current_force_2 = 0

angle = []
f1 = []
f2 = []
states = []

#variables for bad-mapping approach, s1=servo1 , s2=servo2,.... 
s1_stop = 377
s1_fwd_1 = 380
s1_bwd_1 = 374
#.... normaly there are many more fwd or bwd speeds, but i dont know how to map so many mathematically
s2_stop = 399
s2_fwd_1 = 402
s2_bwd_1 = 396
sx0 = 1050  #do nothing value for not-used servos



#   degree      force1      force2
#         |
#         |         /\         /\
# --------|--------/  \-------/  \---------
#
# for degree we reward only the direct reaching of the angle_goal, 
# for force1 and force2, we reward with a little pyramid, so that 
# it does not need to be exactly there (is that right??)
#
# do i need to put in also the angle_goal, like degree? So that it learns, to get degree to the same value as angle_goal?
#
#
#
# angle_goal   degree      force1      force2
#    |               |
#    |               |         /\         /\
# ---|---------------|--------/  \-------/  \---------



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


	#c = []
	f1.extend(f_list1)
	f1.extend(f_list_pos)
	f1.extend(f_list_neg)
	f1.extend(f_list2)
	#print(f1)
	#copy the same into f2
        f2.extend(f_list1)
        f2.extend(f_list_pos)
        f2.extend(f_list_neg)
        f2.extend(f_list2)
        #print(f2)

	angle = [(x==angle_goal) for x in range(angle_possible_max)]
	#print(angle)

	states.extend(angle)
	states.extend(f1)
	states.extend(f2)
	print "length of states >%d>" %len(states)


# we get the current state, that means, current degree and current forces. 
# We build a list, like the states-list, so we can compute reward
def get_current_state():
	a = [(x==current_degree) for x in range(angle_possible_max)]
	b = [(x==current_force_1) for x in range(force_max_value)]
	c = [(x==current_force_2) for x in range(force_max_value)]
	d = []
	d.extend(a)
	d.extend(b)
	d.extend(c)
	print "length curr-state d >%d<" %len(d)
	return d

# callback which delivers us periodically the adc values of the force sensors
# adc values are floats from 0.0 to 5.0.  we convert them to int from 0-1023
def adc_callback(data):
    #rospy.loginfo(rospy.get_caller_id() + "adc heard %s", data.data)
    #rospy.loginfo("adc-val0: %f", data.data[0])
    #rospy.loginfo("adc-val1: %f", data.data[1])
    current_force_1 = (1023/5.0)*data.data[0]
    current_force_2 = (1023/5.0)*data.data[1]
    
#callback which delivers us periodically the degree, from 0-200 degree
def degree_callback(data):
    #rospy.loginfo(rospy.get_caller_id() + "degree heard %f", data.data)
    current_degree = data.data

#the main thread/program
#it runs in a loop, todo something and learn to reach the angle_goal (in degree)
def listener():

    rospy.init_node('listener', anonymous=True)

    session = tf.Session()
    build_reward_state()

    state = tf.placeholder("float", [None, NUM_STATES])
    targets = tf.placeholder("float", [None, NUM_ACTIONS])

    #hidden_weights = tf.Variable(tf.constant(0., shape=[NUM_STATES, NUM_ACTIONS]))
    hidden_weights = tf.Variable(tf.truncated_normal([NUM_STATES, NUM_ACTIONS], mean=0.0, stddev=0.02, dtype=tf.float32, seed=1), name="hidden_weights")
    h_w_hist = tf.histogram_summary("hidden_weights", hidden_weights)

    #bias needed?

    output = tf.matmul(state, hidden_weights)
    o_hist = tf.histogram_summary("output", output)

    #readout_action = tf.reduce_sum(tf.mul(output, targets), reduction_indices=1)
	
    with tf.name_scope("loss_summary"):
    	loss = tf.reduce_mean(tf.square(output - targets))
	#loss = tf.reduce_mean(tf.square(targets - readout_action))
    	tf.scalar_summary("loss", loss)

    merged = tf.merge_all_summaries()

    sum_writer = tf.train.SummaryWriter('/tmp/train', session.graph)

    train_operation = tf.train.AdamOptimizer(0.1).minimize(loss)

    session.run(tf.initialize_all_variables())

    state_batch = []
    rewards_batch = []
    actions_batch = []
    

    #connect callbacks, so that we periodically get our values, degree and force
    rospy.Subscriber("adc_pi_plus_pub", Float32MultiArray, adc_callback)
    rospy.Subscriber("degree", Float32, degree_callback)
    servo_pub = rospy.Publisher('servo_pwm_pi_sub', Int16MultiArray, queue_size=1)

    #the loop runs at 1hz
    rate = rospy.Rate(1)

    a=0
    sum_writer_index = 0
    probability_of_random_action = 1
    servo_pub_values = Int16MultiArray()
    servo_pub_values.data = []



    while not rospy.is_shutdown():

	if a==0:
		rospy.loginfo("a0")
		#get current state
		state_batch.append(get_current_state())

		action_rewards = [0.,0.,0.,0.,0.,0.,0.,0.,0.] #states[ + GAMMA * np.max(state_reward)  
                rewards_batch.append(action_rewards)

		
		a=1
		#rospy.loginfo("get_current_state >%s<", str(state_batch))
	elif a==1:
		rospy.loginfo("a1")
		#random action is better to explore bigger state space

		probability_of_random_action -= 0.01

		#build random action
		if random.random() <= probability_of_random_action :
			rospy.loginfo("random")
			current_action_state = np.zeros([NUM_ACTIONS])
			rand = random.randrange(NUM_ACTIONS)
			current_action_state[rand] = 1		
			
		else :
			rospy.loginfo("NOTrandom")
			#or we readout learned action
			current_action_state = session.run(output, feed_dict={state: [state_batch[-1]]})

		#get the index of the max value to map this value to an original-action
		max_idx = np.argmax(current_action_state)
		rospy.loginfo("max_idx >%d<", max_idx)
		#how do i map 32 or even more values to the appropriate action?
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


		actions_batch.append(current_action_state)

		#servo_pub.publish(servo_pub_values)
		# after publishing we publish stop servo values, so we are not continous, thats why i use this if-elif-elif construct

		a=2	

	elif a==2:
		rospy.loginfo("a2")
		#publish stop servo values, and let one ros-rate-cycle run, to settle the servos
		servo_pub_values.data = [s1_stop,s2_stop, sx0, sx0, sx0, sx0, sx0, sx0]
		#servo_pub.publish(servo_pub_values)
		a=3
	
	elif a==3:
		rospy.loginfo("a3")

		#we get the action that the NN would do 
		state_reward = session.run(output, feed_dict={state: [state_batch[-1]]})

		#get current state, so we can perhaps reward this random action
                last_state = get_current_state()
                state_batch.append(last_state)


		#we calculate the reward, for that we use states[] from build_reward_state()
		#the reward for reaching the degree/angle_goal
		r1 = last_state[current_degree] + states[current_degree]
		#the reward for holding a specified force on wire-1
		r2 = last_state[angle_possible_max-1 + current_force_1] + states[angle_possible_max-1 + current_force_1]
		#the reward for holding a specified force on wire-2
		r3 = last_state[angle_possible_max-1 + force_max_value + current_force_2] + states[angle_possible_max-1 + force_max_value + current_force_2]
		#add them
		r = r1 + r2 + r3
		
		#q-function ?
		action_rewards = r + GAMMA * state_reward #np.max(state_reward) #   [0.,0.,0.,0.,0.,0.,0.,0.,0.] # [states[current_degree] + GAMMA * np.max(state_reward)]   
		rewards_batch.append(action_rewards.tolist())
		rospy.loginfo("a3-rewards_batch >%s<", rewards_batch) 
		#rospy.loginfo("a3-state_batch >%s<", state_batch)


# the error the above line produces

		
#	[INFO] [WallTime: 1466173268.044297] a3-rewards_batch >[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [[0.965092122554779, 0.9984378814697266, 1.0046687126159668, 0.9839721322059631, 0.9959143996238708, 1.038711428642273, 0.9991259574890137, 0.9968912601470947, 1.0089023113250732]]]<
#Traceback (most recent call last):
#  File "/home/ros/catkin_ws/src/tensorflow_controller/nodes/listener.py", line 301, in <module>
#    listener()
#  File "/home/ros/catkin_ws/src/tensorflow_controller/nodes/listener.py", line 285, in listener
#    _, result = session.run([train_operation, merged], feed_dict={state: state_batch, targets: rewards_batch})
#  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 340, in run
#    run_metadata_ptr)
#  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 548, in _run
#    np_val = np.array(subfeed_val, dtype=subfeed_t.dtype.as_numpy_dtype)
#ValueError: setting an array element with a sequence.
	




		_, result = session.run([train_operation, merged], feed_dict={state: state_batch, targets: rewards_batch})
		sum_writer.add_summary(result, sum_writer_index)
		sum_writer_index += 1
	
		#in deep-q pong of deepmind they use the last 4 frames, to get a feeling for the direction of the ball, this means i must use, the last 4 states together. Does this mean i must wait 4 states at the very first beginning?	
		a=1

	rate.sleep()




    # spin() simply keeps python from exiting until this node is stopped
    #rospy.spin()

if __name__ == '__main__':
    listener()

