#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
from std_msgs.msg import Int16MultiArray
import tensorflow as tf
import numpy as np
import random
from collections import deque

####### description
#### http://discourse.ros.org/t/robotic-humanoid-hand/188
########


NUM_STATES = 200+200+1024+1024  #200 degree angle_goal, 200 possible degrees the joint could move, 1024 force values, two times
NUM_ACTIONS = 9  #3^2=9      ,one stop-state, one different speed left, one diff.speed right, two servos
GAMMA = 0.5

force_reward_max = 15  #where should the max/middle point be, we get force values from 0.0 - 1023.0 (float),
force_reward_length = 10  #how long/big the area around max
force_max_value = 1024     #how much force values possible
angle_goal = 5		#to which angle should it go, get reward
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
s2_stop = 399
s2_fwd_1 = 402
s2_bwd_1 = 396
sx0 = 1050  #do nothing value for not-used servos

observations = deque()
#state_batch = []
#rewards_batch = []
#actions_batch = []


MINI_BATCH_SIZE = 5 

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
#
#
#
# angle_goal   degree      force1      force2
#    |               |
#    |               |         /\         /\
# ---|---------------|--------/  \-------/  \---------


def get_current_state():
	a1 = [(x==angle_goal) for x in range(angle_possible_max)]
	a = [(x==current_degree) for x in range(angle_possible_max)]
	b = [(x==current_force_1) for x in range(force_max_value)]
	c = [(x==current_force_2) for x in range(force_max_value)]
	d = []
	d.extend(a1)
	d.extend(a)
	d.extend(b)
	d.extend(c)
	print "length curr-state d >%d<" %len(d)
	return d

# callback which delivers us periodically the adc values of the force sensors
# adc values are floats from 0.0 to 5.0.  we convert them to int from 0.0-1023.0  (float)
def adc_callback(data):
    #rospy.loginfo(rospy.get_caller_id() + "adc heard %s", data.data)
    #rospy.loginfo("adc-val0: %f", (1023/5.0)*data.data[0])
    #rospy.loginfo("adc-val1: %f", (1023/5.0)*data.data[1])
    current_force_1 = (1023/5.0)*data.data[0]
    current_force_2 = (1023/5.0)*data.data[1]
    
#callback which delivers us periodically the degree, from 0.0-200.0 degree (float)
def degree_callback(data):
    #rospy.loginfo(rospy.get_caller_id() + "degree heard %f", data.data)
    current_degree = data.data


#the main thread/program
#it runs in a loop, todo something and learn to reach the angle_goal (in degree)
def listener():

    rospy.init_node('listener', anonymous=True)

    session = tf.Session()
    build_reward_state()

    state = tf.placeholder("float", [None, NUM_STATES*4])
    targets = tf.placeholder("float", [None, NUM_ACTIONS])

    #hidden_weights = tf.Variable(tf.constant(0., shape=[NUM_STATES, NUM_ACTIONS]))
    hidden_weights = tf.Variable(tf.truncated_normal([NUM_STATES*4, NUM_ACTIONS], mean=0.0, stddev=0.02, dtype=tf.float32, seed=1), name="hidden_weights")
    h_w_hist = tf.histogram_summary("hidden_weights", hidden_weights)

    #bias
    biases = tf.Variable(tf.zeros([NUM_ACTIONS]), name="biases")
    b_hist = tf.histogram_summary("biases", biases) 

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


    #connect callbacks, so that we periodically get our values, degree and force
    rospy.Subscriber("adc_pi_plus_pub", Float32MultiArray, adc_callback)
    rospy.Subscriber("degree", Float32, degree_callback)
    servo_pub = rospy.Publisher('servo_pwm_pi_sub', Int16MultiArray, queue_size=1)

    #the loop runs at 1hz
    rate = rospy.Rate(1)

    a=0
    probability_of_random_action = 1
    servo_pub_values = Int16MultiArray()
    servo_pub_values.data = []

    last_action = np.zeros([NUM_ACTIONS])
    last_action[0] = 1   #stop action
    sum_writer_index = 0
    MEMORY_SIZE = 10000
    OBSERVATION_STEPS = 5
    FUTURE_REWARD_DISCOUNT = 0.9

    while not rospy.is_shutdown():

	if a==0:
		rospy.loginfo("build last_state and use stop-action, or load checkpoint file and go on from there, should only run once")
		#get current state, that means degree, force1 and force2 in one list
		state_from_env = get_current_state()
		state_from_env = np.reshape(state_from_env,(NUM_STATES,1))
		#print("state_from_env shape", state_from_env.shape)
		#------>('state_from_env shape', (2448, 1))

		#for the first time we run, we build last_state with 4-last-states
		last_state = np.stack(tuple(state_from_env for _ in range(4)), axis=2)
		#print("a0 last_state.shape", last_state.shape)		
		#------>('a0 last_state.shape', (2448, 1, 4))		
		
		a=2
	elif a==1:
		rospy.loginfo("a1 publish random or learned action")
		
		#badly simple decreasing probability
		probability_of_random_action -= 0.01

		#build random action
		if random.random() >= probability_of_random_action :
			rospy.loginfo("random action")
			current_action = np.zeros([NUM_ACTIONS])
			rand = random.randrange(NUM_ACTIONS)
			current_action[rand] = 1		
			
		else :
			rospy.loginfo("learned action")
			#or we readout learned action
			last_state_array = np.reshape(last_state, (NUM_STATES*4))
			current_action = session.run(output, feed_dict={state: [last_state_array]})

		#get the index of the max value to map this value to an original-action
		max_idx = np.argmax(current_action)
		rospy.loginfo("action we publish >%d<", max_idx)
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
		#print("state_from_env shape", state_from_env.shape)
		#---->('state_from_env shape', (2448, 1, 1))
		#print("last_state shape", last_state.shape)
		#---->('last_state shape', (2448, 1, 4))
		current_state = np.append(last_state[:,:,1:], state_from_env1, axis=2)
		#print("a3 current_state.shape", current_state.shape)
		#---->('a3 current_state.shape', (2448, 1, 4))

		test_s = np.reshape(current_state, (NUM_STATES*4, 1))
		print("test_s shape", test_s.shape)

		#we calculate the reward, for that we use states[] from build_reward_state()
                #the reward for reaching the degree/angle_goal
                r1 = state_from_env[angle_possible_max-1 + current_degree] + states[angle_possible_max-1 + current_degree]
                #the reward for holding a specified force on wire-1
                r2 = state_from_env[angle_possible_max-1 + angle_possible_max + current_force_1] + states[angle_possible_max-1 + angle_possible_max + current_force_1]
                #the reward for holding a specified force on wire-2
                r3 = state_from_env[angle_possible_max-1 + angle_possible_max + force_max_value + current_force_2] + states[angle_possible_max-1 + angle_possible_max + force_max_value + current_force_2]
                #add them
                reward = r1 + r2 + r3

		print("reward", reward)


		#we append it to our observations
		observations.append((last_state, last_action, reward, current_state))
		if len(observations) > MEMORY_SIZE:
			observations.popleft()
		
		if len(observations) > OBSERVATION_STEPS:
			#train
			print("train")

			mini_batch = random.sample(observations, MINI_BATCH_SIZE)
        		previous_states = [d[0] for d in mini_batch]
			print("t-prev-states len", len(previous_states))
        		actions = [d[1] for d in mini_batch]
			print("t-actions", actions)
			rewards = [d[2] for d in mini_batch]
			print("t-rewards", rewards)
			current_states = [d[3] for d in mini_batch]
			print("t-cur-state len", len(current_states))
			print("t-cur-state type", type(current_states))
			print("t-cur-state[0] len", len(current_states[0]))
			test_c = np.reshape(current_states, (5, NUM_STATES*4))
			print("t-test_c shape", test_c.shape)
					
		


			agents_expected_reward = []
			
			#print("t-prev-states", previous_states)
			#wrong ???	
			agents_reward_per_action = session.run(output, feed_dict={state: [test_c]})


			print("t-agents-reward-per-action", agents_reward_per_action)

			for i in range(len(mini_batch)):
				agents_expected_reward.append(rewards[i] + FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))


        		_, result = session.run([train_operation, merged], feed_dict={state: state_batch, targets: rewards_batch})

			sum_writer.add_summary(result, sum_writer_index)
			sum_writer_index += 1
		
		last_state = current_state
		#last_action = choose_next_action(),  that is a1 in this loop

	
		a=1

	rate.sleep()




    # spin() simply keeps python from exiting until this node is stopped
    #rospy.spin()

if __name__ == '__main__':
    listener()

