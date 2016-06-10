#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
from std_msgs.msg import Int16MultiArray
import tensorflow as tf
import numpy as np

NUM_STATES = 200+1024+1024  #possible degrees the joint could move, 1024 force values, two times
NUM_ACTIONS = 32 #2^5   ,one stop-state, two different speed left, two diff.speed right, two servos
GAMMA = 0.5

force_reward_max = 150  #where should the max point be
force_reward_length = 100  #how long/big the area around max
force_max_value = 1024     #how much force values possible
angle_goal = 90
angle_possible_max = 200  #how many degrees the angle can go max
current_degree = 0
current_force_1 = 0.0
current_force_2 = 0.0

angle = []
f1 = []
f2 = []
states = []

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


def adc_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "adc heard %s", data.data)
    rospy.loginfo("adc-val0: %f", data.data[0])
    rospy.loginfo("adc-val1: %f", data.data[1])
    current_force_1 = data.data[0]
    current_force_2 = data.data[1]
    
def degree_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "degree heard %f", data.data)
    current_degree = data.data

def listener():

    rospy.init_node('listener', anonymous=True)

    session = tf.Session()
    build_reward_state()

    state = tf.placeholder("float", [None, NUM_STATES])
    targets = tf.placeholder("float", [None, NUM_ACTIONS])

    hidden_weights = tf.Variable(tf.constant(0., shape=[NUM_STATES, NUM_ACTIONS]))

    output = tf.matmul(state, hidden_weights)
	
    with tf.name_scope("summaries"):
    	loss = tf.reduce_mean(tf.square(output - targets))
    	tf.scalar_summary("loss", loss)

    merged = tf.merge_all_summaries()

    sum_writer = tf.train.SummaryWriter('/tmp/train', session.graph)

    train_operation = tf.train.AdamOptimizer(0.1).minimize(loss)

    session.run(tf.initialize_all_variables())

    state_batch = []
    rewards_batch = []
    actions_batch = []
    


    rospy.Subscriber("adc_pi_plus_pub", Float32MultiArray, adc_callback)
    rospy.Subscriber("degree", Float32, degree_callback)
    servo_pub = rospy.Publisher('servo_pwm_pi_sub', Int16MultiArray, queue_size=1)

    rate = rospy.Rate(1)
    a=0

    while not rospy.is_shutdown():
	rospy.loginfo("here")

	if a==0:
		#get current state
		state_batch.append(get_current_state())
		a=1

	elif a==1:
		#instead of do random action with decreasing probability,i directly publish learned values which are at the beginning very random-like, or ? ==> publish 2 servo values
		# after publishing we publish stop servo values, so we are not continous, thats why i use this if-elif-elif construct

		#readout = session.run(output, feed_dict={state: [state_batch]})
		#servo_pub.publish(readout)
		
		a=2	

	elif a==2:
		#publish stop servo values, and let one ros-rate-cycle run, to settle the servos
		
		#build int16MultiArray with stop values for all servos (command uses values for 8 servos)
		#stop_val = 380,380,380,380,380,380,380,380
		#servo_pub.publish(stop_val)
		a=3
	
	elif a==3:
		#get current state, so we can perhaps reward this random action
 		# ?? state_batch.append(get_current_state())

		#use build_reward_state() to calc reward, if we have not reached goal_degree, we get no reward. If we have to much or too less force on the wire-ropes, we get no reward.
		#compare states[0] up to states[angle_possible_max-1] with get_current_state()[0] to get_current_state()[angle_possible_max-1]  ???
		#compare states[angle_possible_max] up to states[angle_possible_max + force_max_value-1] with get_current_state()[angle_possible_max] to get_current_state()[angle_possible_max + force_max_value-1]
		#compare the second force value like the above one
		#add up all three rewards into one value ???
		# ??? use this one reward value and the previous state and the current state for training ? how ?
		# first run "output" , then run "train_operation" ?
		#as we start from scratch, should it train with every step ? would be best, or?
		#running the output-op and then the train_operation-op ?
	
		#in deep-q pong of deepmind they use the last 4 frames, to get a feeling for the direction of the ball, this means i must use, the last 4 states together. Does this mean i must wait 4 states at the very first beginning?	
		a=1

	rate.sleep()




    # spin() simply keeps python from exiting until this node is stopped
    #rospy.spin()

if __name__ == '__main__':
    listener()

