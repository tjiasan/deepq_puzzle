import tensorflow as tf
import cv2
import sys
sys.path.append("Wrapped Game Code/")
#import draw_frame# whichever is imported "as game" will be used
#import  draw_frame as game
import draw_frame as game

import random
import numpy as np
from collections import deque

GAME = 'catdog' # the name of the game being played for log files
ACTIONS = 3 # number of valid actions
GAMMA = 0.95 # decay rate of past observations
OBSERVE = 2000. # timesteps to observe before training
EXPLORE = 2000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 85000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others
#text_file = open("result.txt","w")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 2, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 2])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_pool1_drop = tf.nn.dropout(h_pool1,0.8)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1,0.8)
    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    game_state = game.Maze()

    D = deque()


    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("/home/jason/maze/save")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"

    t=0
    t2=0
    ttc=0
    while "pigs" != "fly":
        #get the state before choosing action 
        t2 +=1
        s_t= game_state.draw_frame()
        
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        
    
        action_index = 0
        action_index = np.argmax(readout_t)

        terminal, reward, state = game_state.game(action_index)
        
        if terminal ==True:
            ttc= t2
            t2=0
            print "The number of steps it takes to complete the maze is ", ttc
        
        





  

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
