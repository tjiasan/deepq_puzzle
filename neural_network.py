import tensorflow as tf
import cv2
import sys
sys.path.append("Wrapped Game Code/")
import draw_frame# whichever is imported "as game" will be used
import draw_frame as game
import random
import numpy as np
from collections import deque

GAME = 'maze' # the name of the game being played for log files
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
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.Maze()
    # store the previous observations in replay memory
    D = deque()


    # do nothing 
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1



    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("/home/jason/maze/save")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"

    epsilon = INITIAL_EPSILON
    t = 0
    t2= 0
    ttc = 0

    while "pigs" != "fly":
        #get the state before choosing action 
        s_t= game_state.draw_frame()
        
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        
        
        if random.random() <= epsilon:
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
            

        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

  
        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        for i in range(0, K):
            #play the maze game
            terminal,r_t,s_t1,= game_state.game(action_index)

            if terminal == True:
                ttc = t2
                t2=0

            # store the transition in D
            D.append((s_t, a_t, r_t,s_t1,terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            
            readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
            
                # if end of the game
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
              # if normal, add future predicted value, weighted by gamma
                else:
                    y_batch.append(r_batch[i] + GAMMA* np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch})

        # update the old values
       
        t += 1
        t2 +=1



        # save progress every 10000 iterations
        if t % 10000 == 0:
            
            saver.save(sess, '/home/jason/maze/save/' + GAME + '-dqn', global_step = t)
            
        
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        if t %100 ==0:
            print "TIMESTEP", t, "/ STATE", state, "/ LINES", "/ ACTION",action_index ,"/ steps to completion", ttc, "/ Q_MAX %e" % np.max(readout_t)
            #print "TIMESTEP", t,"/ score", score
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("/home/jason/logs_dummy2/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
