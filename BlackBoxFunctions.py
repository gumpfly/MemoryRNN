import tensorflow as tf
import numpy as np


def HiddenStateSpeedLSTM(RNNVarWeights, RNNVarBiases, HiddenStateH, HiddenStateC, RNNInput, RNNInputTF):

    # Given a LSTM RNN, a batch of Hidden States, and a RNN Input (batch or single), returns the velocity of each Hidden State
    if(RNNInputTF):
        xt_rand = RNNInput
    else:
        xt_rand = tf.constant(RNNInput)

    f = tf.sigmoid(tf.matmul(xt_rand, RNNVarWeights['Wf']) + tf.matmul(HiddenStateH, RNNVarWeights['Uf']) + RNNVarBiases['bf'])
    i = tf.sigmoid(tf.matmul(xt_rand, RNNVarWeights['Wi']) + tf.matmul(HiddenStateH, RNNVarWeights['Ui']) + RNNVarBiases['bi'])
    o = tf.sigmoid(tf.matmul(xt_rand, RNNVarWeights['Wo']) + tf.matmul(HiddenStateH, RNNVarWeights['Uo']) + RNNVarBiases['bo'])
    cNew = tf.multiply(f, HiddenStateC) + tf.multiply(i, tf.tanh(tf.matmul(xt_rand, RNNVarWeights['Wc']) + tf.matmul(HiddenStateH, RNNVarWeights['Uc']) + RNNVarBiases['bc']))
    hDer = tf.multiply(o, tf.tanh(cNew)) - HiddenStateH
    cDer = tf.tanh(cNew) - tf.tanh(HiddenStateC)

    return(tf.reduce_sum(tf.square(hDer) + tf.square(cDer), axis=1))

def HiddenStateSpeedGRU(RNNVarWeights,RNNVarBiases, HiddenStateH, RNNInput, RNNInputTF):

    # Given a GRU RNN, a batch of Hidden States (C & H), and a RNN Input (batch or single), returns the velocity of each Hidden State

    if(RNNInputTF):
        xt_rand = RNNInput
    else:
        xt_rand = tf.constant(RNNInput)

    z = tf.sigmoid(tf.matmul(xt_rand, RNNVarWeights['Wz']) + tf.matmul(HiddenStateH, RNNVarWeights['Uz']) + RNNVarBiases['bz'])
    r = tf.sigmoid(tf.matmul(xt_rand, RNNVarWeights['Wr']) + tf.matmul(HiddenStateH, RNNVarWeights['Ur']) + RNNVarBiases['br'])
    hDer = tf.add(tf.multiply(1 - z, HiddenStateH), tf.multiply(z, tf.tanh(tf.matmul(xt_rand, RNNVarWeights['Wh']) + tf.matmul(tf.multiply(r, HiddenStateH), RNNVarWeights['Uh']) + RNNVarBiases['bh']))) - HiddenStateH

    return(tf.reduce_sum(tf.square(hDer), axis=1))

def BlackBoxAlgorithmLSTM(RNNVarWeights, RNNVarBiases, InitPointsH, InitPointsC, RNNInput, NumSteps = 1, Lr = 0.001):

    # Given a LSTM RNN, a batch of Initial Hidden States (Can be random, Can be center-of-mass of classes), and a RNN Input (batch or single),
    # the code performs the Black - Box Algorithm and returns the slow-points to which each hidden-state converged to

    LearnRate = tf.placeholder(tf.float32, [])
    h = tf.Variable(InitPointsH, dtype=tf.float32)
    c = tf.Variable(InitPointsC, dtype=tf.float32)

    SpeedEach = HiddenStateSpeedLSTM(RNNVarWeights, RNNVarBiases, h, c, RNNInput, RNNInputTF = False)
    loss = tf.reduce_sum(SpeedEach)
    optimizer = tf.train.AdamOptimizer(learning_rate=LearnRate)
    train_op = optimizer.minimize(loss)
    clip_op = tf.assign(h, tf.clip_by_value(h, -1, 1))

    init = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)
        for j in range(NumSteps):
            if(j == int(NumSteps * 0.9)):
                Lr = Lr / 10
            _, lossP, h_fix, c_fix, SpeedDig = sess.run([train_op, loss, h, c, SpeedEach],feed_dict={LearnRate: Lr})
            sess.run(clip_op)
            if (j % 100 == 0):
                print(np.sqrt(lossP))

    return(h_fix, c_fix, SpeedDig)



def BlackBoxAlgorithmGRU(RNNVarWeights,RNNVarBiases, InitPointsH, RNNInput, NumSteps=1, Lr=0.001):

    # Given a GRU RNN, a batch of Initial Hidden States (Can be random, Can be center-of-mass of classes), and a RNN Input (batch or single),
    # the code performs the Black - Box Algorithm and returns the slow-points to which each hidden-state converged to

    LearnRate = tf.placeholder(tf.float32, [])
    h = tf.Variable(InitPointsH, dtype=tf.float32)

    SpeedEach = HiddenStateSpeedGRU(RNNVarWeights, RNNVarBiases, h, RNNInput, RNNInputTF = False)
    loss = tf.reduce_sum(SpeedEach)

    optimizer = tf.train.AdamOptimizer(learning_rate=LearnRate)
    train_op = optimizer.minimize(loss, var_list=[h])
    clip_op = tf.assign(h, tf.clip_by_value(h, -1, 1))

    init = tf.global_variables_initializer()



    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)
        for j in range(NumSteps):
            if(j == int(NumSteps * 0.9)):
                Lr = Lr / 10
            _, lossP, h_fix, SpeedDig = sess.run([train_op, loss, h, SpeedEach], feed_dict={LearnRate: Lr})
            sess.run(clip_op)
            if(j % 100 == 0):
                print(np.sqrt(lossP))

    return (h_fix, SpeedDig)
