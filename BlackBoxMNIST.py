import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import sys
import BlackBoxFunctions as BBF

dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

MinSize = 10000000

for digit in range(10):
    if(np.count_nonzero(np.argmax(dataset.train.labels,1)==digit)<MinSize):
        MinSize = np.count_nonzero(np.argmax(dataset.train.labels,1)==digit)

MNIST_Images = np.zeros((MinSize,784,10))
for digit in range(10):
    MNIST_Images[:,:,digit] = (dataset.train.images[np.where(np.argmax(dataset.train.labels,1)==digit)[0][0:MinSize]])



config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      )

config.gpu_options.allow_growth =True



SysInputs=sys.argv

MNIST=True
DIGITS=False

NoiseMu = 0.1307
NoiseSig = 0.30816

def generate_batch(digit,batch_size,RandImages):
  Images=RandImages
  for i in range(batch_size):
      RandomIndexes = np.random.randint(0, MinSize - 1, 1)
      TrueImage = MNIST_Images[RandomIndexes,:,digit]
      Images[i,0] = TrueImage
  return(Images)



Arc = SysInputs[1]
LearningCurr = SysInputs[2]
Instance = SysInputs[3]
Mode = SysInputs[4]


timestepsSample = 15 #int(SysInputs[4])
timesteps = 15

BatchSize=100
num_hidden=200
num_input=784+1
num_classes=11
HiddenStateDer='H'
Noise=1
LossThres=-100000
learning_rate=0.001

FileName = 'RNNDynamicsPaper/MNIST_SpeedReg/' + Mode + '/' + '10Dig_' + str(LearningCurr) + '_' + Arc + '_' + Instance + '/'
path = 'RNNDynamicsPaper/MNISTAttractor/' + Mode + '/BlackBoxAnalysis/' + Arc + '/' + LearningCurr + '_' + Instance + '/'


if not os.path.exists(path):
    os.makedirs(path)



HiddenStep='End'
ReadOutStep='End'
weightsFile= FileName+'/Variables/WeightsAt'+HiddenStep+'.npy'
biasesFile= FileName+'/Variables/BiasesAt'+HiddenStep+'.npy'
OutBiasesFile= FileName+'/Variables/OutBiasesAt'+HiddenStep+'.npy'
OutWeightsFile= FileName+'/Variables/OutWeightsAt'+HiddenStep+'.npy'
weights=np.load(weightsFile)
biases=np.load(biasesFile)
OutWeights=np.load(OutWeightsFile)
OutBiases=np.load(OutBiasesFile)
Weights=weights.item(0)
Biases=biases.item(0)




xRand = np.ones((784))*NoiseMu
xt_Rand = np.concatenate((xRand, [0]), axis=0)
xt_Rand = np.reshape(xt_Rand, [1, -1]).astype(np.float32)
xReadOut = np.concatenate((xRand, [1]), axis=0)
xReadOut = np.reshape(xReadOut, [1, -1]).astype(np.float32)

if(Arc=='LSTM'):
    weights = {'Wf': tf.constant(Weights['Wf']),
               'Uf': tf.constant(Weights['Uf']),
               'Wi': tf.constant(Weights['Wi']),
               'Ui': tf.constant(Weights['Ui']),
               'Wo': tf.constant(Weights['Wo']),
               'Uo': tf.constant(Weights['Uo']),
               'Wc': tf.constant(Weights['Wc']),
               'Uc': tf.constant(Weights['Uc'])}

    biases = {'bf': tf.constant(Biases['bf']),
              'bi': tf.constant(Biases['bi']),
              'bo': tf.constant(Biases['bo']),
              'bc': tf.constant(Biases['bc'])}

if (Arc == 'GRU'):
    weights = {'Wz': tf.constant(Weights['Wz']),
               'Uz': tf.constant(Weights['Uz']),
               'Wr': tf.constant(Weights['Wr']),
               'Ur': tf.constant(Weights['Ur']),
               'Wh': tf.constant(Weights['Wh']),
               'Uh': tf.constant(Weights['Uh'])}

    biases = {'bz': tf.constant(Biases['bz']),
              'br': tf.constant(Biases['br']),
              'bh': tf.constant(Biases['bh'])}


out_weights = tf.constant(OutWeights)
out_biases = tf.constant(OutBiases)


X = tf.placeholder("float", [None, None ,None])
ZerosMat =  tf.placeholder("float", [None ,None])
H_init = tf.placeholder("float", [None,None])

if(Arc == 'LSTM'):
    C_init = tf.placeholder("float", [None,None])

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def ReadOutLSTM(STATEH, STATEC):
    STATEF = sigmoid(np.matmul(xReadOut, Weights['Wf'])+np.matmul(STATEH, Weights['Uf'])+Biases['bf'])
    STATEI = sigmoid(np.matmul(xReadOut, Weights['Wi']) + np.matmul(STATEH, Weights['Ui']) + Biases['bi'])
    STATEO = sigmoid(np.matmul(xReadOut, Weights['Wo'])+np.matmul(STATEH, Weights['Uo'])+Biases['bo'])
    STATEC = STATEF*STATEC + STATEI*np.tanh(np.matmul(xReadOut, Weights['Wc'])+np.matmul(STATEH, Weights['Uc'])+Biases['bc'])
    STATEH = STATEO * np.tanh(STATEC)
    return(np.matmul(STATEH, OutWeights)+OutBiases)


def ReadOutGRU(STATEH):
    STATER = sigmoid(np.matmul(xReadOut, Weights['Wr'])+np.matmul(STATEH, Weights['Ur'])+Biases['br'])
    STATEZ = sigmoid(np.matmul(xReadOut, Weights['Wz']) + np.matmul(STATEH, Weights['Uz']) + Biases['bz'])
    STATEH = (1-STATEZ)*STATEH+STATEZ*np.tanh(np.matmul(xReadOut, Weights['Wh'])+np.matmul(STATER*STATEH, Weights['Uh'])+Biases['bh'])
    return(np.matmul(STATEH, OutWeights)+OutBiases)

def SigmaGTag(Vec):
    return(tf.sigmoid(Vec)*(1-tf.sigmoid(Vec)))

def SigmaHTag(Vec):
    return(1-tf.square(tf.tanh(Vec)))


def LSTMRNN(x,TS,Ts):
    i=0
    for j in range(Ts, TS):
        xt=tf.concat((x[:,j,:],ZerosMat),axis=1)
        WfXt = tf.matmul(xt, weights['Wf'])
        WiXt = tf.matmul(xt, weights['Wi'])
        WoXt = tf.matmul(xt, weights['Wo'])
        WcXt = tf.matmul(xt, weights['Wc'])
        if(i==0):
            Ufht = tf.matmul(H_init, weights['Uf'])
            Uiht = tf.matmul(H_init,weights['Ui'])
            Uoht = tf.matmul(H_init,weights['Uo'])
            Ucht = tf.matmul(H_init,weights['Uc'])
            ct = C_init
        else:
            Ufht = tf.matmul(ht,weights['Uf'])
            Uiht = tf.matmul(ht,weights['Ui'])
            Uoht = tf.matmul(ht,weights['Uo'])
            Ucht = tf.matmul(ht,weights['Uc'])
        ft=tf.sigmoid(WfXt+Ufht+biases['bf'])
        it=tf.sigmoid(WiXt+Uiht+biases['bi'])
        ot=tf.sigmoid(WoXt+Uoht+biases['bo'])
        ct=tf.multiply(ft,ct)+tf.multiply(it,tf.tanh(WcXt+Ucht+biases['bc']))
        ht=tf.multiply(ot,tf.tanh(ct))
        i=i+1
    return(ht, ct)


def GRURNN(x,TS, Ts):
    i=0
    for j in range(Ts, TS):
        xt=tf.concat((x[:,j,:],ZerosMat),axis=1)
        WzXt = tf.matmul(xt, weights['Wz'])
        WrXt = tf.matmul(xt, weights['Wr'])
        WhXt = tf.matmul(xt, weights['Wh'])
        if(i==0):
            Uzht = tf.matmul(H_init, weights['Uz'])
            Urht = tf.matmul(H_init, weights['Ur'])
            zt = tf.sigmoid(WzXt + Uzht + biases['bz'])
            rt = tf.sigmoid(WrXt + Urht + biases['br'])
            Uhrtht = tf.matmul(tf.multiply(rt, H_init), weights['Uh'])
            ht = tf.add(tf.multiply(1 - zt, H_init), tf.multiply(zt, tf.tanh(WhXt + Uhrtht + biases['bh'])))
        else:
            Uzht = tf.matmul(ht, weights['Uz'])
            Urht = tf.matmul(ht, weights['Ur'])
            zt = tf.sigmoid(WzXt + Uzht + biases['bz'])
            rt = tf.sigmoid(WrXt + Urht + biases['br'])
            Uhrtht = tf.matmul(tf.multiply(rt, ht), weights['Uh'])
            ht = tf.add(tf.multiply(1 - zt, ht), tf.multiply(zt, tf.tanh(WhXt + Uhrtht + biases['bh'])))
        i=i+1
    return (ht)

if(Arc=='LSTM'):
    h0, c0 = LSTMRNN(X,timestepsSample,0)
if (Arc == 'GRU'):
    h0 = GRURNN(X,timestepsSample,0)

init1 = tf.global_variables_initializer()

sess=tf.Session(config = config)
sess.run(init1)

Images = np.zeros((10 * BatchSize, timesteps,784))
NoiseImages = np.random.normal(loc = NoiseMu, scale = NoiseSig/3, size = (1000,784))[0:timesteps]
FeedImages = np.repeat(NoiseImages[np.newaxis, :, :], int(BatchSize), axis=0)
for Class in range(10):
    Images[Class*BatchSize:(Class+1)*BatchSize] = generate_batch(Class, BatchSize, FeedImages)

print(Images.shape)
if(Arc=='LSTM'):
    InitHiddenH = np.zeros((10,num_hidden))
    InitHiddenC = np.zeros((10,num_hidden))
    H0, C0 = sess.run([h0,c0],feed_dict={X:Images, H_init: np.zeros((10 * BatchSize,num_hidden)), C_init: np.zeros((10 * BatchSize,num_hidden)), ZerosMat: np.zeros((10 * BatchSize,1))})

    for Digit in range(10):
        HiddenHDigit = H0[Digit*BatchSize:(Digit+1)*BatchSize]
        HiddenCDigit = C0[Digit*BatchSize:(Digit+1)*BatchSize]
        ReadOutH0 = np.argmax(ReadOutLSTM(HiddenHDigit,HiddenCDigit), axis=1) - 1
        WeightVec = (ReadOutH0 == Digit) / np.count_nonzero(ReadOutH0 == Digit)
        InitHiddenH[Digit] = np.average(HiddenHDigit, weights=WeightVec, axis=0).astype('float32')
        InitHiddenC[Digit] = np.average(HiddenCDigit, weights=WeightVec, axis=0).astype('float32')
    h = tf.Variable(InitHiddenH, dtype=tf.float32)
    c = tf.Variable(InitHiddenC, dtype=tf.float32)

if(Arc!='LSTM'):
    InitHiddenH = np.zeros((10,num_hidden))
    H0 = sess.run([h0],feed_dict={X:Images, H_init: np.zeros((10 * BatchSize,num_hidden)), ZerosMat: np.zeros((10 * BatchSize,1))})[0]
    for Digit in range(10):
        HiddenHDigit = H0[Digit*BatchSize:(Digit+1)*BatchSize]
        ReadOutH0 = np.argmax(ReadOutGRU(HiddenHDigit), axis=1) - 1
        WeightVec = (ReadOutH0 == Digit) / np.count_nonzero(ReadOutH0 == Digit)
        InitHiddenH[Digit] = np.average(HiddenHDigit, weights=WeightVec, axis=0).astype('float32')
    h = tf.Variable(InitHiddenH, dtype=tf.float32)

if(Arc == 'GRU'):
    h_fix, SpeedDig = BBF.BlackBoxAlgorithmGRU(RNNVarWeights = weights, RNNVarBiases = biases, InitPointsH = InitHiddenH, RNNInput = xt_Rand)
if(Arc == 'LSTM'):
    h_fix, c_fix, SpeedDig = BBF.BlackBoxAlgorithmLSTM(RNNVarWeights = weights, RNNVarBiases = biases, InitPointsH = InitHiddenH, InitPointsC = InitHiddenC, RNNInput = xt_Rand)
#


if (Arc == 'LSTM'):
    ReadOutF20 = np.argmax(ReadOutLSTM(h_fix,c_fix), axis=1)-1
    ReadOut20 = np.argmax(ReadOutLSTM(InitHiddenH,InitHiddenC), axis=1)-1

    print('\nReadOutFF_20: ' + str(ReadOutF20))
    print('ReadOut_20: ' + str(ReadOut20))

if (Arc == 'GRU'):

    ReadOutF20 = np.argmax(ReadOutGRU(h_fix), axis=1)-1
    ReadOut20 = np.argmax(ReadOutGRU(InitHiddenH), axis=1)-1

    print('\nReadOutFF_20: ' + str(ReadOutF20))
    print('ReadOut_20: ' + str(ReadOut20))


print('h20-h20F: ' + str(np.sqrt(np.sum(np.square(InitHiddenH-h_fix)))))

if(Arc == 'LSTM'):
    np.save(path + '/CHid20.npy', InitHiddenC)
    np.save(path + '/CHidF20.npy', c_fix)
    np.save(path + '/FinalCHid.npy', c_fix)

np.save(path+'/Hid20.npy',InitHiddenH)
np.save(path+'/HidF20.npy',h_fix)
np.save(path+'/FinalHid.npy',h_fix)
file = open(path + '/EigVal.txt', 'w')


np.save(path + '/FinalSpeed', np.sqrt(SpeedDig))
file.write('\n\nReadOutFF_20: ' + str(ReadOutF20))
np.save(path + '/ReadOutF20', ReadOutF20)
file.write('\nReadOut_20: ' + str(ReadOut20))
np.save(path + '/ReadOut_20', ReadOut20)
file.write('h20-h20F: ' + str(np.sqrt(np.sum(np.square(InitHiddenH - h_fix)))))

file.write('\nHdif: ' + str(np.sqrt(np.sum(np.square(InitHiddenH - h_fix)))))
file.write("\nFinal Hidden: " + str(h_fix))
file.close()


