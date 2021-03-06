from numpy import array
import pandas as pd
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import GRU
import numpy as np
from tcn import TCN
import pandas as pd
from keras import losses
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn import preprocessing
from tcn_pre.get_forcast_obs_data import get_data
from tcn_pre.filterpool import filter_config
import math
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.layers import LSTM, Dense,GRU,Dropout

wfid = ['130735','140827','152523','152525','320603','320924','331022','341524',
        '360402','360729','360734','371428','420922','422805','430430','430981',
        '450901','513433','522730','530622','532331','610621','610826','620525',
        '620908','632801','6403','640326','640327','652106','652111','652255','654205','652268']
wf = wfid[32]
print(wf)
kwargs = {
    'wfid': wf,
    'start_time': '2017-10-01',
    'end_time': '2019-10-31',
    "source": ['CMA','EC','GFS1'],
    "point": ['001'],
    'data_count': 20
}
wspd = get_data(**kwargs)
out = wspd.data_merge()
# filter_data = out
filter_data = filter_config(out,int(wf))
numb = filter_data.shape[0]
print(numb)
data_in = filter_data
data_in['day'] = pd.to_datetime(data_in['ptime']).dt.day
data_in['hour'] = pd.to_datetime(data_in['ptime']).dt.hour
data_in['minute'] = pd.to_datetime(data_in['ptime']).dt.minute
# cma_max = data_in.loc[:,"MIX_001_speed"].max()
# mete_max = data_in.loc[:,"METE_001_speed"].max()
cma_max = data_in.loc[:,"CMA_001_speed"].max()
ec_max = data_in.loc[:,"EC_001_speed"].max()
gfs_max = data_in.loc[:,"GFS1_001_speed"].max()
power_max = data_in.loc[:,"obs_power"].max()
wind_max = data_in.loc[:,"obs_w"].max()
def get_mse(records_real, records_predict):
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None
def get_rmse(records_real, records_predict):
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None
def scheduler(epoch):
    if epoch  == 130 :
        lr = K.get_value(train.optimizer.lr)
        K.set_value(train.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    if epoch  == 140 :
        lr = K.get_value(train.optimizer.lr)
        K.set_value(train.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(train.optimizer.lr)
#ec 25.095  mix  22.6545   power  50.391
def get_dataset(n_star,n_end):
    X1,y = list(), list()
    for num in range(n_star,n_end-96-16):
        # 生成输入序列
        dataset = data_in[num:num+96+16]
        source7 = dataset['day'][-16:] /31
        source4 = dataset['hour'][-16:]/23
        source5 = dataset['minute'][-16:]/45
        source6 = dataset['GFS1_001_speed'][-16:] /gfs_max
        source1 = dataset['CMA_001_speed'][-16:]/cma_max
        source2 = dataset['EC_001_speed'][-16:]/ec_max
        source3 = dataset['obs_power'][:-16]/power_max
        source8 = dataset['obs_w'][:-16]/wind_max
        source = source1.append(source2).append(source3).append(source4).append(source5).append(source6).append(source7).append(source8)

        source = array(source).reshape(128+64+96, 1)
        target = dataset['obs_power'][-16:]/power_max
        X1.append(source)
        y.append(target)
    return array(X1), array(y)

def get_test_dataset(n_star,n_end):
    X1,y = list(), list()
    for num in range((n_end-n_star)//16):
        # 生成输入序列
        dataset = data_in[n_star+num*16-96:n_star+num*16+16]
        source7 = dataset['day'][-16:] /31
        source4 = dataset['hour'][-16:]/23
        source5 = dataset['minute'][-16:]/45
        source6 = dataset['GFS1_001_speed'][-16:] /gfs_max
        source1 = dataset['CMA_001_speed'][-16:]/cma_max
        source2 = dataset['EC_001_speed'][-16:]/ec_max
        source3 = dataset['obs_power'][:-16]/power_max
        source8 = dataset['obs_w'][:-16]/wind_max
        source = source1.append(source2).append(source3).append(source4).append(source5).append(source6).append(source7).append(source8)

        source = array(source).reshape(128+64+96, 1)
        target = dataset['obs_power'][-16:]/power_max
        X1.append(source)
        y.append(target)
    return array(X1), array(y)

def define_models(n_input, n_output):
    # 训练模型中的encoder  32 16 128
    encoder_inputs = Input(shape=(None, n_input)) #500  32
    encoder = TCN(64,activation='relu',return_sequences=True)(encoder_inputs)
    encoder2 = TCN(32, activation='relu',return_sequences=True)(encoder)
    encoder3 = TCN(16, activation='relu')(encoder2)
    decoder_dense = Dense(n_output,activation='linear')(encoder3)
    model = Model(encoder_inputs, decoder_dense)
    return model

# 参数设置
true, predict = list(), list()
n_steps_in = 32+96+64+96
n_steps_out = 16
# 定义模型
train= define_models(n_steps_in, n_steps_out)
train.compile(optimizer='adam', loss=losses.mse, metrics=['acc'])
X1,y = get_dataset(0,50086)
X1 = np.reshape(X1, (X1.shape[0],1,-1))
print(X1.shape, y.shape)
reduce_lr = LearningRateScheduler(scheduler)
train.fit(X1,y, epochs=150,batch_size=512,shuffle=1,callbacks=[reduce_lr])
# train.fit(X1, y, epochs=100,batch_size=512,shuffle=1)
train.save(r"D:\PycharmProjects\untitled\tcn_pre\model_TCN\tcn200conf_"+str(wf) +"time.h5")
# model = load_model(r"D:\PycharmProjects\untitled\tcn_pre\model_TCN\tcn_"+str(wf) +"time.h5")
# for i in range(20):
#     X1,y = get_test_dataset(39671+i*16,39671+16+i*16)
#     X1 = np.reshape(X1, (X1.shape[0], 1, X1.shape[1]))
#     state = model.predict(X1,batch_size=1)
#     print('y=%s, yhat=%s' % ((y[0]*power_max), (state[0]*power_max)))
#     yy = y[0]*power_max
#     zz = state[0]*power_max
#     yy.reshape(16, 1)
#     zz.reshape(16, 1)
#     true.append(yy)
#     predict.append(zz)
# aa = array(true).reshape(16 * 20, 1)
# bb = array(predict).reshape(16 * 20, 1)
# getRMSE = 1-get_rmse(aa,bb)/49.5
# print(getRMSE)
# plt.plot(aa, 'blue')
# plt.plot(bb, 'red')
# plt.show()