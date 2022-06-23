#!/opt/share/bin/anaconda3/bin python
# coding: utf-8

import shap  # 这行代码没有用，但是不能删除，删除后不可运行！！！

from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve,auc
from tensorflow_core.python.keras.backend import set_session
import tensorflow.keras.backend as K
from tensorflow_core.python.keras.callbacks import Callback

np.random.seed(1337)
from sklearn.metrics import roc_auc_score, confusion_matrix
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import CuDNNLSTM, Bidirectional, LSTM, Dropout, Dense, Activation, Flatten, Conv1D, Reshape, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import sys,os
import joblib

class UseANN(object):
    def __init__(self, fold, self_test=False, gpu_id=0):
        self.fold = fold
        self.self_test = self_test
        self.need = list(range(0,31))
        self.average = 'no'
        self.val_scores = 0

        print("\n第{} fold开始..".format(self.fold))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # don't pre-allocate memory; allocate as-needed
        config.gpu_options.per_process_gpu_memory_fraction = 0.4  # limit memory to be allocated
        set_session(tf.Session(config=config))

    # 读取1D数据的时候使用，只读取CLS
    def get_CLS_data(self, train_file, label_file):
        print('从硬盘读取数据开始...')
        self.get_class = 'CLS' # 控制 generate_test 的读取
        self.train = []
        with open(train_file, 'r') as f:
            for line in f:
                line = line.split(',')
                self.train.append([float(x) for x in line[1:]])

        label = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.split(',')
                label.append(int(line[0]))

        self.train = np.array(self.train)
        le = preprocessing.LabelEncoder()
        le.fit(label)
        self.y_train = le.transform(label)
        self.train = self.train.astype(np.float32)
        try:
            shape_2 = self.train .shape[2]
        except Exception:
            shape_2 = 1
        self.train  = self.train.reshape(-1, self.train.shape[1], shape_2)

        self.y_train = self.y_train.astype(np.float)
        onehot_encoder = OneHotEncoder(sparse=False)
        self.y_train = self.y_train.reshape(len(self.y_train), 1)
        # train_label = np.array(train_label).reshape(len(train_label), 1)
        self.y_train = onehot_encoder.fit_transform(self.y_train)

        return self.train, self.y_train

    # 读取2D数据的时候使用,也就是31+3的数据
    def get_all_data(self, train_file, label_file):
        print('从硬盘读取数据开始...')
        self.get_class = 'all'
        import joblib
        self.train = np.array(joblib.load(train_file))
        self.train = self.train.astype(np.float32)
        # need = [1]
        need = self.need
        self.train = self.train[:, need, :]
        if isinstance(need, list):
            self.train = self.train.transpose((0, 2, 1))
            if self.average == 'yes':
                self.train = np.average(self.train, axis=2)

        label = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.split(',')
                label.append(int(line[0]))
        le = preprocessing.LabelEncoder()
        le.fit(label)
        self.y_train = le.transform(label)

        try:
            shape_2 = self.train .shape[2]
        except Exception:
            shape_2 = 1
        self.train  = self.train.reshape(-1, self.train.shape[1], shape_2)

        self.y_train = self.y_train.astype(np.float)
        onehot_encoder = OneHotEncoder(sparse=False)
        self.y_train = self.y_train.reshape(len(self.y_train), 1)
        # train_label = np.array(train_label).reshape(len(train_label), 1)
        self.y_train = onehot_encoder.fit_transform(self.y_train)

        return self.train, self.y_train

    # 1d数据生成，以及2d数据使用1d卷积请使用此函数
    def generate_1D_test(self, test_file, test_label):
        self.y_test = []
        self.x_test = []
        def get_CLS_data_test(self):
            with open(test_file, 'r') as f:
                for line in f:
                    line = line.split(',')
                    self.x_test.append([float(x) for x in line[1:]])
            self.x_test = np.array(self.x_test)

            self.y_test = []
            with open(test_label, 'r') as f:
                for line in f:
                    line = line.split(',')
                    self.y_test.append(int(line[0]))

            le = preprocessing.LabelEncoder()
            le.fit(self.y_test)
            self.y_test = le.transform(self.y_test)
            self.x_test = self.x_test.astype(np.float32)

            return self.x_test, self.y_test

        # 读取2D数据的时候使用,也就是31+3的数据
        def get_all_data_test(self):
            import joblib
            self.x_test = joblib.load(test_file)
            self.x_test = np.array(self.x_test)
            # need = [0]
            # need.extend(list(range(1,33)))
            need = self.need
            self.x_test = self.x_test[:, need, :]
            if isinstance(need, list):
                self.x_test = self.x_test.transpose((0, 2, 1))
                if self.average == 'yes':
                    self.x_test = np.average(self.x_test, axis=2)

            self.y_test = []
            with open(test_label, 'r') as f:
                for line in f:
                    line = line.split(',')
                    self.y_test.append(int(line[0]))
            le = preprocessing.LabelEncoder()
            le.fit(self.y_test)
            self.y_test = le.transform(self.y_test)
            self.x_test = self.x_test.astype(np.float32)

            return self.x_test, self.y_test

        if self.get_class == 'CLS':
            get_CLS_data_test(self)
        elif self.get_class == 'all':
            get_all_data_test(self)

        self.y_test = self.y_test
        self.x_test = np.array(self.x_test)
        try:
            shape_2 = self.x_test.shape[2]
        except Exception:
            shape_2 = 1
        self.x_test  = self.x_test .reshape(-1, self.x_test .shape[1], shape_2)

        return self.x_test , self.y_test

    def generate_indp_test(self, test_file, test_label):
        self.y_test = self.y_test.astype(np.float)
        onehot_encoder = OneHotEncoder(sparse=False)
        self.y_test = self.y_test.reshape(len(self.y_test), 1)
        # train_label = np.array(train_label).reshape(len(train_label), 1)
        self.y_test = onehot_encoder.fit_transform(self.y_test)

        self.train = np.concatenate((self.train,self.x_test), axis=0)
        self.y_train = np.concatenate((self.y_train, self.y_test), axis=0)

        try:
            shape_2 = self.train .shape[2]
        except Exception:
            shape_2 = 1
        self.train  = self.train.reshape(-1, self.train.shape[1], shape_2)


        self.y_test = []
        self.x_test = []
        def get_CLS_data_test(self):
            with open(test_file, 'r') as f:
                for line in f:
                    line = line.split(',')
                    self.x_test.append([float(x) for x in line[1:]])
            self.x_test = np.array(self.x_test)

            self.y_test = []
            with open(test_label, 'r') as f:
                for line in f:
                    line = line.split(',')
                    self.y_test.append(int(line[0]))

            le = preprocessing.LabelEncoder()
            le.fit(self.y_test)
            self.y_test = le.transform(self.y_test)
            self.x_test = self.x_test.astype(np.float32)

            return self.x_test, self.y_test

        # 读取2D数据的时候使用,也就是31+3的数据
        def get_all_data_test(self):
            import joblib
            self.x_test = joblib.load(test_file)
            self.x_test = np.array(self.x_test)
            # need = [0]
            # need.extend(list(range(1,33)))
            need = self.need
            self.x_test = self.x_test[:, need, :]
            if isinstance(need, list):
                self.x_test = self.x_test.transpose((0, 2, 1))
                if self.average == 'yes':
                    self.x_test = np.average(self.x_test, axis=2)

            self.y_test = []
            with open(test_label, 'r') as f:
                for line in f:
                    line = line.split('\t')
                    self.y_test.append(int(line[0]))
            le = preprocessing.LabelEncoder()
            le.fit(self.y_test)
            self.y_test = le.transform(self.y_test)
            self.x_test = self.x_test.astype(np.float32)

            return self.x_test, self.y_test

        if self.get_class == 'CLS':
            get_CLS_data_test(self)
        elif self.get_class == 'all':
            get_all_data_test(self)

        self.y_test = self.y_test
        self.x_test = np.array(self.x_test)
        try:
            shape_2 = self.x_test.shape[2]
        except Exception:
            shape_2 = 1
        self.x_test  = self.x_test .reshape(-1, self.x_test .shape[1], shape_2)

        return self.x_test , self.y_test

    def __config_GPU(self):
        keras.backend.clear_session()

    def __use_Fnn(self):
        keras.backend.clear_session()
        model = Sequential()
        model.add(Reshape(input_shape=(self.train.shape[1], self.train.shape[2]),
                          target_shape=(self.train.shape[1], self.train.shape[2])))
        model.add(Dense(units=1024,kernel_regularizer=tf.keras.regularizers.l2(0.001),activity_regularizer=tf.keras.regularizers.l1(0.001)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())

        model.add(Dense(2))
        model.add(Activation('softmax'))
        sys.stdout.flush()
        self.model = model
        return model


    def __use_CNN1D(self):
        keras.backend.clear_session()

        model = Sequential()
        model.add(Conv1D(64, kernel_size=3, padding='same',batch_input_shape=(None, self.train.shape[1], self.train.shape[2])))
        # add output layer
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                        activity_regularizer=tf.keras.regularizers.l1(0.001)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        sys.stdout.flush()
        self.model = model
        return model

    def __use_BiLSTM(self):
        keras.backend.clear_session()

        model = Sequential()
        model.add(Bidirectional(CuDNNLSTM(units=64,
                                          batch_input_shape=(None, self.train.shape[1], self.train.shape[2]),
                                          return_sequences=True,
                                          # True: output at all steps. False: output as last step.
                                          ), merge_mode='concat'))
        # add output layer
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                        activity_regularizer=tf.keras.regularizers.l1(0.001)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        sys.stdout.flush()
        self.model = model
        return model

    def __use_CNN1D_BiLSTM(self):
        keras.backend.clear_session()

        model = Sequential()
        model.add(Conv1D(64, kernel_size=3, padding='same',batch_input_shape=(None, self.train.shape[1], self.train.shape[2])))
        model.add(Bidirectional(CuDNNLSTM(units=64,
                                          return_sequences=True,
                                          ), merge_mode='concat'))
        # add output layer
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                        activity_regularizer=tf.keras.regularizers.l1(0.001)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        self.model = model
        return model

    def __fit_eval(self, e=2, b=32, l=2e-4, beta=0.9):
        # def auc_pred(y_true, y_pred):
        #     y_pred = y_pred[:,1]
        #     y_true_val = K.argmax(y_true,axis=1)
        #     return auc

        class LossHistory(Callback):  # 继承自Callback类
            def __init__(self,validation_x,validation_y, bs):
                self.x_val = validation_x
                self.y_val = validation_y
                self.b = bs
                self.e = 0

            # 在模型开始的时候定义四个属性，每一个属性都是字典类型，存储相对应的值和epoch
            def on_train_begin(self, logs={}):
                self.val_acc = []
                self.val_EI = []
                self.val_pre_score = [[],[],[]]

            # 在每一个epoch之后记录相应的值
            def on_epoch_end(self, batch, logs={}):
                self.e = self.e + 1
                if self.e in [16, 32, 64]:
                    yy_pred = self.model.predict_proba(self.x_val, batch_size=self.b)
                    y_pred = np.argmax(yy_pred, axis=1)
                    true_values = np.array(self.y_val)
                    y_scores = yy_pred[:, 1]
                    
                    if self.e == 32:
                        for x in y_scores:
                            w_data.write(str(x))
                            w_data.write('\n')
                    
                    TN, FP, FN, TP = confusion_matrix(self.y_val, y_pred).ravel()
                    Sensitivity = TP / (TP + FN)
                    Specificity = TN / (TN + FP)
                    ACC = (TP + TN) / (TP + FP + FN + TN)
                    Precision = TP / (TP + FP)
                    F1Score = 2 * TP / (2 * TP + FP + FN)
                    MCC = ((TP * TN) - (FP * FN)) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
                    AUC = roc_auc_score(true_values, y_scores)
                    pre, rec, thresholds = precision_recall_curve(self.y_val, y_scores)
                    prc_area = auc(rec, pre)
                    EI = [self.e,',', TN, FP, FN, TP, Precision, Sensitivity, Specificity, F1Score, MCC, ACC, prc_area, AUC]
                    print('val_auc:{}'.format(AUC))
                    print(self.y_val[0:5], y_pred[0:5])
                    self.val_acc.append(logs.get('val_acc'))
                    self.val_EI.append(EI)
                    self.val_pre_score[0] = self.y_val
                    self.val_pre_score[1] = y_pred
                    self.val_pre_score[2] = yy_pred

        losshistory = LossHistory(self.x_test,self.y_test,self.b)

        adam = Adam(l, beta_1=beta)
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'],)
        self.model.fit(self.train, self.y_train, epochs=e, batch_size=b, verbose=2, validation_split=0, use_multiprocessing=True, workers=4, callbacks=[losshistory])
        self.val_scores = losshistory.val_pre_score
        return losshistory.val_EI

    def train_model(self,model_type, e=2, b=32, l=2e-4, beta=0.9):
        self.e = e
        self.b = b
        self.l = l
        self.beta = beta
        self.model_type = model_type
        print('\nfold{}  此次参数搜索如下：'.format(self.fold))
        print('m_type={} epoch={}  bs={}  l={} beta={}'.format(model_type, e, b, l, beta))
        print('\n训练集数据结构如下：')
        print([self.train.shape[0],self.train.shape[1],self.train.shape[2]], '\n')
        print('average:{}  K:{}'.format(self.average, self.need))

        if model_type == 'FNN':
            self.__use_Fnn()
        elif model_type == 'CNN1D':
            self.__use_CNN1D()
        elif model_type == 'BiLSTM':
            self.__use_BiLSTM()
        elif model_type == 'CNN1D_BiLSTM':
            self.__use_CNN1D_BiLSTM()
        else:
            print('model_type 输入错误..')

        self.EI = self.__fit_eval(e, b, l, beta)

    def predict_EI(self):
        return self.EI


def use_grid_ANN(use_class, model_type_list):
    best_auc = -1
    save_param_EI = []
    save_score = []
    data = pd.read_csv('./param.csv', index_col=False)
    print(data)

    for model_type in model_type_list:
        idx = data.iloc[list(data.iloc[:, 0] == model_type),:].index
        print(idx)
        b = data.iloc[idx, 1].item()
        l = data.iloc[idx, 2].item()
        beta = data.iloc[idx, 3].item()
        e = data.iloc[idx, 4].item()
        print(b, l, e)

        use_class.train_model(model_type=model_type, e=e, b=b, l=l, beta=beta)
        EI = use_class.predict_EI()
        scores = use_class.val_scores
        EI = np.array(EI)
        temp = [[model_type, b, l, beta]] * EI.shape[0]
        temp = np.array(temp)
        save_param_EI_temp = np.concatenate((temp, EI), axis=1)
        save_param_EI.append((save_param_EI_temp.tolist())[-1])
        save_score.append(scores)
    return save_param_EI, save_score

def use_fold_ANN(model_type_list=[],number=[1], train_dir='./', label_dir='./',j=0):

    # bert_type2 = ['BCE-probert', 'probert-base']

    Train_ANN = UseANN(number)

    train_file = train_dir + 'fold-mn/fold1/train.txt'
    label_file = label_dir + 'fold-mn/fold1/train.txt'

    test_file = train_dir + 'fold-mn/fold1/test.txt'
    test_label_file = label_dir + 'fold-mn/fold1/test.txt'

    indp_file = train_dir + 'fold-mn/te-PSAAC-ctd-aac-aap-aat-avg.txt'
    indp_label_file = label_dir + 'fold-mn/te-PSAAC-ctd-aac-aap-aat-avg.txt'

    Train_ANN.get_CLS_data(train_file, label_file)  # 这里决定使用什么CLS还是all_31


    print('训练集数据读取完成...')
    Train_ANN.generate_1D_test(test_file,test_label_file)
    print('数据读取完成...')

    Train_ANN.generate_indp_test(indp_file,indp_label_file)
    print('独立测试集数据读取完成...')

    param_EI,scores = use_grid_ANN(Train_ANN,model_type_list)

    import pandas as pd
    param_EI = pd.DataFrame(param_EI,columns=None,index=None)
    param_EI.reset_index(drop=True)

    param_EI.to_csv('./test-CNN1D_BiLSTM.csv', header=False,index=False)

    print('保存完成...')

    del scores
    del param_EI
    del Train_ANN


if __name__ == '__main__':

    number = [1]
    j = 0

    #model_list = ['FNN', 'CNN1D', 'BiLSTM', 'CNN1D_BiLSTM']
    model_list = ['CNN1D_BiLSTM']

    train_dir = '/mnt/raid5/data3/yfliu/antibacterial_peptide_bert/neuropeptides/combin_fea/neuropeptides/'
    label_dir = '/mnt/raid5/data3/yfliu/antibacterial_peptide_bert/neuropeptides/combin_fea/neuropeptides/'  # 必须以 / 结尾
    w_data = open('./scores/neuro-CNN1D_BiLSTM-test.txt', 'w')    

    print(model_list)

    use_fold_ANN(model_type_list=model_list, number=number, train_dir=train_dir,
                         label_dir=label_dir, j=j)

