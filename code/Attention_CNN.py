import numpy as np
from tensorflow.keras.layers import Input, add, Dense, Activation, GlobalAveragePooling2D, Flatten, Conv2D, Reshape, MaxPooling2D, Dropout,multiply
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import  train_test_split
from scipy.stats import pearsonr
import pandas as pd


def one_hot(seq):
	# define universe of possible input values
	seq = seq.strip().upper()
	alphabet = 'ACGT'
	onehot_encoded = []
	char_to_int = dict((c, i) for i, c in enumerate(alphabet))
	int_to_char = dict((i, c) for i, c in enumerate(alphabet))
	other_letters = []
	for s in seq:
		if not alphabet.count(s):
			other_letters.append(s)
			flag = True
	if not other_letters:
		integer_encoded = [char_to_int[char] for char in seq]
		for value in integer_encoded:
				letter = [0 for _ in range(len(alphabet))]
				letter[value] = 1
				onehot_encoded.append(letter)
	return np.array(onehot_encoded)


def load_data():
    fa_file_lath = "../data/AP1sgRNA20.fa"
    fasta_file = open(fa_file_lath).readlines()
    seqs = [seq_num for seq_num in fasta_file if not seq_num.startswith('>')]
    x = np.array(map(one_hot,seqs))
    x = x.reshape([x.shape[0], x.shape[1], x.shape[2], 1])
    y = np.array(
        pd.read_csv('../data/labels.csv',
                    header=None).values).reshape([x.shape[0], 1])

    X, X_test, Y, Y_test = train_test_split(x,
                                            y,
                                            test_size=0.2,
                                            random_state=123456)

    print(X.shape, Y.shape)
    return X, X_test, Y, Y_test


def se_block(input_feature, ratio = 4):
	channel = input_feature.shape[-1]
	# Squeeze: H*W*C压缩为1*1*C，大小的特征图，这个特征图可以理解为具有全局感受野
	se_feature = GlobalAveragePooling2D()(input_feature) 
	se_feature = Reshape((1, 1, channel))(se_feature)

	assert se_feature.shape[1:] == (1, 1, channel)
	# Excitation: 使用一个全连接神经网络，对Squeeze之后的结果做一个非线性变换，得到不同channel的重要性大小
	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature) 
	assert se_feature.shape[1:] == (1, 1, channel//ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1, 1, channel)
	# 特征重标定: 使用 Excitation 得到的结果作为权重，乘到输入特征上
	se_feature = multiply([input_feature, se_feature]) # multiply()将输入的list按顺序相乘

	return se_feature


class EarlyStoppingAtMaxPCC(Callback):
    def __init__(self, validation_data):
        self.patience= 0
        super(EarlyStoppingAtMaxPCC, self).__init__()
        self.best_weights = None
        self.x_val = validation_data[0]
        self.y_val = validation_data[1].ravel()
        

    def on_train_begin(self, logs=None):
            # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = -np.inf


    def on_epoch_end(self, epoch, logs=None):
        y_pred_val = self.model.predict(self.x_val).ravel()
        cur_pcc,_ = pearsonr(self.y_val, y_pred_val)
        print('\tpcc_val: %s' % (str(round(cur_pcc,4))),end=100*' '+'\n')     
        if np.less(self.best, cur_pcc):
            self.best = cur_pcc
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)


    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def Attention_CNN(input_shape=(20, 4, 1), filters=16, conv_1=7):
    X_input = Input(input_shape)

    # stage1
    X = Conv2D(filters=filters,
               kernel_size=(conv_1, 4),
               strides=(1, 1),
               name="conv1",
               padding = 'valid',
               activation = 'relu',
               kernel_initializer= RandomNormal(stddev=0.05)
               )(X_input)
    #X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    y = se_block(X)
    X = add([X, y])
    X = MaxPooling2D(pool_size=(2, 1))(X)
    # 输出层
    X = Flatten()(X)
    X = Dense(128,
              activation='relu',
              kernel_initializer=RandomNormal(stddev=0.05)
              )(X)
    X = Dropout(0.4)(X)
    X = Dense(1,
              name='fc',
              activation='linear',
              kernel_initializer=RandomNormal(stddev=0.05)
              )(X)

    model = Model(inputs=X_input, outputs=X, name='simplenet')

    return model


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    X, X_test, Y, Y_test = load_data()
    batch_size = 16
    epochs = 200
    plt.figure(5,5)
    x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1,random_state=2021)
    model = Attention_CNN()
    model.compile(loss='mse', optimizer=Adam(lr=0.01))
    model.fit(x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_valid, y_valid),
                verbose=0,
                callbacks=[
                    EarlyStoppingAtMaxPCC(validation_data=[x_valid, y_valid])
                    ]
            )
    model.save('../model/retrained.h5')
    # 检查模型预测效果
    y_train_pred = model.predict(X) # 训练集预测值
    y_test_pred = model.predict(X_test) # 测试集预测值
    index_train = Y[:,0].argsort()[::-1] # 返回被打乱的数据集 的由大到小排列的索引
    index_test =Y_test[:,0].argsort()[::-1]
    sgRNA71 = (Y[index_train][2], y_train_pred[index_train][2])
    sgRNA69 = (Y_test[index_test][1], y_test_pred[index_test][1])

    plt.scatter(Y.ravel(), y_train_pred.ravel(), s = 3, c = 'darkgrey')
    plt.scatter(Y_test.ravel(), y_test_pred.ravel(),s = 3, c = 'black')
    plt.scatter(sgRNA71[0].ravel(), sgRNA71[1].ravel(), s =7, c = 'r')
    plt.scatter(sgRNA69[0].ravel(), sgRNA69[1].ravel(), s =7, c = 'g')
    plt.show()
    pcc_train = pearsonr(Y.ravel(), y_train_pred.ravel())[0]
    pcc_test = pearsonr(Y_test.ravel(), y_test_pred.ravel())[0]
    pcc_top = pearsonr(Y_test[index_test][:9].ravel(),
                        y_test_pred[index_test][:9].ravel())[0]
    print(pcc_train, pcc_test, pcc_top)

print('Done')
