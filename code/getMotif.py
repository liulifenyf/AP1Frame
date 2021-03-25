# %%
from keras.models import load_model, Model
import numpy as np
import pandas as pd


def load_data():
    x = np.load('../data/AP1sgRNA20.npy')
    x = x.reshape([x.shape[0], x.shape[1], x.shape[2], 1])
    y = np.array(
        pd.read_csv('../data/labels.csv',
                    header=None).values).reshape([x.shape[0], 1])
    return x, y


def NUMPY2STRING(input_array):

    # convert numpy to string for 2 dimension numpy array.

    output_str = ""

    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            output_str = output_str + str(input_array[i, j]) + "\t"
        output_str += "\n"

    return output_str


def matrix2meme(pwm_txt_name, pwm_meme_name, pwm_len):

    # convert PWM to meme format used in tomtom.

    write_ofl = open(pwm_meme_name, "w")  #####

    write_ofl.write("MEME version 5.0.4\n\n")
    write_ofl.write("ALPHABET= ACGT\n\n")
    write_ofl.write("strands: + -\n\n")
    write_ofl.write("Background letter frequencies\n")
    write_ofl.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")

    read_ofl = open(pwm_txt_name)  #####
    oflst = read_ofl.readlines()
    read_ofl.close()

    count = 0
    for line in oflst:
        line = line.strip()
        if re.search(">", line):
            write_ofl.write("\n")
            write_ofl.write("MOTIF" + "\t" + "filter" + str(count + 1) + "\n")
            write_ofl.write("letter-probability matrix: alength= 4 w= " +
                            str(pwm_len) + "\n")  #######
            count += 1
        else:
            write_ofl.write(line + "\n")
    write_ofl.close()


X, Y = load_data()
pwm_meme_name = "/home/lfliu/workspace/regulation/AP1/result/pwm0-8.meme"
pwm_txt_name = "/home/lfliu/workspace/regulation/AP1/result/pwm0-8.txt"
model = load_model('../Attention_CNN_model.h5')


layer_output = Model(inputs=model.input, outputs=model.layers[1].output)
conv_out = layer_output.predict(X)
conv_out = conv_out.reshape([conv_out.shape[0], conv_out.shape[1], 16])
WEIGHTS, BIAS = model.get_layer('conv1').get_weights()
WEIGHTS = WEIGHTS.reshape(
    [WEIGHTS.shape[0], WEIGHTS.shape[1], WEIGHTS.shape[3]])
INSTANCE_LENGTH = WEIGHTS.shape[0]
X = X.reshape([X.shape[0], X.shape[1], X.shape[2]])
print(conv_out.shape)


# 用于提取pwm矩阵
import re
motif_ofl = open(pwm_txt_name, "w")
count = []
for i in range(WEIGHTS.shape[2]):
    one_filter_weight = WEIGHTS[:, :, i]
    THRESHOLD = (np.sum(np.max(one_filter_weight, 1))+ BIAS[i]) * 0.5
    model_c = conv_out[:,:,i] - BIAS[i]
    position_m = np.where(model_c >= THRESHOLD)
    top_num = position_m[0][position_m[0]<48].shape[0]
    end_num = position_m[0][position_m[0]>680].shape[0]
    count.append([top_num, end_num])
    INSTANCE_FILTERED_NUMBER = position_m[0].shape[0]
    print(INSTANCE_FILTERED_NUMBER)
    INSTANCE_FILTERED = np.zeros(
        [INSTANCE_FILTERED_NUMBER, INSTANCE_LENGTH, 4])
    for j in range(INSTANCE_FILTERED_NUMBER):
        INSTANCE_FILTERED[j] = X[position_m[0][j],
                                 (position_m[1][j]):(position_m[1][j] +
                                                   INSTANCE_LENGTH), :]
    pwm_matrix = np.mean(INSTANCE_FILTERED, 0)
    pwm_sum = np.sum(pwm_matrix, -1)
    for col in range(pwm_matrix.shape[1]):
        pwm_matrix[:, col] = pwm_matrix[:, col] / pwm_sum
    outline = ">MOTIF" + str(i + 1) + "\n"
    motif_ofl.write(outline)
    outline = NUMPY2STRING(pwm_matrix)
    motif_ofl.write(outline)
    motif_ofl.flush()
motif_ofl.close()
print("PWM-txt Done")

matrix2meme(pwm_txt_name, pwm_meme_name, INSTANCE_LENGTH)
print("PWM-meme Done")
