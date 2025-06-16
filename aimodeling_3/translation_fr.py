import os
import shutil
import urllib3
import zipfile
import pandas as pd
import requests

# 데이터 파일 읽어오기
filename = "/Users/junyoung/study/로봇공학/fra-eng.zip"
path = os.getcwd()
zipfilename = os.path.join(path, filename)

# 압축을 풀어주기
with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)

# 파일을 읽어서 컬럼name을 부여하고 불필요한 컬럼 lic는 삭제
lines = pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']

# 데이터의 일부만 사용하고 시작과 끝은 나타내는 토큰으로 '\t', '\n' 을 사용
lines = lines.loc[:, 'src':'tar']
lines = lines[0:30000]
lines.tar = lines.tar.apply(lambda x : '\t' + x + '\n')
print(lines[:5])

src_vocab = set()
for line in lines.src:
    for char in line:
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)

src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))

src_vocab_size = len(src_vocab)+1
tar_vocab_size = len(tar_vocab)+1

src_to_idx = dict([(word, i+1) for i, word in enumerate(src_vocab)])
tar_to_idx = dict([(word, i+1) for i, word in enumerate(tar_vocab)])

print(src_to_idx)
print(tar_to_idx)

encoder_input = []
for line in lines.src:
    encoder_input.append([src_to_idx[w] for w in line])
print('encoder_input=', encoder_input[:5])

decoder_input = []
for line in lines.tar:
    decoder_input.append([tar_to_idx[w] for w in line])
print('decoder_input=', decoder_input[:5])

decoder_target = []
for line in lines.tar:
    decoder_target.append([tar_to_idx[w] for w in line if w != '\t'])
print('decoder_target=', decoder_target[:5])

from tensorflow.keras.preprocessing.sequence import pad_sequences

max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])
encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

from tensorflow.keras.utils import to_categorical

encoder_onehot_input = to_categorical(encoder_input)
decoder_onehot_input = to_categorical(decoder_input)
decoder_onehot_target = to_categorical(decoder_target)

from keras.layers import Input, LSTM, Dense
encoder_inputs = Input(shape=(None, src_vocab_size))
print('encoder_inputs=', encoder_inputs.shape)
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
print('encoder_outputs=', encoder_outputs.shape)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, tar_vocab_size))
print('decoder_inputs=', decoder_input.shape)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)
print('decoder_output=', decoder_outputs.shape)

from keras.models import Model

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=[encoder_onehot_input, decoder_onehot_input], y=decoder_onehot_target, batch_size=128, epochs=25, validation_split=0.2)

print('2482018 최준영')

import numpy as np

encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
decoder_state_input_h = Input(shape=(256, ))
decoder_state_input_c = Input(shape=(256, ))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

decoder_softmax_outputs = decoder_softmax_layer(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_softmax_outputs]+ decoder_states)

idx_to_src = dict((i, char) for i, char in src_to_idx.items())
idx_to_tar = dict((i, char) for i, char in tar_to_idx.items())

def predict_decode(input_seq):
    states_values = encoder_model.predict(input_seq)
    target_seq = np.zeros((1,1, tar_vocab_size))
    target_seq[0, 0, tar_to_idx['\t']] = 1
    stop = False
    decoded_sentence = ""

    while not stop:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_values, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        sampled_char = idx_to_tar[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_tar_len:
            stop = True

        target_seq = np.zeros((1,1, tar_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1.0

        states_values = [h, c]

    return decoded_sentence


for seq_index in [0, 40, 300, 1000]:
    input_seq = encoder_onehot_input[seq_index:seq_index+1]
    decoded_sentence = predict_decode(input_seq)

    print("입력 : ", lines.src[seq_index])

    print("정답 : ", lines.tar[seq_index][1:len(lines.tar[seq_index])-1])
    print("번역 : ", decoded_sentence[:len(decoded_sentence)-1],'\n')

print('2482018 최준영')
