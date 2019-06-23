#!/usr/bin/env python
# coding: utf-8

# # LSTM text generation

# Example script to generate text from text files.
# 
# At least 20 epochs are required before the generated text starts sounding coherent.
# 
# It is recommended to run this script on GPU, as recurrent networks are quite computationally intensive.
# 
# If you try this script on new data, make sure your corpus has at least ~100k characters. ~1M is better.

# In[1]:


from keras.callbacks import LambdaCallback
from keras.callbacks import EarlyStopping

from keras.models import Model, load_model

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import concatenate

from keras.optimizers import Adam
from keras.utils import Sequence
from keras.utils.data_utils import get_file

import numpy as np
import random
import io

import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

import logging
logging.getLogger('tensorflow').disabled = True


# In[2]:


class DataGenerator(Sequence):
    def __init__(self, text, char_indices, batch_size=128, maxlen=40, step=3):
        self.text = text
        self.char_indices = char_indices
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.step = step

    def __len__(self):
        return ((len(self.text) - self.maxlen) // self.step) // self.batch_size

    def __getitem__(self, index):
        x = np.zeros((self.batch_size, self.maxlen, len(self.char_indices)), dtype=np.bool)
        y = np.zeros((self.batch_size, len(self.char_indices)), dtype=np.bool)

        for i in range(self.batch_size):
            idx = (i + index) * self.step

            for t, char in enumerate(self.text[idx: idx + self.maxlen]):
                x[i, t, self.char_indices[char]] = 1

            y[i, self.char_indices[self.text[idx + self.maxlen]]] = 1

        return x, y


# In[3]:


path = get_file(
    'lovecraft.txt',
    origin='https://bashkirtsevich.pro/shared/lovecraft.txt'
)

with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# In[4]:


maxlen = 40  # cut the text in semi-redundant sequences of maxlen characters
training_generator = DataGenerator(text, char_indices, maxlen=maxlen)


# ## Build the model: a single LSTM

# In[5]:


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print(f'----- Generating text after Epoch: {epoch + 1}')

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print(f'----- diversity: {diversity}')

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print(f'----- Generating with seed: "{sentence}"')

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

        print(generated)
        
    model.save(f"model.h5")


# In[6]:


# build/load the model: a single LSTM
model_path = None

if model_path:
    print('Load model...')
    model = load_model(model_path)
else:
    print('Build model...')

    num_chars = len(chars)

    vec = Input(shape=(maxlen, num_chars))
    l1 = LSTM(activation='tanh', return_sequences=True, units=128)(vec)
    l1_d = Dropout(0.2)(l1)

    input2 = concatenate([vec, l1_d])
    l2 = LSTM(activation='tanh', return_sequences=True, units=128)(input2)
    l2_d = Dropout(0.2)(l2)

    input3 = concatenate([vec, l2_d])
    l3 = LSTM(activation='tanh', return_sequences=True, units=128)(input3)
    l3_d = Dropout(0.2)(l3)

    input_d = concatenate([l1_d, l2_d, l3_d])
    
    l4 = LSTM(activation='tanh', return_sequences=False, units=128)(input_d)
    l4_d = Dropout(0.2)(l4)
    
    dense3 = Dense(units=num_chars)(l4_d)
    output_res = Activation('softmax')(dense3)
    
    model = Model(inputs=vec, outputs=output_res)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(clipnorm=1.), metrics=['accuracy'])
    


# In[7]:


model.summary()


# In[ ]:


model.fit_generator(
    generator=training_generator,
    validation_data=training_generator,
    epochs=60,
    callbacks=[
        LambdaCallback(on_epoch_end=on_epoch_end),
        EarlyStopping(monitor="loss", min_delta=0.001, patience=3, mode="min")
    ],
)


# In[ ]:




