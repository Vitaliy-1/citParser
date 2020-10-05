import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import TextVectorization

from DataProcessor import DataProcessor
from matplotlib import pyplot as plt
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
import tensorflowjs as tfjs

# Params
BATCH_SIZE = 512  # Number of examples used in each iteration
EPOCHS = 100  # Number of passes through entire dataset
EMBEDDING = 40  # Dimension of word embedding vector

# importing the data
dir_path = 'annotated/corpus'
dataProcessor = DataProcessor(dir_path, 'tei')
sentences = dataProcessor.getListOfTuples()

word2idx = {w: i + 2 for i, w in enumerate(dataProcessor.getWords())}
word2idx['unk'] = 1
word2idx['pad'] = 0

idx2word = {i: w for w, i in word2idx.items()}

tag2idx = {t: i + 1 for i, t in enumerate(dataProcessor.getTags())}
tag2idx['pad'] = 0

idx2tag = {i: w for w, i in tag2idx.items()}

# Write dictionary
import json
with open('model4_js/vocab/word2idx.json', 'w') as fp:
    json.dump(word2idx, fp)
with open('model4_js/vocab/idx2word.json', 'w') as fp:
    json.dump(idx2word, fp)
with open('model4_js/vocab/idx2tag.json', 'w') as fp:
    json.dump(idx2tag, fp)

from keras.preprocessing.sequence import pad_sequences

# Convert each sentence from list of Token to list of word_index
X = [[word2idx[w[0]] for w in s] for s in sentences]
# Padding each sentence to have the same lenght
X = pad_sequences(maxlen=dataProcessor.getMaxLength(), sequences=X, padding='post', value=word2idx['pad'])

# Convert Tag/Label to tag_index
y = [[tag2idx[w[1]] for w in s] for s in sentences]
# Padding each sentence to have the same lenght
y = pad_sequences(maxlen=dataProcessor.getMaxLength(), sequences=y, padding='post', value=tag2idx['pad'])

from keras.utils import to_categorical

# One-Hot encode
y = [to_categorical(i, num_classes=len(dataProcessor.getTags()) + 1) for i in y]  # n_tags+1(PAD)

from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
X_tr.shape, X_te.shape, np.array(y_tr).shape, np.array(y_te).shape

# Model definition

model = keras.Sequential()
model.add(layers.Input(shape=(dataProcessor.maxLength,)))
model.add(layers.Embedding(input_dim=len(dataProcessor.getWords()) + 2, output_dim=EMBEDDING,  # n_words + 2 (PAD & UNK)
                           input_length=dataProcessor.maxLength, mask_zero=True))
model.add(layers.Bidirectional(layers.LSTM(units=50, return_sequences=True, recurrent_dropout=0.1)))
model.add(layers.TimeDistributed(layers.Dense(50, activation="relu")))
model.add(layers.Dense(len(dataProcessor.getTags()) + 1, activation='sigmoid'))  # number of tags/labels plus 'pad'
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])  # seems like categorical_crossentropy is sutable for the tasks with multiple labels

history = model.fit(X_tr, np.array(y_tr), batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_split=0.2, verbose=2)


# Eval
pred_cat = model.predict(X_te)
pred = np.argmax(pred_cat, axis=-1)
y_te_true = np.argmax(y_te, -1)

from sklearn_crfsuite.metrics import flat_classification_report

# Convert the index to tag
pred_tag = [[idx2tag[i] for i in row] for row in pred]
y_te_true_tag = [[idx2tag[i] for i in row] for row in y_te_true]

report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
print(report)

model.save('model4')
tfjs.converters.save_keras_model(model, 'model4_js')

# i = np.random.randint(0,X_te.shape[0]) # choose a random number between 0 and len(X_te)
# Testing on evaluation data
for i, value in enumerate(X_te):
    if i == 20:
        break
    p = model.predict(np.array([X_te[i]]))
    p = np.argmax(p, axis=-1)
    true = np.argmax(y_te[i], -1)

    print("Sample number {} of {} (Test Set)".format(i, X_te.shape[0]))
    # Visualization
    print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
    print(30 * "=")
    for w, t, pred in zip(X_te[i], true, p[0]):
        if w != 0:
            print("{:15}: {:5} {}".format(dataProcessor.getWords()[w - 2], idx2tag[t], idx2tag[pred]))

# Example of the prediction
# model = keras.models.load_model('model4')
newData = [
    'Baile W, Buckman R, Lenzi R, Glober G, Beale E, Kudelka A. SPIKES—A Six-Step Protocol for Delivering Bad News: Application to the Patient with Cancer. The Oncologist. 2000;5(4):302-11.',
    'Patel K, Tatham K. Complete OSCE Skills for Medical and Surgical Finals. London: Hodder Arnold; 2010.',
    'Burton N, Birdi K. Clinical Skills for OSCEs. London: Informa; 2006.']

newDataTokenized = []
for textString in newData:
    newDataTokenized.append(
        text_to_word_sequence(textString, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=False))

test_sample = []
for item in newDataTokenized:
    vectorizedItem = []
    for word in item:
        number = word2idx.get(word)
        if number is not None:
            vectorizedItem.append(number)
        else:
            vectorizedItem.append(word2idx['unk'])
    test_sample.append(vectorizedItem)

test_sample = pad_sequences(maxlen=dataProcessor.getMaxLength(), sequences=test_sample, padding='post',
                            value=word2idx['pad'])

for i, value in enumerate(test_sample):
    p = model.predict(np.array([test_sample[i]]))
    p = np.argmax(p, axis=-1)
    print('--------------------------------------------------')
    for index, test_sample_item_number in enumerate(test_sample[i]):
        if test_sample_item_number == 0:
            continue
        predictedLabel = p[0][index]
        print(idx2word[test_sample_item_number], idx2tag[predictedLabel])
