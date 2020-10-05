from tensorflow import keras
import numpy as np
from tensorflow.python.keras.layers import TextVectorization

from DataProcessor import DataProcessor
import json

model = keras.models.load_model('model4')

string_input = keras.Input(shape=(1,), dtype="string")
vectorizer = TextVectorization(output_sequence_length=150)
x = vectorizer(string_input)
preds = model(x)
end_to_end_model = keras.Model(string_input, preds)

test_sample = [
        'Baile W, Buckman R, Lenzi R, Glober G, Beale E, Kudelka A. SPIKES—A Six-Step Protocol for Delivering Bad News: Application to the Patient with Cancer. The Oncologist. 2000;5(4):302-11.',
        'Patel K, Tatham K. Complete OSCE Skills for Medical and Surgical Finals. London: Hodder Arnold; 2010.',
        'Burton N, Birdi K. Clinical Skills for OSCEs. London: Informa; 2006.']


p = end_to_end_model.predict(np.array(test_sample))
p = np.argmax(p, axis=-1)
print(p)
#for sNumber, sentence in enumerate(p):

