from tensorflow import keras
import numpy as np
from tensorflow.python.keras.layers import TextVectorization

from DataProcessor import DataProcessor

model = keras.models.load_model('model3/model.hd5')

dataProcessor = DataProcessor('dataToTest/data.txt', 'text')

string_input = keras.Input(shape=(1,), dtype="string")
vectorizer = TextVectorization(output_sequence_length=148)
x = vectorizer(string_input)
preds = model(x)
end_to_end_model = keras.Model(string_input, preds)

probabilities = end_to_end_model.predict(
    [['Baile W, Buckman R, Lenzi R, Glober G, Beale E, Kudelka A. SPIKES—A Six-Step Protocol for Delivering Bad News: Application to the Patient with Cancer. The Oncologist. 2000;5(4):302-11.']]

)


"""
i = np.random.randint(0,X_te.shape[0]) # choose a random number between 0 and len(X_te)
p = model.predict(np.array([X_te[i]]))
p = np.argmax(p, axis=-1)
true = np.argmax(y_te[i], -1)

print("Sample number {} of {} (Test Set)".format(i, X_te.shape[0]))
# Visualization
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_te[i], true, p[0]):
    if w != 0:
        print("{:15}: {:5} {}".format(dataProcessor.getWords()[w-2], idx2tag[t], idx2tag[pred]))
"""