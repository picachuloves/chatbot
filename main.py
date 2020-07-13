import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import pickle
import numpy as np
from preproc import preprocess_text

intents = pd.read_json('intents.json')

texts = []
labels = []
varios = []
answers = []

for intent in intents['intents']:
    texts.extend(intent['patterns'])
    for i in range(len(intent['patterns'])):
        labels.append(intent['tag'])
    if intent['tag'] not in varios:
        varios.append(intent['tag'])
    answers.append(intent['responses'])

pickle.dump(varios, open('varios.pkl', 'wb'))
pickle.dump(answers, open('answers.pkl', 'wb'))

texts_p = [preprocess_text(t) for t in texts]

vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(texts_p)
pickle.dump(vectorizer, open("feature.pkl", "wb"))

output_empty = [0] * len(varios)
labels_n = []
for label in labels:
    output_row = list(output_empty)
    output_row[varios.index(label)] = 1
    labels_n.append(output_row)


model = Sequential()
model.add(Dense(128, input_shape=(vectors.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(labels_n[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

vectors = vectors.toarray()
vectors = np.array(vectors)
labels_n = np.array(labels_n)

model.fit(vectors, labels_n, epochs=200, batch_size=4, verbose=1)
model.save('chatbot_model.h5')
print("model created")





