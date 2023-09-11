import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random as rn

np.random.seed(123)
rn.seed(123)
tf.random.set_seed(123)

data_path = '/data'
X = np.load('speechx.npy')
y = np.load('speechy.npy')

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=10,
                                             restore_best_weights=True)
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)

def model_dense():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(axis=-1,
              input_shape=(x_train.shape[1:])))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(16, activation='softmax'))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

model = model_dense()
print(model.summary())

tf.keras.utils.plot_model(model,to_file='mlp_model.pdf',show_shapes=True)

hist = model.fit(x_train,
                 y_train,
                 epochs=100,
                 shuffle=True,

                 validation_split=0.1,
                 batch_size=16)

evaluate = model.evaluate(x_test, y_test, batch_size=16)
print("Loss={:.6f}, Accuracy={:.6f}".format(evaluate[0], evaluate[1]))

plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.grid()
plt.legend(['Training', 'Validation'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('mlp model')
plt.savefig('mlp hasil.svg')

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
predict = model.predict(x_test, batch_size=16)
emotions = ['natural', 'kalem', 'senang', 'sedih', 'marah', 'tenang', 'jijik', 'terkejut']

y_pred = np.argmax(predict, 1)
predicted_emo = []
for i in range(0,y_test.shape[0]):
    emo = emotions[y_pred[i]]
    predicted_emo.append(emo)

actual_emo=[]
y_true = y_test
for i in range(0, y_test.shape[0]):
    emo = emotions[y_true[i]]
    actual_emo.append(emo)

cm = confusion_matrix(actual_emo, predicted_emo)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

index = ['natural', 'kalem', 'senang', 'sedih', 'marah', 'tenang', 'jijik', 'terkejut']
columns = ['natural', 'kalem', 'senang', 'sedih', 'marah', 'tenang', 'jijik', 'terkejut']
cm_df = pd.DataFrame(cm, index, columns)
plt.figure(figsize=(10, 6))
plt.title('Confusion matriks MLP')
sns.heatmap(cm_df, annot=True)
plt.savefig('Confusion Matriks MLP.svg')

print("UAR", cm.trace()/cm.shape[0])