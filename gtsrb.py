import pickle
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
# from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import tf2onnx
from myVeriX2 import *
import time
import sys

"""
load and process GTSRB data.
"""
gtsrb_path = 'models/gtsrb.pickle'
with open(gtsrb_path, 'rb') as handle:
    gtsrb = pickle.load(handle)
x_train, y_train = gtsrb['x_train'], gtsrb['y_train']
x_test, y_test = gtsrb['x_test'], gtsrb['y_test']
x_valid, y_valid = gtsrb['x_valid'], gtsrb['y_valid']
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
y_valid = to_categorical(y_valid, 10)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_valid = x_valid.astype('float32') / 255
gtsrb_labels = ['50 mph', '30 mph', 'yield', 'priority road',
                'keep right', 'no passing for large vechicles', '70 mph', '80 mph',
                'road work', 'no passing']

"""
show a simple example usage of VeriX. 
"""
np.random.seed(42)
indices = np.random.choice(x_test.shape[0], 1000)[:10]
print(indices)

tot_sat_len = 0
tot_timeout_len = 0
i = 0
for idx in indices:
    time_start = time.time()
    verix = VeriX(dataset="GTSRB",
                name=idx,
                image=x_test[idx],
                model_path="models/gtsrb-10x2.onnx")
    verix.traversal_order(traverse="heuristic")
    len_sat_set, len_timeout_set = verix.get_explanation(epsilon=0.01)
    time_end = time.time()
    tot_sat_len += len_sat_set
    tot_timeout_len += len_timeout_set
    print(f"{idx} {time_end - time_start} {len_sat_set}", file=sys.stdout, flush=True)
    i += 1
    if i == 100:
        exit()


verix = VeriX(dataset="GTSRB",
              image=x_test[0],
              model_path="models/gtsrb-10x2.onnx")
verix.traversal_order(traverse="heuristic")
verix.get_explanation(epsilon=0.01)

exit()

"""
or you can train your own GTSRB model.
Note: to obtain sound and complete explanations, train the model from logits directly.
 """
model_name = 'gtsrb-10x2'
model = Sequential(name=model_name)
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10))
model.summary()
model.compile(loss=CategoricalCrossentropy(from_logits=True),
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
datagen = ImageDataGenerator()
model.fit(datagen.flow(x=x_train, y=y_train, batch_size=64),
          steps_per_epoch=100,
          epochs=30,
          validation_data=(x_valid, y_valid),
          shuffle=1)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
# model.save('models/' + model_name + '.h5')
model_proto, _ = tf2onnx.convert.from_keras(model, output_path='models/' + model_name + '.onnx')