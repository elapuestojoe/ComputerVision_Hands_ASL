from keras.preprocessing.image import ImageDataGenerator
import os
# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.applications.mobilenet import MobileNet
from matplotlib import pyplot as plt
import time
from keras.applications.xception import Xception
from keras.callbacks import EarlyStopping

base_model = MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling="max")
top = base_model.output
top = Dropout(0.5)(top)
top = Dense(1024, activation="relu")(top)
top = Dropout(0.5)(top)
top = Dense(1024, activation="relu")(top)
top = Dense(26, activation="softmax")(top)
model = Model(inputs=base_model.input, outputs=top)

model.compile(loss="categorical_crossentropy",
	optimizer=Adam(0.00005),
	metrics=["categorical_accuracy"])

model.summary()

# SET UP IMAGE GENERATOR
train_datagen = ImageDataGenerator(
	rescale=1./255,
	# featurewise_center=True,
	# featurewise_std_normalization=True,
	# rotation_range=15,
	width_shift_range=0.25,
	height_shift_range=0.25,
	# horizontal_flip=True,
	# brightness_range=(0.5, 1.0)
	)

train_generator = train_datagen.flow_from_directory(
	'asl-alphabet/asl_alphabet_train_SAVE',
	target_size=(224, 224),
	batch_size=8,
	class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
	"asl-alphabet/test_dataset",
	target_size=(224, 224),
	batch_size=8,
	class_mode="categorical"
)

def saveModel(model, modelName):
	modelJson = "{}.json".format(modelName)
	modelH5 = "{}.h5".format(modelName)
	print("saveModel")
	model_json = model.to_json()
	with open(modelJson, "w") as json_file:
		json_file.write(model_json)
	#seralize weights to HDF5
	model.save_weights(modelH5)
	print("Saved model to disk")

def saveModelHistory(history):
	plt.plot(history.history["categorical_accuracy"])
	plt.plot(history.history["val_categorical_accuracy"])
	plt.title("Training Accuracy")
	plt.ylabel("categorical_accuracy")
	plt.xlabel("epoch")
	plt.legend(["train"], loc="upper left")
	plt.savefig("{}.png".format(time.time()))

def loadModel(modelName):
	print("loadModel")

	base_model = MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling="max")
	top = base_model.output
	top = Dense(1024, activation="relu")(top)
	top = Dense(26, activation="softmax")(top)
	model = Model(inputs=base_model.input, outputs=top)

	#load weights
	modelH5 = "{}.h5".format(modelName)
	model.load_weights(modelH5)
	model.compile(loss="categorical_crossentropy", optimizer=Adam(0.0001), metrics=["categorical_accuracy"])
	# model.summary()
	return model

earlystop = EarlyStopping(monitor='categorical_accuracy', min_delta=0.0001, patience=5, \
                          verbose=1, mode='auto')
callbacks_list = [earlystop]
# model = loadModel("Mobilenet2")
history = model.fit_generator(
		train_generator,
		steps_per_epoch=1024,
		epochs=3,
		validation_data=test_generator,
		callbacks=callbacks_list
		)

saveModel(model, "MobileNet")


saveModelHistory(history)

# # TEST:
# DIR = "asl-alphabet/asl_alphabet_test/"
# arr = []
# for file in os.listdir(DIR)
# 	img = cv2.imread(DIR+file)
# 	img = cv2.resize(img, (224, 224))

