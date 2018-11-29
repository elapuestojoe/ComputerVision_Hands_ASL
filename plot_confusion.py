from keras.layers.core import Dense
from keras.models import Model
from keras.applications.mobilenet import MobileNet
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.applications.xception import Xception
import pandas as pd

def loadModel(modelName):
	print("loadModel")

	base_model = Xception(input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling="max")
	top = base_model.output
	top = Dropout(0.5)(top)
	top = Dense(1024, activation="relu")(top)
	top = Dropout(0.5)(top)
	top = Dense(1024, activation="relu")(top)
	top = Dense(26, activation="softmax")(top)
	model = Model(inputs=base_model.input, outputs=top)

	#load weights
	modelH5 = "{}.h5".format(modelName)
	model.load_weights(modelH5)
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
	# model.summary()
	return model

test_data_path = "asl-alphabet/validation_dataset"
test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
    test_data_path, 
    target_size=(224, 224),
    batch_size=8, 
    class_mode='categorical',
	shuffle=False)

model = loadModel("Xception")
print(validation_generator.class_indices)
labels = [""]*26
for key in validation_generator.class_indices.keys():
	labels[validation_generator.class_indices[key]] = key

Y_pred = model.evaluate_generator(validation_generator)
print(model.metrics_names)
print("YPRED", Y_pred)

predictions = model.predict_generator(validation_generator)
y_pred = np.argmax(predictions, axis=1)

cf = confusion_matrix(validation_generator.classes, y_pred)
pd_cf = pd.DataFrame(data = cf, index = labels, columns = labels)

print(cf)
print("Classification Report")

print(classification_report(validation_generator.classes, y_pred))
import seaborn as sns; sns.set()
ax = sns.heatmap(pd_cf, fmt="d", annot=True)
plt.title("Xception hand recognition validation confusion matrix")
plt.show()

from keras.utils import plot_model
plot_model(model, to_file='XceptionArch.png')