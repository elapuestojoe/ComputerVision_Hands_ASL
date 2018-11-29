from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import numpy as np

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
from keras.models import model_from_json
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception

LABELS = ["A", "B", "C", "D", "del", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "nothing", "O", "P", "Q", "R", "S", 
"space", "T", "U", "V", "W", "X", "Y", "Z"]

detection_graph, sess = detector_utils.load_inference_graph()
sess = tf.Session(graph=detection_graph)
cap = cv2.VideoCapture(0)
# cap.set(cv2.CV_CAP_PROP_FPS, 10)
score_thresh = 0.2
frame = 0

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,200)
fontScale              = 2
fontColor              = (255,0,0)
lineType               = 2

def loadModel(modelName):
	print("loadModel")

	base_model = MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False, pooling="max")
	top = base_model.output
	top = Dense(29, activation="softmax")(top)
	model = Model(inputs=base_model.input, outputs=top)

	#load weights
	modelH5 = "{}.h5".format(modelName)
	model.load_weights(modelH5)

	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
	# model.summary()
	return model

model = loadModel("model")

while 1:
	ret, img = cap.read()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	height, width, channels = img.shape

	boxes, scores = detector_utils.detect_objects(
					img, detection_graph, sess)

	cropped_images = detector_utils.draw_box_on_image(
					2, score_thresh,
					scores, boxes, height, width,
					img)

	# for i in range(len(cropped_images)):
	# 	print(cropped_images[i].shape)
	# 	cv2.imshow("HAND: "+str(i), cropped_images[i])
	TEXT = ""
	if(len(cropped_images) > 0):
		cropped_images = np.array(cropped_images)
		results = model.predict(cropped_images)

		for result in results:
			maxIndex = np.argmax(result)
			if(result[maxIndex] > 0.85):
				TEXT += LABELS[maxIndex]# + str(maxIndex)

		cv2.putText(img,TEXT, 
			bottomLeftCornerOfText, 
			font, 
			fontScale,
			fontColor,
			lineType)

	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	cv2.imshow("Original", img)

	k = cv2.waitKey(30)
	if(k == 27):
		break
	if(k == 115):
		print("SAVE")
		cv2.imwrite("output_images/"+TEXT+".png", img)

