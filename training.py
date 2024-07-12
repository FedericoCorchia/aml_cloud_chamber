import os
import sys
import json
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout, RandomFlip, RandomRotation, RandomTranslation
from keras.preprocessing import image
from keras.metrics import Precision, Recall

#definition of folders with training and valid samples
dataDir = str(sys.argv[1])
trainDir = os.path.join(dataDir, "train")
validDir = os.path.join(dataDir, "valid")
print("dataDir:", dataDir)

#pixel rescaling from [0, 255] to [0, 1]
trainImageDataGenerator = image.ImageDataGenerator(rescale=1./255)
validImageDataGenerator = image.ImageDataGenerator(rescale=1./255)

#generators
trainGenerator = trainImageDataGenerator.flow_from_directory(trainDir, target_size=(768, 432), color_mode="grayscale", batch_size=30, class_mode="binary")
trainClassIndices = trainGenerator.class_indices #checks that signal class has index 1 as metrics Recall refers by default to the class with index 1
print("trainClassIndices:", trainClassIndices)
validGenerator = validImageDataGenerator.flow_from_directory(validDir, target_size=(768, 432), color_mode="grayscale", batch_size=30, class_mode="binary")
validClassIndices = validGenerator.class_indices
print("validClassIndices:", validClassIndices)

#data augmentation, as the dataset is still too small for proper training
dataAugmentation = Sequential([
  RandomFlip("horizontal_and_vertical"),
  RandomTranslation(0.3, 0.3),
  RandomRotation(0.3),
])

#model definition
model = Sequential()
model.add(dataAugmentation)
model.add(Conv2D(16, (3, 3), activation="relu", padding="same", name="conv_1", input_shape=(768, 432, 1)))
model.add(MaxPooling2D((3, 3), name="maxpool_1"))
model.add(Conv2D(16, (3, 3), activation="relu", padding="same", name="conv_2"))
model.add(MaxPooling2D((3, 3), name="maxpool_2"))
model.add(Conv2D(16, (3, 3), activation="relu", padding="same", name="conv_3"))
model.add(MaxPooling2D((3, 3), name="maxpool_3"))
model.add(Conv2D(16, (3, 3), activation="relu", padding="same", name="conv_4"))
model.add(MaxPooling2D((3, 3), name="maxpool_4"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu", name="dense_1"))
model.add(Dense(32, activation="relu", name="dense_2"))
model.add(Dense(1, activation="sigmoid", name="output"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[Precision(), Recall()])

#training
history = model.fit(trainGenerator, steps_per_epoch=150, epochs=20, validation_data=validGenerator, validation_steps=50, verbose=1)

#model saving
model.save("savedModel/model")
model.summary()

#saving history to file
historyDict = history.history
with open("{}/../historyDict.json".format(dataDir), "w") as historyDictFile:
  json.dump(historyDict, historyDictFile)
