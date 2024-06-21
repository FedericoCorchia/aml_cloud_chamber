import os
import sys
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout, RandomFlip, RandomRotation
from keras.preprocessing import image
from keras.metrics import Recall

dataDir = str(sys.argv[1])
trainDir = os.path.join(dataDir, "train")
testDir = os.path.join(dataDir, "test")
print("dataDir:", dataDir)

trainImageDataGenerator = image.ImageDataGenerator(rescale=1./255)
testImageDataGenerator = image.ImageDataGenerator(rescale=1./255)

trainGenerator = trainImageDataGenerator.flow_from_directory(trainDir, target_size=(1024, 576), color_mode="grayscale", batch_size=20, class_mode="binary")

trainClassIndices = trainGenerator.class_indices
print("trainClassIndices:", trainClassIndices)

testGenerator = testImageDataGenerator.flow_from_directory(testDir, target_size=(1024, 576), color_mode="grayscale", batch_size=20, class_mode="binary")

testClassIndices = testGenerator.class_indices
print("testClassIndices:", testClassIndices)

dataAugmentation = Sequential([
  RandomFlip("horizontal_and_vertical"),
  RandomRotation(0.2),
])

model = Sequential()
model.add(dataAugmentation)
model.add(Conv2D(32, (3, 3), activation="relu", padding="same", name="conv_1", input_shape=(1024, 576, 1)))
model.add(MaxPooling2D((2, 2), name="maxpool_1"))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same", name="conv_2"))
model.add(MaxPooling2D((2, 2), name="maxpool_2"))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same", name="conv_3"))
model.add(MaxPooling2D((2, 2), name="maxpool_3"))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same", name="conv_4"))
model.add(MaxPooling2D((2, 2), name="maxpool_4"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1024, activation="relu", name="dense_1"))
model.add(Dense(512, activation="relu", name="dense_2"))
model.add(Dense(256, activation="relu", name="dense_3"))
model.add(Dense(1, activation="sigmoid", name="output"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[Recall()])

history = model.fit(trainGenerator, steps_per_epoch=100, epochs=50, validation_data=testGenerator, validation_steps=50, verbose=1)
