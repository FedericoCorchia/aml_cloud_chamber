This project proposes a Convolutional Neural Network (CNN) to recognise traces left by alpha particles in cloud chambers.

## Requirements
- Keras (for training)

- Matplotlib (for plotting)

##  Commands
To run training: ```python3 training.py <path_to_dataset_directory>```

To plot performance variables over training epochs: ```python3 plotting.py <path_to_historyDict.json> <number_of_training_epochs>```

More on these commands in the paragraphs below.

## Recap on Cloud Chambers
Cloud chambers (also called Wilson chambers or diffusion cloud chambers), are devices able to display invisible tracks of ionising radiation. Inside the cloud chamber, a 1 cm-thick layer of supersaturated vapour of isopropyl alcohol is created. When an electrically charged particle flies through the vapour, its passage causes ionisation of the isopropyl alcohol molecules, which act as condensation nuclei for surrounding vapour which rapidly condenses into very small droplets, forming a white track. This track is clearly visible with the naked eye (it also contrasts with the cloud chamber surface, which is black) and shows the path of the particle that produced it. Moreover, different particle properties lead to tracks with different aspect, making it also possible to determine the particle type and other properties.

This project focuses on the identification of alpha particles. In cloud chambers they create short and thick tracks. Sources of alpha radiation include uranium, radium, radon and americium-241.

## Dataset and Data Preparation
The dataset consists in video footage of an operating cloud chamber installed at the Department of Physics and Astronomy of the University of Bologna. During recording no sample of alpha-emitting source was put inside or near the chamber, therefore it only recorded radiation naturally occurring in the environment.

Total footage amounts to 32 minutes of length shot in HD 720p. To have images on which to apply the CNN, frames were extracted from the footage and saved as images. Frames were saved at a rate of 6 per second, using the _Scene Filter_ function of VLC Media Player. As high definition frames would imply an excessive load for the neural network, frame resolution was reduced with function ```resize``` of Python library PIL to 1024x576 (standard definition television resolution under the PAL standard) which guarantees anyway a proper resolution for this study. Even though images are _de facto_ in grayscale (white or gray tracks over a black background), footage was shot in colour (i.e. it is coded in RGB), so frames were converted (with function ```convert``` of PIL) from RGB coding into proper grayscale coding in order to have actually one single colour channel instead of three, with further storage saving and less load for the neural network. About 9000 images are obtained.

Frames are then split into a training (70% of the images) and a test dataset (30% of the images), corresponding to a training directory and a test one. Frames containing traces left by alpha particles (for the totality of the permanence of the traces in the cloud chamber, i.e. including when they start fading away) are considered as signal, the ones not containing any are considered background. When splitting between training and test set, "neighbouring" frames referring to the same passage of an alpha particle are kept in the same group (i.e. all frames in the training set or all frames in the test set). Training frames are the ones until 70% of footage, testing frames the ones afterwards. No random selection is done since frames related to the same event (i.e. with similarity in path) would end up in both the training and the test set and so provide an improper advantage to the neural network; frames in the test set must refer to alpha particle crossing events the neural network has never seen during training. Labelling of training images is done by creating a signal subdirectory and a background one and distributing the frames into them accordingly; the same is done for test images. This is because in the training code function ```tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory``` is used, which samples from the training set (or the test set during testing) and gets the label from the name of the subdirectory from which the individual sampled image comes. In conclusion, the dataset has the following directory structure:
```
-train
--signal
--background
-test
--signal
--background
``` 

Below are an example of a background image and one of a signal image. The traces appearing in both images are left by other particles (e.g. electrons), not of our interest in this study.

TO DO: ADD IMG. BACKGROUND

TO DO: ADD IMG. SIGNAL

## Training

The training script is ```train.py```. Tu run it: ```python3 training.py <path_to_dataset_directory>```

Images for training are sampled from the training set using function ```tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory```. This operation includes rescaling of the grayscale pixels from values in the interval [0, 255] to [0, 1], which favours good training. An analogous operation is also done for the test set. Even though training images amount to more than 6000 frames, this amount is found to be insufficient for good training of the CNN, also because this is a very skewed dataset, i.e. signal images are much less (about 1/10) than background images. For this reason, data augmentation is performed, introducing random flips and rotations of the images, which is equivalent to seeing more (and actually likely) events than the ones actually provided.

After this, the CNN model is defined, with first convolutional layers, each one implementing padding and being followed by a MaxPooling layer. The output of this block is flattened and fed to the neural network layers, not before having implemented dropout, again for better training. The output of the dense layers is a binary classification into signal or background. As metrics for evaluation, recall is used: the logic is that this system must be able to catch the most signal frames. This metrics is appropriate for such a skewed dataset (for absurd, a system that would label all images as background would have a precision of above 90%, which would be misleading) and it would be a worse error losing signal images (the experimenter would never know about their existence, leading to loss of useful information) than incorrectly labelling a background image as signal (the experimenter would then just ignore the misclassified image). The used loss function is binary crossentropy and the optimiser is Adam. The model is then trained and saved. A JSON file with performance data (loss and recall for both training and testing) called ```historyDict.json``` is also produced: it can be read to produce a plot by invoking the plotting script with ```python3 plotting.py <path_to_historyDict.json> <number_of_training_epochs>```

## Performance


## How This System May Be Used in Production for Research
As shown, this system is intended to locate traces of alpha particles in cloud chamber footage. Instead of having a human manually inspect the footage, this programme does it automatically for them, returning just the meaningful frames ready for study. An automatical workflow can be devised where footage is recorded of the cloud chamber, frames are automatically extracted (even, if this precision is needed for the study, at a higher rate than the 6 frames per second used to build the dataset for this project, up to the actual recording rate of the footage - 30 frames per second for the footage used for the dataset of this project) and then passed to the trained model, which then selects meaningful frames and marks them. The experimenter would be only left with useful frames with alpha traces for their study. This idea may be potentially expanded to traces left by other particle types.
