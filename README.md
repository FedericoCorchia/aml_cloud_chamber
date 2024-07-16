This project proposes a Convolutional Neural Network (CNN) to recognise traces left by alpha particles in cloud chambers.

# Running Instructions
## Options for Running
This CNN may be run in two ways:

- by running directly the Python scripts on your device;
- by using the provided notebook, whose code is the very same of the Python scripts (but for just minor specificities of the notebook environment, e.g. the dataset imports and the ```sys.argv``` command line inputs being replaced by simple strings able to be edited on the fly by the user).

The first way is recommended if you have your own GPUs available (e.g. you have access to a cluster with GPU-equipped nodes), while the second one is recommended if you do not have such an access and/or you prefer to run on resources like Google Colab.

The dataset is not included in this repository because of its size, but it can be requested to the author of this repository.

## Requirements
- Keras (for training)
- Matplotlib (for plotting)

Both libraries are available on the notebook version if run with Google Colab; if you run directly the scripts you should instead make sure you have them available on your device.

##  Commands to Directly Run the Scripts
To run training: ```python3 training.py <path_to_dataset_directory>```

To plot performance metrics over training epochs: ```python3 plotting.py <path_to_historyDict.json> <path_to_where_to_save_the_output_plot>```

More on these commands in the paragraphs below.

# Details on the CNN
## Recap on Cloud Chambers
Cloud chambers (also called Wilson chambers or diffusion cloud chambers), are devices able to display invisible tracks of ionising radiation. Inside the cloud chamber, a 1 cm-thick layer of supersaturated vapour of isopropyl alcohol is created. When an electrically charged particle flies through the vapour, its passage causes ionisation of the isopropyl alcohol molecules, that act as condensation nuclei for surrounding vapour which rapidly condenses into very small droplets, forming a white track. This track is clearly visible with the naked eye (it also contrasts with the cloud chamber surface, which is black) and shows the path of the particle that produced it. Moreover, different particle properties lead to tracks with different aspect, making it also possible to determine the particle type and other properties.

This project focuses on the identification of alpha particles. In cloud chambers they create short and thick tracks. Sources of alpha radiation include uranium, radium, radon and americium-241.

## Dataset and Data Preparation
The dataset consists in video footage of an operating cloud chamber installed at the Department of Physics and Astronomy of the University of Bologna. During recording no sample of alpha-emitting source was put inside the chamber, therefore it only recorded alpha radiation naturally occurring in the air inside the cloud chamber, from the radon fraction naturally present in the air.

Total footage amounts to 32 minutes of length shot in HD 720p. To have images on which to apply the CNN, frames were extracted from the footage and saved as images. Frames were saved at a rate of 6 per second, using the _Scene Filter_ function of VLC Media Player. As high definition frames would imply an excessive load for the neural network, frame resolution was reduced with function ```resize``` of Python library PIL to 768x432 (75% of horizontal resolution for standard definition television under the PAL standard and the same reduction for vertical resolution) which guarantees anyway a proper resolution for this study. Even though images are _de facto_ in greyscale (white or grey tracks over a black background), footage was shot in colour (i.e. it is coded in RGB), so frames were converted (with function ```convert``` of PIL) from RGB coding into proper greyscale coding in order to have actually one single colour channel instead of three, with further storage saving and less load for the neural network. About 9000 images are obtained.

Frames are then split into a training set (60% of the images), a validation one (20% of the images) and a test one (20% of the images), corresponding to a training directory, a validation one and a test one. Frames containing traces left by alpha particles (for the time of the permanence of the traces inside the cloud chamber, i.e. including when they start fading away) are considered as signal, the ones not containing any are considered background. In order to have a cleaner dataset, the final one or two frames of each alpha trace event, where the trace has already faded away a lot and what is left is just a rather "ambiguous" very weak trace, are removed. When splitting between training, validation and test set, "neighbouring" frames referring to the same passage of an alpha particle (i.e. to the same event) are kept in the same group (i.e. all frames of a given event in just one of the three directories). Test frames are the ones in the first 20% of footage, validation frames the ones between 20% and 40% of footage and training frames the ones afterwards. No random selection of what frames to put in which set is also done, since frames related to the same event (i.e. with similarity in path) would end up in multiple sets that are supposed to be independent: evaluation of training performance should be done on events the neural network has never seen during its training, otherwise the neural network would obtain an improper advantage. Labelling of training images is done by creating a signal subdirectory and a background one and by distributing the frames into them accordingly; the same is done for validation and test frames. This is because in the training code function ```tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory``` is used, which samples from the training/validation/test set and gets the label from the name of the subdirectory from which the individual sampled image comes. In conclusion, the dataset has the following directory structure:

```
-train
--signal
--background
-valid
--signal
--background
-test
--signal
--background
``` 

Here are an example of a background image (above) and one of a signal image (below). The traces appearing in both images are left by other particles (e.g. electrons), not of our interest in this study.

![Background Frame](https://github.com/FedericoCorchia/aml_cloud_chamber/blob/1d2501cf247a9a0b504c3629ca7a08df9518cf69/scene22039.png)

![Signal Frame](https://github.com/FedericoCorchia/aml_cloud_chamber/blob/1d2501cf247a9a0b504c3629ca7a08df9518cf69/scene49267.png)

TO DO: ADD IMG. SIGNAL

## Training

The training script is ```training.py```. Tu run it: ```python3 training.py <path_to_dataset_directory>```

Images for training are sampled from the training set using function ```tensorflow.keras.preprocessing.image.ImageDataGenerator.flow_from_directory```. This operation includes rescaling of the greyscale pixels from values in the interval [0, 255] to [0, 1], which favours good training. An analogous operation is also done for the validation set. Even though training images amount to more than 5000 frames, this amount is found to be insufficient for good training of the CNN, also because this is a very skewed dataset, i.e. signal images are in a much lower number (slightly less than 1/10) than background images. For this reason, data augmentation is performed, introducing random flips, translations and rotations of the images, which is equivalent to seeing more events than the ones actually provided and so the CNN becomes more flexible, able to recognise traces wherever they appear and whatever path they take, as long as compatible with an alpha track.

After this, the CNN model is defined, with firstly four convolutional layers with 16 3x3-sized filters each, each layer implementing padding and being followed by a MaxPooling layer with 3x3 window. 16 3x3-sized filters per layer are reasonable for identification of rather simple shapes, while the MaxPooling 3x3 window setting is done because of the still large resolution of the frames (even after the reduction described in the data preparation section), so that the deeper convolutional layers, supposed to identify more complex patterns, can work with input of appropriate size for their good functioning. The output of this block of convolutional layers is flattened and fed to the fully connected layers, not before having implemented dropout, again for better training. The optimal dropout was found to be the common 0.5. For the fully connected layers, two layers of size 64 and 32 respectively are used, again reasonable for the actual difficulty of this shape identification, followed by the output layer. The output of the dense layers is a binary classification into signal or background. As metrics for evaluation, precision and recall are used, since more appropriate for such a skewed dataset (for absurd, a system that labels all images as background would have an accuracy of above 90%, which would be misleading since it would not detect any signal image); the logic is that this system must be able to catch the most signal frames, also preserving good selection purity. The best threshold was found to be the default one of 0.5, which guarantees very good recall (i.e. less signal images are lost) and also preserves good precision. The used loss function is binary crossentropy and the optimiser is Adam.

The model is then trained for 20 epochs and saved in the standard Tensorflow _SavedModel_ format inside a directory called ```savedModel```. A JSON file with performance data (loss, precision and recall for both training and validation) called ```historyDict.json``` is also produced: it can be read to produce a plot by invoking the plotting script with ```python3 plotting.py <path_to_historyDict.json> <path_to_where_to_save_the_output_plot>```

## Performance
With the final CNN setup, in the test set very good recall is observed (about 90%) and also good precision is noticed (above 80%), meaning that the CNN is indeed able to recognise the vast majority of alpha particle traces frames and also to keep the selection it makes with good purity. For its behaviour over training epochs, it can be observed in the plot below (all trainings show the same general behaviour) that validation performance tends to be better than the training one for loss and in a relevant way for recall, this is because of dropout being used. Loss tends to follow a path of decrease, precision rises quickly at the beginning and stays high, while recall tends to increase but with fluctuations between epochs, anyway tending to a more stable equilibrium towards the end. For runtime performance, the CNN training was run on both NVIDIA V100 GPUs (on the CNAF-HPC cluster at INFN-CNAF in Bologna) and T4 GPUs (on Google Colab), using in both cases one GPU: each epoch takes between 70 s and 130 s on the V100 GPUs and between 45 s and 70 s on the T4 ones.

![Training Performance Plot](https://github.com/FedericoCorchia/aml_cloud_chamber/blob/1d2501cf247a9a0b504c3629ca7a08df9518cf69/plot.png)


## How This System May Be Used in Production for Research
As shown, this system is intended to locate traces of alpha particles in cloud chamber footage. Instead of having a human manually inspect the footage, this programme does it automatically for them, returning just the meaningful frames ready for study. An automatical workflow can be devised where footage is recorded of the cloud chamber, frames are automatically extracted (even, if this precision is needed for the study, at a higher rate than the 6 frames per second used to build the dataset for this project, up to the actual recording rate of the footage - 30 frames per second for the footage used for the dataset of this project) and then passed to the trained model, which then selects meaningful frames and marks them. The experimenter would be only left with useful frames with alpha traces for their study. This idea may be potentially expanded to traces left by other particle types.
