## Background

Breast cancer is the second leading cause of cancer related deaths in women. It has been reported that one in eight women in the U.S. will be diagnosed with invasive breast cancer in their lifetime. In 2017, about 255,180 new cases of invasive breast cancer are expected to be diagnosed and 40,610 women in the U.S. are expected to die from breast cancer.

![alt text](https://farm1.staticflickr.com/579/32671533255_a112f9d7cb_b.jpg)

Traditionally, once-a-year screening is recommended to check a woman’s breasts health before there are signs or symptoms of the disease, especially for those who are over 50 years old. The screening normally involves X-ray mammograms followed by ultrasound imaging. However, the diagnosis on breast ultrasound is a subjective procedure and highly dependent on the experience of the surgeons. 

[iSono Health](http://www.isonohealth.com/) is a startup company committed to developing an affordable, automated ultrasound imaging platform to facilitate monthly self-monitoring for women to help with early breast cancer detection. The device is bundled with iSono app that can analyze the results and tag any changes in the back end in real time. During the time at Insight Health Data Fellow program, I was working with iSono Health focusing on building a deep learning model to differentiate benign and malignant breast lesions based on the ultrasound images.

![alt text](https://farm1.staticflickr.com/778/32518628362_5e8e668c23_b.jpg)

## Data overview 

The raw dataset (courtesey of iSono Health) contains 2,091 labeled 2-D breast ultrasound images in JPEG format:

- Benign cases: 815
- Malignant cases: 1098
- Unusual cases: 178
- Subtypes in benign: 12
- Subtypes in malignant: 13
- Subtypes in unusual: 3

Over 99% of the images have the size of 300 x 225 pixels, each pixel has a value ranging from 0 to 255. There are 12 subtypes in the benign cases and 13 subtypes in the malignant cases. The images were obtained from different sources, so they have various field of views, resolutions, scales, etc. Below shows some benign and malignant breast lesions with different subtypes.

![alt text](https://farm1.staticflickr.com/290/32292792970_224a161014_b.jpg)

In order to make it a binary classification problem (benign or malignant), I ignored all the subtypes as well as the unusual cases. I randomly selected 1600 images to build my model, half of them were benign and half of them were malignant, therefore it was a balanced dataset. 

## Data preparation

I applied a 3x3 median filter to remove the speckle noise on ultrasound images. I also normalized the pixel values on each image so that they had zero mean. 

It is worth mentioning that 1600 images as a whole is still a relative small dataset for such a complicated image classification problem. Also, the features that characterize breast lesions should be rotation invariant. Therefore, I applied an image augmentation method by perturbing the existing dataset. Specifically, I rotated each image a random small degree from -7° to 7° and I did it for fourteen times, so I eventually got 1600 x (14 + 1) = 24000 images. 

Based on the observatin that the most interesting part (lesion and its surroundings) of almost all the images is located around the center of the image, I cropped all the images so that their dimensions were reduced to 200 x 200 pixels. I then downsampled them 5x to 40 x 40 pixels, in order to keep the input size of the network manageable. The flow chart of data preparation is shown below.

![alt text](https://farm1.staticflickr.com/731/32550953861_b9b119f581_b.jpg)

Finally, 18000 of them were randomly selected for training, 3000 of them were for validation, and the other 3000 were for testing. Training, validation, and test set all consist of 50% benign and 50% malignant cases.

![alt text](https://farm1.staticflickr.com/686/32631809276_9b69b5d9d9_b.jpg)

## Algorithms

To achieve correct classification, the conventional method is often composed of three main steps: feature extraction, feature selection, and classification. Tese three steps need to be well-addressed separately and then integrated together. Extraction of discriminative features could potentially ease the latter steps of feature selection and classification. Nevertheless, the engineering of effective features is problem-oriented and highly depends on the quality of each intermediate result in the image processing, which often needs many passes of trial-and-error design and case-by-case user interventions [1].

The recent advances of deep learning technology can potentially change the design paradigm of image classification. Deep learning can directly uncover features from the training data without the explicit elaboration on feature extraction and selection. With the help of back propagation, the internal structures of neural networks get updated automatically based on the error information obtained from each iteration.

In this project, I investigated and compared the performances of two different deep learning architectures, namely fully connected neural network and convolutional neural network.

#### Fully connected neural network

In fully connected neural network, each neuron is connected to all the neurons in the previous layer, and each connection has its own weight. However, the information of the weights are not shared by neurons. Before implementing fully connected neural network, there was an extra step of converting each 2D image into a 1D array with a size of 1600. After that, the input layer became a 18000 x 1600 matrix.

![alt text](https://farm1.staticflickr.com/319/31831201844_e63578572f_b.jpg)

The constructed fully connected neural network has one input layer, three hidden layers that have 512, 256, 128 nodes respectively, and one output layer that has two outputs. The diagram is shown below.

![alt text](https://farm1.staticflickr.com/719/32521161552_0d397712a0_b.jpg)

#### Convolutional neural network

In convolutional neural network, each neuron is only connected with a few local neurons in the previous layer, and the weight is shared for every neuron in that layer. Convolutional neural network is effective for image classification problems because the convolution operation produces information on spacially correlated features of the image. For example, convolution may result in edges becoming more prominent.

By feeding the output of one convolutional layer to another, higher-order features can be extracted. After convolution, these features can be more readily learned by a fully connected neural network. 

## Discussion

## Final remarks

I want to thank all the people at Insight and the fellow fellows for giving me tremendous help in this project. I also want to thank iSono Health and CTO Dr. Shadi Saberi for giving me the opportunity to work on this interesting and challenging problem. I really learned a lot during my journey at Insight.

Reference: 
[1] Cheng, Jie-Zhi, et al. "Computer-Aided diagnosis with deep learning architecture: applications to breast lesions in us images and pulmonary nodules in CT scans." Scientific reports 6 (2016).
