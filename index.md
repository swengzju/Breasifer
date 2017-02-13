# Differentiate Benign and Malignant Breast Ultrasound Using Deep Learning

## Background

Breast cancer is the second leading cause of cancer related deaths in women. It has been reported that one in eight women in the U.S. will be diagnosed with invasive breast cancer in their lifetime. In 2017, about 255,180 new cases of invasive breast cancer are expected to be diagnosed and 40,610 women in the U.S. are expected to die from breast cancer.

![alt text](https://farm1.staticflickr.com/754/32555142611_c4f9e68617_b.jpg)

Traditionally, once-a-year screening is recommended to check a woman’s breasts health before there are signs or symptoms of the disease, especially for those who are over 50 years old. The screening normally involves X-ray mammograms followed by ultrasound imaging. However, the diagnosis on breast ultrasound is a subjective procedure and highly dependent on the experience of the surgeons. 

[iSono Health](http://www.isonohealth.com/) is a startup company committed to developing an affordable, automated ultrasound imaging platform to facilitate monthly self-monitoring for women to help with early breast cancer detection. The device is bundled with iSono app that can analyze the results and tag any changes in the back end in real time (details shown in the picture below). During the time at Insight Health Data Fellow program, I was working with iSono Health focusing on building a deep learning model to differentiate benign and malignant breast lesions based on the ultrasound images.

![alt text](https://farm1.staticflickr.com/778/32518628362_5e8e668c23_b.jpg)

## Data overview 

The raw dataset (courtesy of iSono Health) contains 2,684 labeled 2-D breast ultrasound images in JPEG format:

- Benign cases: 1007
- Malignant cases: 1499
- Unusual cases: 178
- Subtypes in benign: 12
- Subtypes in malignant: 13
- Subtypes in unusual: 3

Most images have the size of 300 x 225 pixels, each pixel has a value ranging from 0 to 255. There are 12 subtypes in the benign cases and 13 subtypes in the malignant cases. The images were obtained from different sources, so they have various field of views, resolutions, scales, etc. The picture below shows some benign and malignant breast lesions with different subtypes.

![alt text](https://farm1.staticflickr.com/290/32292792970_224a161014_b.jpg)

In order to make it a binary classification problem (benign or malignant), I ignored all the subtypes as well as the unusual cases. I randomly selected 1920 images to build my model, half of them were benign and half of them were malignant, therefore it was a balanced dataset. 

## Data preparation

I applied a 3x3 median filter to remove the speckle noise on ultrasound images. I also normalized the pixel values on each image so that they had zero mean. 

It is worth mentioning that 1920 images as a whole is still a relative small dataset for such a complicated image classification problem. Also, the features that characterize breast lesions should be rotation invariant. Therefore, I applied an image augmentation method by perturbing the existing dataset. Specifically, I rotated each image a random small degree from -10° to 10° and I did it for 12 times, so I eventually got 1920 x 12 = 23040 images. 

Based on the observation that the most interesting part (lesion and its surroundings) of almost all the images is located around the center of the image, I cropped all the images so that their dimensions were reduced to 200 x 200 pixels. I then downsampled them 5x to 40 x 40 pixels, in order to keep the input size of the network manageable. The flow chart of data preparation is shown below.

![alt text](https://farm1.staticflickr.com/353/32056769623_19591d1d5d_b.jpg)

Finally, 18000 of them were randomly selected for training, 3000 of them were for validation, and the other 3000 were for testing. Training, validation, and test set all consisted of 50% benign and 50% malignant cases. The holdout test dataset was separated from the training/validation set prior to the image augmentation, so there was no overlapping original images across the two groups.

## Algorithms

To achieve correct classification, the conventional method is often composed of three main steps: feature extraction, feature selection, and classification. Tese three steps need to be well-addressed separately and then integrated together. Extraction of discriminative features could potentially ease the latter steps of feature selection and classification. Nevertheless, the engineering of effective features is problem-oriented and highly depends on the quality of each intermediate result in the image processing, which often needs many passes of trial-and-error design and case-by-case user interventions [1].

The recent advances in deep learning technology can potentially change the design paradigm of image classification. Nature recently reported a work on classification of skin cancer using deep convolutional neural networks, which demonstrated a level of competence comparable to dermatologists [2]. When we speak of "deep" learning, we are not simply referrring to the number of layers. While there is no concrete definition of what "deep" means, it is the number of possible causal connections each neuron has that really shapes the "depth" of deep learning structures.

Deep learning can directly uncover features from the training data without the explicit elaboration on feature extraction and selection. Neural networks operate by passing the input information through layers of neurons that transform the input information into the output. With the help of back propagation, the internal weights of neural networks get updated automatically based on the error information obtained from each iteration.

In this project, I investigated and compared the performances of two different deep learning architectures, namely fully connected neural network and convolutional neural network. Both models were implemented using python and TensorFlow on a Nvidia Tesla K80 GPU hosted by Amazon Web Services (AWS) EC2 p2.xlarge instance.

![alt text](https://farm1.staticflickr.com/607/32028535324_2977015ce5_b.jpg)

#### Fully connected neural network

In fully connected neural network, each neuron is connected to all the neurons in the previous layer, and each connection has its own weight. However, the information of the weights are not shared by the neurons. Before implementing fully connected neural network, there was an extra step of converting each 2D image into a 1D array with a size of 1600. After that, the input layer became a 18000 x 1600 matrix.

![alt text](https://farm1.staticflickr.com/521/32724769326_df42e1a299_b.jpg)

The constructed fully connected neural network has one input layer, three hidden layers that have 512, 256, 128 nodes respectively, and one output layer that has two outputs. The batch size is 128. The detailed diagram is shown below.

![alt text](https://farm1.staticflickr.com/735/32642251811_d078b56364_b.jpg)

#### Convolutional neural network

In convolutional neural network, each neuron is only connected with a few local neurons in the previous layer, and the weight is shared for every neuron in that layer. Convolutional neural network is effective for image classification problems because the convolution operation produces information on spatially correlated features of the image. For example, convolution may result in edges becoming more prominent.

The constructed convolutional neural network has two convolutional layers, each of them has 64 5x5 filters. By feeding the output of one convolutional layer to another, higher-order features can be extracted. After convolution, these features can be more readily learned by a fully connected neural network. The convolutional layers can be thought of as preparing the data so that the fully connected layers can take advangtage of the spatial structure of the input image. The intuition is that after the image has been passed through multiple convolutional layers, the neurons will have been encoded with all the relavant spatial features. Following the convolutional layers, there are three fully connected layers with 512, 256, 128 nodes respectively. The batch size is 512. The detailed diagram is shown below.

![alt text](https://farm3.staticflickr.com/2204/32028535384_0133034a0e_b.jpg)

## Model evaluation

The prediction accuracy on the holdout test dataset was:

- Fully connected neural network: 0.67
- Convolutional neural network: 0.71

The convolutional neural network outperformed the fully connected neural network by four percent in terms of prediction accuracy. Based on the confusion matrix as shown below, the convolutional neural network model achieved:

- Sensitivity: 0.73
- Specificity: 0.69
- F1 score: 0.72

![alt text](https://farm3.staticflickr.com/2305/32028535434_c331f3494c_b.jpg)

The advantages of using convolutional neural network can also be observed from the following two figures. As the number of training iteration increased, the validation accuracy of the convolutional neural network quickly and smoothly ramped up to 0.9 after 2000 iterations, while the fully connected neural network did not reach 0.9 until around 4500 iterations. On the other hand, starting from 1000 iterations, the loss value of the convolutional neural network was always lower than that of the fully connected neural network, which indicated that the gradient descent function inside the convolutional neural network had a better performance in converging to the local minimum point. The loss value is calculated by a cost function, which essentially defines how far the model is from the desired output. The gradient descent is attempting to converge on a result that minimizes the cost function by slowing changing the weights.

![alt text](https://farm3.staticflickr.com/2341/32056769743_af515d3ba1_b.jpg)

To speed up the training procedure, stochastic gradient descent (SGD) and rectified linear unit (ReLu) were applied in both models. Also, learning rate decay was adopted to increase the performances. Dropout was used to prevent overfitting.

## Final remarks

The convolutional neural network has many hyperparameters that can be tuned, including but not limited to: number of convolutional layers, number of fully connected layers, number of filters, size of filters, number of hidden nodes, batch size, learning rate, max pooling size, dropout ratio, etc. Therefore, the performance of the model could be potentially further improved by fine tuning. Alternatively, sophisticated models along with sufficient number of data can potentially improve the model performance.

Lastly, I want to thank all the people at Insight and the fellow fellows for giving me tremendous help in this project. I also want to thank iSono Health and CTO Dr. Shadi Saberi for giving me the opportunity to work on this interesting and challenging problem. I really learned a lot during my journey at Insight.

[My presentation on this project](https://docs.google.com/presentation/d/1XtiPld3UoD4ynqO4Cv2Jg21637gvzf3G0sQgVUhsGR8/pub?start=false&loop=false&delayms=3000) 

<dl>
<iframe src="https://docs.google.com/presentation/d/1XtiPld3UoD4ynqO4Cv2Jg21637gvzf3G0sQgVUhsGR8/embed?start=false&loop=true&delayms=30000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
</dl>

Reference: 

[1] Cheng, Jie-Zhi, et al. "Computer-Aided diagnosis with deep learning architecture: applications to breast lesions in us images and pulmonary nodules in CT scans." Scientific reports 6 (2016).

[2] Esteva, Andre, et al. "Dermatologist-level classification of skin cancer with deep neural networks." Nature (2017).
