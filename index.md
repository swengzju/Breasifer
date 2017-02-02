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

I applied a 3x3 median filter to remove the speckle noise on ultrasound images. I normalized the pixel values on each image so that they had zero mean. Based on the observatin that the most interesting part (lesion and its surroundings) of almost all the images is located around the center of the image, I cropped all the images so that their dimensions were reduced to 200 x 200 pixels.

It is worth mentioning that 2000 images as a whole is still a relative small dataset for such a complicated image classification problem. Also, the features that characterize breast lesions should be rotation invariant. Therefore, I applied an image augmentation method by perturbing the existing dataset. Specifically, I rotated each image a random small degree from -7° to 7° and I did it for fourteen times, so I eventually got 1600 x (14 + 1) = 24000 images. I then downsampled them 5x to 40 x 40 pixels, in order to keep the input size of the network manageable. The flow chart of data preparation is shown below.



Finally, 18000 of them were randomly selected for training, 3000 of them were for validation, and the other 3000 were for testing. Training, validation, and test set all have 50% benign and 50% malignant cases.



## Algorithms

#### Fully connected neural networks

#### Convolutional neural networks

## Discussion

## Final remarks
