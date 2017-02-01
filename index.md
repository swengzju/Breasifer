## Background

Breast cancer is the second leading cause of cancer related deaths in women. It has been reported that one in eight women will be diagnosed with invasive breast cancer in their lifetime.

This project is a consulting project provided by iSono Health. The mission of iSono Health is developing a low-cost, automated ultrasound imaging platform to facilitate monthly self-monitoring for women to help with early breast cancer detection.

## Data overview 

The raw dataset (courtesey of iSono Health) contains 2,091 labeled 2-D ultrasound images:

- Benign cases: 815
- Malignant cases: 1098
- Unusual cases: 178
- Subtypes in benign: 12
- Subtypes in malignant: 13
- Subtypes in unusual: 3

Over 99% of the images have the size of 300 x 225 pixels, each pixel has a value ranging from 0 to 255.

## Data preparation

I applied a 3x3 median filter to remove the speckle noise on ultrasound images. I resized all images to 300 x 300 by zero padding, for the purpose of easy convolutional neural networks modeling. I also normalized the pixel values so that they had zero mean.

In order to make a binary classification problem (benign or malignant), I ignored all the subtypes as well as the unusual cases. I randomly selected 1600 images to build my model, half of them were benign and half of them were malignant. Among all the images, 1200 of them were randomly selected for training, 200 of them were for validation, and the other 200 were for testing.

## Algorithms

Flow chart

#### Principle component analysis

#### Logistic regression

#### Random forest

#### Fully connected neural networks

#### Convolutional neural networks

## Discussion

## Conclusion
