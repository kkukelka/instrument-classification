# Musical Instrument Classification Using Convolutional Neural Networks With Residual Connections

This project originated while writing my bachelor thesis in musicology. It is now publicly available as reference
and study material. A convolutional neural network (CNN) is used to categorize an input signal into one of seven classes.
The data set consists of short audio recordings of musical instruments typically found in symphony orchestras. 
To provide the CNN with the appropriate data, the signals are transferred into the digital domain using log mel-spectrograms.

## Model and framework
My thesis model uses the convolutional base of Inception-ResNet-v2, which is based on the original Inception architecture 
developed by Szegedy et al. at Google - also known as "GoogleNet" ([Latest Paper](https://ai.google/research/pubs/pub45169)). Inception-ResNet-v2 introduces residual connections to the initial network, 
further improving training speed. While features are extracted using the pre-trained convolutional base, 
a separately added classification network consisting of fully connected layers is trained on the new data set. Several 
configurations of this classifier get tested and compared. The loss score is being evaluated using a softmax activation 
function with cross entropy loss. While stochastic gradient descent is utilized for weight updating, some configurations 
implement nesterov’s accelerated gradient in order to boost the optimization process.

