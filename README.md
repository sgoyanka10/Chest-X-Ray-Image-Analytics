## Chest-X-Ray-Image-Analytics

In this project, I have exploited deep learning to automate detection of ailments from X-rays. I have trained two models - a binary and a multi-class supervised model.

1. pneumonia-detection-in-chest-xrays-using-transfer-learning.ipynb - Transfer learning to predict the probability of a chest x-ray having signs of pneumonia. Used python with Keras to develop the model.

    data-set - 6056 images (4290 pneumonia, 1766 normal), link to one of the sources of dataset I have used - https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
    
    model - Retrained VGG16 model on given dataset, 
    result - F1 score ~ 91, Test accuracy ~ 92

2. ailments-detection-in-chest-xrays-using-transfer-learning.ipynb - Transfer learning to predict the probability of a chest x-ray having signs of multiple ailments (Nodule, Mass, Pneumonia). Used python with Keras to develop the model.

    data-set - 8000 images (2000 pneumonia, 2000 normal, 2000 mass, 2000 nodule), link to the sources of dataset - https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia , https://www.kaggle.com/nih-chest-xrays/data
    model - Retrained VGG16 model on given dataset, 
    result - Test accuracy ~ 70

Some important observations:

1. More labelled data can be added to make classifier more robust and less prone to overfitting/under-fitting i.e. more generalizable. Although adding more data may require more cpu/gpu/ram resource.
2. Model can further be fine tuned - different resolutions of images can be tried, different model-architecture altogether can be tried, multiple parameters like epoch, batch-size, optimizers can be optimized. 
3. Still CNN models doesn’t always look at the appropriate features from images to predict the output. For example to predict pneumonia signs, model does not always just look at lungs to predict the output. Continued work is needed to understand what specific features are being learned by these CNNs. Check this article to find out more - https://medium.com/@jrzech/what-are-radiological-deep-learning-models-actually-learning-f97a546c5b98


