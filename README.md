# diabetic-retinopathy-detection
This project is to use the Kaggle dataset and implement deep learning algorithms to predict Diabetic Retinopathy in eye vessel images.

This model was trained on the APTOS 2019 dataset to help automatically screen images for Diabetic Retinopathy, the #1 cause of blindness in adults.

Currently, this implementation is using transfer learning, extracting best features using the Inception V3 model from Keras and then predicting the categories from the flattened feature vector.
The model achieves 77.87% accuracy on an unseen portion of the training set, since the testing set provided by APTOS does not have labels.