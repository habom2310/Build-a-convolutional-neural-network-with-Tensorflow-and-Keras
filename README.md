# Build-a-convolutional-neural-network-with-Tensorflow-and-Keras
This is a tutorial for building a CNN with tensorflow and keras. The model will be used for plate recognition. The model is then converted to used by OpenCV dnn in a [C# application](https://github.com/habom2310/ANPR-system). 

# Prerequisite
 - Packages: tensorflow, keras, scikit-learn, matplotlib
 - Project structure

```
├───train_data
│   ├───0
│   ├───1
│   ├───2
│   ├───3
│   ├───4
│   ├───5
│   ├───6
│   ├───7
│   ├───8
│   ├───9
├───Convert_keras_to_tf.py
├───Prediction_tf_pb.py
├───train.py
```
 - Link for the train_image https://drive.google.com/open?id=1PbX-2fRDKEewqMhG-E17VLY3DQh2-42V

# Implementation
- Train the model
> `python train.py --dataset train_data --model model.model --label-bin bin --plot plot`
- Convert the model to tensorflow model
> `python Convert_keras_to_tf.py --keras_model model.model --tf_model tf_model.pb`
- Test the tensorflow model
> `python Prediction_tf_pb.py`

# Result
- Accuracy: 99,9% after 20 epochs
- Processing time: 3ms/1 image
