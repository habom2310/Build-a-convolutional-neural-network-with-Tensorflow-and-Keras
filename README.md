# Build-a-convolutional-neural-network-with-Tensorflow-and-Keras
Build a convolutional neural network with Tensorflow and Keras

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
│   ├───A
│   ├───C
│   ├───E
│   ├───F
├───Convert_keras_to_tf.py
├───Prediction_tf_pb.py
├───train.py
```

# Implementation
- Train the model
> `python train.py --dataset train_data --model model.model --label-bin bin --plot plot`
- Convert the model to tensorflow model
> `python Convert_keras_to_tf.py --keras_model model.model --tf_model tf_model.pb`
- Test the tensorflow model
> `python Prediction_tf_pb.py`
