import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2
import numpy as np
import time
import glob

files = glob.glob("binary_train_data/6/*.jpg")
print(files)
#LOAD LABLES
labels = []
proto_as_ascii_lines = tf.gfile.GFile("labels.txt").readlines()
for l in proto_as_ascii_lines:
    labels.append(l.rstrip())

#LOAD MODEL
graph = tf.Graph()
graph_def = tf.GraphDef()
with open("./model/tf_model.pb", 'rb') as f:
    graph_def.ParseFromString(f.read())
with graph.as_default():
    tf.import_graph_def(graph_def)

sess = tf.Session(graph=graph)    

# for op in graph.get_operations():
    # print(str(op.name))

   

#READ TENSOR FROM IMAGE
for file in files:
    t0 = time.time() 

    image = cv2.imread(file,0)
    image = cv2.resize(image, dsize=(32, 32), interpolation = cv2.INTER_CUBIC)
    image = np.reshape(image, (32, 32, 1))
    #print(image.shape)

    np_image_data = np.asarray(image)
    #np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
    image_tensor = np.expand_dims(np_image_data,axis=0)

    #feed tensor into the network
    softmax_tensor = sess.graph.get_tensor_by_name('import/dense_2/Softmax:0')
    predictions = sess.run(softmax_tensor, {'import/conv2d_1_input:0': image_tensor})
    
    print(predictions)
    predictions = np.squeeze(predictions)

    top_k = predictions.argsort()[-1:][::-1]
    print(labels[top_k[0]])
    print("prediction time " + str(time.time()-t0))
    cv2.imshow("a",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



