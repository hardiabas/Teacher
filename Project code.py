# %% Section 1  import lybraris-access directary photos and csv file,
# read photos and append them to an array "X", may be link photo with csv file


from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping  # TensorBord
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, Dense, Flatten, Activation, SimpleRNN
from tensorflow.keras.layers import Conv2D, Conv1D, GlobalAveragePooling2D, MaxPooling2D, Dense, Flatten, Activation, Dropout, Input
from keras.models import Sequential  # Sequential Models
from keras.layers import Dense  # Dense Fully Connected Layer Type
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.layers import LSTM, GRU, Dropout, TimeDistributed, Reshape, Input, Lambda, Add
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential

from keras.applications.inception_v3 import decode_predictions
from tensorflow import keras
from skimage.transform import resize

from skimage.transform import resize
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
import cv2
#path = glob.glob("F:/implementation/test/*.jpg")
image_size = 200
path = "F:/implementation/resize/"

class_nums = {0, 1, 2, 3, 4, 5, 6, 7}
class_nams = ["up,mid", "up,right", "up,left", "up,front",
              "down,mid", "down,right", "down,left", "down,front"]
# reading image using its name
img = plt.imread("F:/implementation/resize/" + "frame11.jpg")
# plt.imshow(img)
print(img)
print(np.ndim(img))
print(img.shape)
print(img.size)

data = pd.read_csv('F:/implementation/mapping7.csv')  # reading the csv file
# print(data.head(10))  # printing first five rows of the file
y = data.iloc[:, 5:]
#print("the labels: ",y.head(10))
#data = data.drop(["Teacher_ID","Act_Class"], axis=1)
# print(data.head(10))  # printing first five rows of the file
# print(y.shape)
X = []  # creating an empty array
"""
for image_path in glob.glob(os.path.join("F:/implementation/resize/", "*.jpg" )):
    img= plt.imread(image_path)
    img= cv2.resize(img, (image_size, image_size))
    X.append(img)
"""
for image_name in data.Image_ID:
    img = plt.imread(path + image_name + ".jpg")
    a = cv2.resize(img, (image_size, image_size))
    X.append(a)  # storing each image in array X
"""
for file_name in path:
    img=plt.imread(file_name )
    X.append(img)  # storing each image in array X

import tqdm
from keras.preprocessing import image
for i in tqdm(range(data.shape[0])):
    img = image.load_img('Images/data/'+ data['Imsge_ID'][i])

    img = image.img_to_array(img)

    X.append(img)
"""
X = np.array(X)
plt.imshow(X[60])
print("the array is :", X)
print(X.shape)
# print(y.shape)
# print(X.shape[2])


# %% Section 2 train&test data split and print them
y = data.Act_Class

y_categorical = np_utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.1, random_state=42, stratify=y)

print(X_train)
print("Train data shape: ", X_train.shape)
print("train label shape: ", y_train.shape)
print("Test data shape: ", X_test.shape)
print("Test label shapee: ", y_test.shape)

print(y_train[300])
plt.imshow(X_train[300])
print(X_train.shape)
print(type(X_train))
#print("the class is: ", class_nams[1])
"""
for i in range (10):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    #plt.xlabel(class_nams[y_train[i]) # label as xlabel
   

plt.show()

#print(y_train[:5])

#plt.imshow(X_train[0])

"""
# %% Section 4 execute InceptionV3 base model on X_train and X_valid data
# prepare input data to our basic fine tuning model
#from tensorflow.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import LayerNormalization, BatchNormalization
def preprocess_data(K, J):
    x = preprocess_input(K)
    y = preprocess_input(J)
    return x, y
X_train, X_test = preprocess_data(X_train, X_test)
print(X_train.shape)

#%%
from sklearn.model_selection import StratifiedKFold
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
resnet_model= keras.applications.resnet50.ResNet50(
    weights='imagenet', include_top=False, input_shape=(200, 200, 3))

for layer in resnet_model.layers:
    layer.trainable = False
# inception_v3_model.trainable=False
#resnet_model.summary()

x = GlobalAveragePooling2D()(resnet_model.output)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
Classify = Dense(8, activation="softmax")(x)
model = Model(inputs=resnet_model.inputs,
              outputs=Classify, name="resnet50")
model.summary()


adam = keras.optimizers.Adam(
    lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# Adam computes individual learning rates for different parameters
reduce_lrt = ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.1, patience=8, verbose=1, min_lr=1e-3)
early_stopping= EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights= True, mode="max" )
checkpoint = ModelCheckpoint(
    'my_best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True)  # mode="max"
callbacks = [checkpoint, reduce_lrt]

model.compile(loss='binary_crossentropy',
              metrics=['accuracy'], optimizer=adam)
model.save('my_best_model.h5')


hist = model.fit(X_train, y_train, batch_size=32, epochs=60,
                 validation_data=(X_test, y_test), 
                 callbacks=[checkpoint, reduce_lrt])

y_predect = model.predict(X_test)
print(y_predect)

final_loss, final_accuracy = model.evaluate(X_test, y_test)
print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))

# %% Section 7 Accuracy and loss plot
epochs = 60
model = tf.keras.models.load_model("my_best_model.h5")
final_loss, final_accuracy = model.evaluate(X_test, y_test)
print(f"final accuracy is: {final_accuracy} final loss is: {final_loss}")
prediction = model.predict([X_test])
print(prediction[0])
print(np.argmax(prediction[20]))
print(class_nams[np.argmax(prediction[20])])

train_acc = hist.history["accuracy"]
train_loss = hist.history["loss"]
val_acc = hist.history["val_accuracy"]
val_loss = hist.history["val_loss"]
N = range(epochs)
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(N, train_acc)
plt.plot(N, val_acc)
plt.xlabel("EPOCHS")
plt.ylabel("Train_acc/Val_acc")
plt.legend(["train_acc", "val_acc"], loc=1)
plt.style.use(["seaborn"])
plt.title("Train/Validation accuracy Plot", fontsize=25, color="green")
plt.subplot(1, 2, 2)
plt.plot(N, train_loss)
plt.plot(N, val_loss)
plt.xlabel("EPOCHS")
plt.ylabel("Train_loss/Val_loss")
plt.legend(["train_loss", "val_loss"], loc=1)
plt.style.use(["seaborn"])
plt.title("Train/Validation loss Plot", fontsize=25, color="red")
plt.show()
# %% Section 9 confusion matrex
#dict_characters = {'no','down,front' , 'down,left','down,right','up,front','up,left','up,mid',"up,right"}
Y_pred = model.predict(X_test)
Truth = np.argmax(y_test, axis=1)
prediction = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(Truth, prediction)
cm_plot = sns.heatmap(cm, cmap=plt.cm.Reds, linecolor="white",
                      annot=True, xticklabels=class_nams, yticklabels=class_nams)
plt.xlabel("Actual", fontsize=20, color="green")
plt.ylabel("predicted", fontsize=20, color="green")
plt.title("confusion matrix", fontsize=25, color="green")
plt.show()
# %% Section 10
print(classification_report(Truth, prediction, digits=2))
print("accuracy score is: ", accuracy_score(Truth, prediction))
print("precision score is: ", precision_score(
    Truth, prediction, average="weighted"))
print("recall score is: ", recall_score(Truth, prediction, average="weighted"))
print("f1 score is: ", f1_score(Truth, prediction, average="weighted"))

#%% Evaluation process e1
image_size = 200
path1 = "F:/implementation/evaluation/teacher34/"
# reading image using its name
#img = plt.imread("F:/implementation/evaluation/teacher34/" + "frame8300.jpg")
# plt.imshow(img)
#print(img)
#print(np.ndim(img))
#print(img.shape)
#print(img.size)

data1 = pd.read_csv('F:/implementation/teacher34.csv')  # reading the csv file
# print(data.head(10))  # printing first five rows of the file
#print("the labels: ",y.head(10))
#data = data.drop(["Teacher_ID","Act_Class"], axis=1)
# print(data.head(10))  # printing first five rows of the file
# print(y.shape)
valid_image = []  # creating an empty array
"""
for image_path in glob.glob(os.path.join("F:/implementation/resize/", "*.jpg" )):
    img= plt.imread(image_path)
    img= cv2.resize(img, (image_size, image_size))
    X.append(img)
"""
for image_name in data1.Image_ID:
    img = plt.imread(path1 + image_name + ".jpg")
    img = cv2.resize(img, (image_size, image_size))
    valid_image.append(img)  # storing each image in array X
"""
for file_name in path:
    img=plt.imread(file_name )
    X.append(img)  # storing each image in array X

import tqdm
from keras.preprocessing import image
for i in tqdm(range(data.shape[0])):
    img = image.load_img('Images/data/'+ data['Imsge_ID'][i])

    img = image.img_to_array(img)

    X.append(img)
"""
valid_image = np.array(valid_image)
plt.imshow(valid_image[60])
print("the array is :", valid_image)
print(valid_image.shape)
# print(valid_image.shape[2])

#%% secti e2
""""
from numpy import argmax
from tensorflow.keras.applications.resnet50 import preprocess_input
valid_image = preprocess_input(valid_image)
print(valid_image.shape)

#valid_image= valid_image/255
prediction = model.predict(valid_image)
predictions = np.argmax(prediction, axis=1)
predictions.shape[0]
print(predictions)

print("up ,mid", predictions[predictions==0].shape[0], "seconds")
print("up,right", predictions[predictions==1].shape[0], "seconds")
print("up,left", predictions[predictions==2].shape[0], "seconds")
print("up,front", predictions[predictions==3].shape[0], "seconds")
print("down,mid", predictions[predictions==4].shape[0], "seconds")
print("down,right", predictions[predictions==5].shape[0], "seconds")
print("down,left", predictions[predictions==6].shape[0], "seconds")
print("down,front", predictions[predictions==7].shape[0], "seconds")

dataset2 = pd.DataFrame({'TEACHER':['up ,right','up,left' ,'up,mid ','up,front','down,right','down,left','down,mid','down,front'],'SECOND':[predictions[predictions==1].shape[0],predictions[predictions==2].shape[0],predictions[predictions==3].shape[0],predictions[predictions==4].shape[0],predictions[predictions==5].shape[0],predictions[predictions==6].shape[0], predictions[predictions==7].shape[0], predictions[predictions==8].shape[0]]})
import seaborn as sns
sns.set(font_scale=1.5)
plt.figure(figsize=(5,4))
plt.xticks(rotation=90)
ax = sns.barplot(x="TEACHER", y="SECOND", data=dataset2 )

up_right=predictions[predictions==0].shape[0]
up_left=predictions[predictions==1].shape[0]
up_mid=predictions[predictions==2].shape[0]
up_front=predictions[predictions==3].shape[0]
down_right=predictions[predictions==4].shape[0]
down_left=predictions[predictions==5].shape[0]
down_mid=predictions[predictions==6].shape[0]
down_front=predictions[predictions==7].shape[0]

Hand_up = up_right+up_left+up_mid+up_front
Hand_down = down_right+down_left+down_mid+down_front
mid_zone= up_mid+down_mid
right_zone=up_right+down_right
left_zone=up_left+down_left
front_zone=up_front+down_front
print("the time of hand shake is: ", Hand_up, "Seconds")
print("the time of hand down: ", Hand_down, "Seconds")
print("the time of mid_zone is: ", mid_zone, "Seconds")
print("the time of right zone is: ", right_zone, "Seconds")
print("the time of left zone is: ", left_zone, "Seconds")
print("the time of front zone: ", front_zone, "Seconds")
#moving=predictions[predictions==2].shape[0]+predictions[predictions==6].shape[0]
#hasel=moving/len(test_image)
#print(hasel)
bad=0
mid=0
good=0
if 10<up_right <(20) or 10<up_left<20 or  10<up_mid<30 or 10<up_front <(20) or  10<down_right<30 or  10<down_left<30 or 10<down_mid<30 or 10<down_front<30:
    bad=0
    mid=70
    good=0
elif up_right <10 or up_left<10 or  up_mid<10 or up_front <(10) or  down_right<10 or  down_left<10 or down_mid<10  or down_front<10:
    bad=70
    mid=0
    good=0
elif up_right >20 or up_left>10 or  up_mid>30 or up_front >(20) or  down_right>30 or  down_left>30 or down_mid>30  or down_front>30:
    bad=0
    mid=0
    good=70
    
dataset3 = pd.DataFrame({'moving':['bad','mid' ,'good '],'situation':[bad,mid, good]})

   
import seaborn as sns
sns.set(font_scale=1.5)
plt.figure(figsize=(5,4))
plt.xticks(rotation=90)
ax = sns.barplot(x="moving", y="situation", data=dataset3)
"""

#%% secti e2


from numpy import argmax
from tensorflow.keras.applications.resnet50 import preprocess_input
valid_image = preprocess_input(valid_image)
print(valid_image.shape)

#valid_image= valid_image/255
prediction = model.predict(valid_image)
predictions = np.argmax(prediction, axis=1)
predictions.shape[0]
print(predictions)
a = predictions[predictions==0].shape[0]
#print(a)
b = predictions[predictions==1].shape[0]
#print(b)
c = predictions[predictions==2].shape[0]
#print(c)
d = predictions[predictions==3].shape[0]
#print(d)
e = predictions[predictions==4].shape[0]
#print(e)
f = predictions[predictions==5].shape[0]
print(f)
g = predictions[predictions==6].shape[0]
#print(g)
h = predictions[predictions==7].shape[0]
#print(h)
list = [a,b,c,d,e,f,g,h]
print(list)
print("up,mid", predictions[predictions==0].shape[0], "seconds")
print("up,right", predictions[predictions==1].shape[0], "seconds")
print("up,left", predictions[predictions==2].shape[0], "seconds")
print("up,front", predictions[predictions==3].shape[0], "seconds")
print("down,mid", predictions[predictions==4].shape[0], "seconds")
print("down,right", predictions[predictions==5].shape[0], "seconds")
print("down,left", predictions[predictions==6].shape[0], "seconds")
print("down,front", predictions[predictions==7].shape[0], "seconds")
"""
up_mid=predictions[predictions==0].shape[0]
up_right=predictions[predictions==1].shape[0]
up_left=predictions[predictions==2].shape[0]
up_front=predictions[predictions==3].shape[0]
down_mid=predictions[predictions==4].shape[0]
down_right=predictions[predictions==5].shape[0]
down_left=predictions[predictions==6].shape[0]
down_front=predictions[predictions==7].shape[0]
Hand_up = up_right+up_left+up_mid+up_front
Hand_down = down_right+down_left+down_mid+down_front
mid_zone= up_mid+down_mid
right_zone=up_right+down_right
left_zone=up_left+down_left
front_zone=up_front+down_front
print("the time of hand shake is: ", Hand_up, "Seconds")
print("the time of hand down: ", Hand_down, "Seconds")
print("the time of mid_zone is: ", mid_zone, "Seconds")
print("the time of right zone is: ", right_zone, "Seconds")
print("the time of left zone is: ", left_zone, "Seconds")
print("the time of front zone: ", front_zone, "Seconds")
"""
dataset2 = pd.DataFrame({'TEACHER':['up,mid','up,right','up,left ','up,front','down,mid','down,right','down,left','down,front'],'SECOND':[predictions[predictions==0].shape[0],predictions[predictions==1].shape[0],predictions[predictions==2].shape[0],predictions[predictions==3].shape[0],predictions[predictions==4].shape[0],predictions[predictions==5].shape[0], predictions[predictions==6].shape[0], predictions[predictions==7].shape[0]]})
import seaborn as sns
sns.set(font_scale=1.5)
plt.figure(figsize=(5,4))
plt.xticks(rotation=90)
ax = sns.barplot(x="TEACHER", y="SECOND", data=dataset2 )

bb_class=0
mm_class=0
gg_class=0
for y in list:
    if y < 12:
        bb_class= bb_class+1
    elif 12<= y<24:
       mm_class= mm_class+1
    else:
       gg_class=gg_class+1

print(bb_class,mm_class,gg_class)
#print("the number of bad class is: ", bad_class)
#print("the number of mid class is: ", mid_class)
#print("the number of good class is: ", good_class)
if bb_class > mm_class and bb_class > gg_class:
    bad=70
    mid=0
    good=0
elif mm_class >= bb_class and mm_class > gg_class:
    bad=0
    mid=70
    good=0
elif gg_class >= bb_class and gg_class >= mm_class:
    bad=0
    mid=0
    good=70
    
dataset3 = pd.DataFrame({'moving':['bad','mid' ,'good '],'situation':[bad,mid, good]})
   
import seaborn as sns
sns.set(font_scale=1.5)
plt.figure(figsize=(5,4))
plt.xticks(rotation=90)
ax = sns.barplot(x="moving", y="situation", data=dataset3)


