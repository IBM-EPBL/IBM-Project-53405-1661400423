import tensorflow as tf
import numpy as np
import trains
import cv2

model=tf.keras.models.load_model("C:\\Users\\ashis\\OneDrive\\Desktop\\Nalaiyathiran\\ps\\IBM-Project-21317-1659777635-main\\Pre-Requisites and Project structure\\Model Building\\sign_1.h5")
image=tf.keras.preprocessing.image
#print(model.summary())
fl_img='C:\\Users\\ashis\\OneDrive\\Desktop\\Nalaiyathiran\\ps\\IBM-Project-21317-1659777635-main\\Project Development Phase\\Sprint 3\\Project\\Data\\Train\\G\\Image_1667714982.6115465.jpg' 
img=image.load_img(fl_img,target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
pred=np.argmax(model.predict(x))
op=trains.dataset
ans=op[pred]
print("\n\t"+ans+"\n")