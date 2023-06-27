from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.layers import BatchNormalization

import os

##영화 데이터 이미지 확인 불가
image_directory = os.listdir('..')
print(image_directory)

# Now let us read metadata to get our Y values (multiple lables)
df = pd.read_csv('test.csv')
print(df.head())  # printing first five rows of the file
print(df.columns)

df = df.iloc[:2000]  #메모리 문제로 2000개만 읽어 놓은 것
# Otherwise, if read directly from the folder then images may not correspond to
# the metadata from the csv file.

SIZE = 200
X_dataset = []
for i in tqdm(range(df.shape[0])):
    img = image.load_img(image_directory + df['Image_Id'][i] + '.jpg', target_size=(SIZE, SIZE, 3))
    img = image.img_to_array(img)
    img = img / 255.
    X_dataset.append(img)

X = np.array(X_dataset)

#
print(df['Image_Id'][500])

print(df['labels'][500])  # Tagged as multiple Genres.

# Id and Genre are not labels to be trained. So drop them from the dataframe.
# No need to convert to categorical as the dataset is already in the right format.
y = np.array(df.drop(['Id', 'Genre'], axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.3)

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(SIZE, SIZE, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, activation='sigmoid'))              #softmax는 multi class 에서 사용하지 않는다


# Softmax is useful for mutually exclusive classes, either cat or dog but not both.
# Also, softmax outputs all add to 1. So good for multi class problems where each
# class is given a probability and all add to 1. Highest one wins.
# 소프트 맥스는 상호 배타적인 클래스에 유용(ex. 강아지, 고양이 각각 한 분류에 속하는 것), 출력은 모두 1에 추가. 다중 분류에 사용
# Sigmoid outputs probability. Can be used for non-mutually exclusive problems.
# like multi label, in this example.
# But, also good for binary mutually exclusive (cat or not cat).
#시그모이드는 확률을 출력, 다중레이블 같은 문제에 이용가능
model.summary()

# Binary cross entropy of each label. So no really a binary classification problem but
# Calculating binary cross entropy for each label. 각 레이블의 binary cross entropy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)

# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#################################################
# Validate on an image
# img = image.load_img('movie_dataset_multilabel/images/tt4425064.jpg', target_size=(SIZE,SIZE,3))
img = image.load_img('ddlj.jpg', target_size=(SIZE, SIZE, 3))

img = image.img_to_array(img)
img = img / 255.
plt.imshow(img)
img = np.expand_dims(img, axis=0)

classes = np.array(df.columns[2:])  # Get array of all classes
proba = model.predict(img)  # Get probabilities for each class
sorted_categories = np.argsort(proba[0])[:-11:-1]  # Get class names for top 10 categories

# Print classes and corresponding probabilities
for i in range(10):
    print("{}".format(classes[sorted_categories[i]]) + " ({:.3})".format(proba[0][sorted_categories[i]]))

###################################################

_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

################################################################