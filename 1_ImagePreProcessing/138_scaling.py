import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.style.use('dark_background')

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.datasets import cifar10
from keras.utils import normalize, to_categorical


(X_train, y_train), (X_test, y_test) = cifar10.load_data()                  #cifar 10 dataset

X_train = normalize(X_train, axis=1)                                        #scaled 255
X_test = normalize(X_test, axis=1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

######### Data augmentation

train_datagen = ImageDataGenerator(rotation_range=45, width_shift_range=0.2, zoom_range = 0.2, horizontal_flip = True)  # 이미지 전처리 클래스
train_datagen.fit(X_train)

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size = 32)

#he_uniform : 균등분포 분산 스케일링 초기값 설정
#평균 activation을 유지

activation = 'relu'
model = Sequential()                #하나의 입력 텐서와 하나의 출력 텐서를 갖는 일반 레이어 스택에 적합
model.add(Conv2D(32, 3, activation = activation, padding = 'same', input_shape = (32, 32, 3)))
model.add(BatchNormalization())

model.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())

model.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(128, activation = activation, kernel_initializer = 'he_uniform'))
model.add(Dense(10, activation = 'softmax'))
#Do not use softmax for binary classification
#Softmax is useful for mutually exclusive classes, either cat or dog but not both.
#Also, softmax outputs all add to 1. So good for multi class problems where each
#class is given a probability and all add to 1. Highest one wins.

#Sigmoid outputs probability. Can be used for non-mutually exclusive problems.
#But, also good for binary mutually exclusive (cat or not cat).

model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())

#NOTE: When we use fit_generator, the number of samples processed
#for each epoch is batch_size * steps_per_epochs.       fit_generator를 사용할 때 처리된 샘플 수는 각 에포크 batch_size*steps_per_epochs
#should typically be equal to the number of unique samples in our dataset divided by the batch size.
#For now let us set it to 500

history = model.fit_generator(
        train_generator,
        steps_per_epoch = 500,
        epochs = 30,
        validation_data = (X_test, y_test)
)

#plot the training and validation accuracy and loss at each epoch
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


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()