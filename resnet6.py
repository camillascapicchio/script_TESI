
from keras import layers
from keras import models
import keras
from keras import callbacks
from keras import initializers
from keras.layers import AveragePooling2D, Input, Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU as LR
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.utils import plot_model
from matplotlib import pyplot
import matplotlib.patches as mpatches
import numpy as np
from scipy.misc import imsave
from numpy.random import seed
from time import time

seed(1)
batch_size = 4
alpha = 0.2
img_height = 450
img_width = 450
img_channels = 1

#
# network params
#

cardinality = 1


def residual_network(x):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.
    
    """
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)
#	y = Dropout(0.2)(y)
        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by  convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y

    # conv1
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)

    # conv2
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 64, 128, _project_shortcut=project_shortcut)

    # conv3
    for i in range(3):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 128, 256, _strides=strides)

    # conv4
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 128, 256, _strides=strides)

    # conv5
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 128, 256, _strides=strides)
    
    x = Dropout(0.2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(4, activation = 'softmax')(x)

    return x


image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
network_output = residual_network(image_tensor)
  
model = models.Model(inputs=[image_tensor], outputs=[network_output])
print(model.summary())
    

# definition of lr of the optimizer
rmsprop = optimizers.RMSprop(lr=0.1)

# definition of callbacks

#reduce_lr_train = ReduceLROnPlateau(monitor='loss', factor=0.1,
#                              patience=5, min_lr=0.01)

reduce_lr_val = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=15, min_lr=0.001, verbose = 1)

filepath="/home/users/camilla/BIRADS_crop/Senograph53/17SETTEMBRE2019/1/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

def on_epoch_end(self, epoch, logs=None):
    print(keras.eval(self.model.optimizer.lr))

# definition of the inputs via ImageDataGenerator

sgd =keras.optimizers.SGD(lr=0.1, decay=1e-1, momentum=0.9, nesterov=True)



train_datagen = ImageDataGenerator(
        zoom_range=0.2,
	samplewise_center = True,
        samplewise_std_normalization = True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=10,
        horizontal_flip=False)


val_datagen = ImageDataGenerator(
        samplewise_center = True,
	samplewise_std_normalization = True)
        

train_generator = train_datagen.flow_from_directory(
        '/arinas/Radioma_AOUP_fixed/camilla/dati_camilla/CC_R/train',
        batch_size = batch_size,
        target_size=(img_width, img_height),
        color_mode = 'grayscale',
        shuffle = True,
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        '/arinas/Radioma_AOUP_fixed/camilla/dati_camilla/CC_R/validation',
        batch_size = batch_size,
        target_size=(img_width, img_height),
        color_mode = 'grayscale',
        shuffle = True,
        class_mode='categorical')



tensorboard = TensorBoard(log_dir="/home/users/camilla/BIRADS_crop/Senograph53/17SETTEMBRE2019/1/logs/{}".format(time()), histogram_freq=0, batch_size=4, write_grads=True, write_images=True)

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_generator,
          steps_per_epoch=210,
          epochs = 100,
          validation_data=validation_generator,
          validation_steps=39,
          callbacks=[reduce_lr_val, checkpoint, tensorboard])


np.savetxt('/home/users/camilla/BIRADS_crop/Senograph53/17SETTEMBRE2019/1/loss_train.txt', history.history['loss'], delimiter=",")
np.savetxt('/home/users/camilla/BIRADS_crop/Senograph53/17SETTEMBRE2019/1/acc_train.txt', history.history['acc'], delimiter=",")
np.savetxt('/home/users/camilla/BIRADS_crop/Senograph53/17SETTEMBRE2019/1/loss_val.txt', history.history['val_loss'], delimiter=",")
np.savetxt('/home/users/camilla/BIRADS_crop/Senograph53/17SETTEMBRE2019/1/acc_val.txt', history.history['val_acc'], delimiter=",")


model.save('/home/users/camilla/BIRADS_crop/Senograph53/17SETTEMBRE2019/1/final_weights.h5')

