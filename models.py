from tensorflow.keras.layers import  Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.models import  Sequential


def build_model2(input_shape, num_classes):
    
    
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(num_classes, activation='softmax')
    ])
    
    return model


# model4 is Min max on enitire data set scaled model
# model5 is Min max on give dataset scaled model    


def cnn_model(input_shape, num_classes):
    model_cnn = Sequential()

    # 1st conv layer
    model_cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model_cnn.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model_cnn.add(BatchNormalization())


    # 2nd conv layer
    model_cnn.add(Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_uniform',kernel_regularizer=l2(0.0001)))
    model_cnn.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model_cnn.add(BatchNormalization())
    model_cnn.add(Dropout(0.3))

    # 3rd conv layer
    model_cnn.add(Conv2D(128, (2, 2), activation='relu',kernel_initializer='he_uniform',kernel_regularizer=l2(0.0001)))
    model_cnn.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model_cnn.add(BatchNormalization())
    model_cnn.add(Dropout(0.3))

    # flatten output and feed it into dense layer
    model_cnn.add(Flatten())
    model_cnn.add(Dense(128, activation='relu',kernel_initializer='he_uniform',kernel_regularizer=l2(0.0001)))
    model_cnn.add(BatchNormalization())
    model_cnn.add(Dropout(0.3))

    # output layer
    model_cnn.add(Dense(num_classes, activation='softmax'))
    
    return model_cnn