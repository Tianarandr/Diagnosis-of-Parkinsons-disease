import tensorflow.keras as keras
import numpy as np
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
from utils.utils import save_logs
from utils.utils import calculate_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score 
import itertools
from keras.utils import np_utils



from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from numpy.testing import assert_allclose
from keras.layers import Conv2D, BatchNormalization, MaxPooling1D, Conv1D



class Classifier_MCDCNN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False,build=True):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        n_t = input_shape[0]
        n_vars = input_shape[1]

        padding = 'valid'

        if n_t < 60: # for ItalyPowerOndemand
            padding = 'same'

        input_layers = []
        conv2_layers = []

        for n_var in range(n_vars):
            model = Sequential()
            model.add(Conv1D(6,2,padding=padding,  input_shape=(n_t,1)))
            model.add(Activation('relu'))

            model.add(MaxPooling1D(6, 2, padding=padding))
            model.add(Dropout(0.3))
            model.add(Conv1D(8,2, padding=padding))
            model.add(Activation('relu'))

            model.add(MaxPooling1D(8,2, padding=padding))
            model.add(Flatten())
            

        model.add(Dense(200))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(nb_classes, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer="Adam",
                      metrics=['accuracy'])

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)

        self.callbacks = [model_checkpoint]

        return model

    def prepare_input(self,x):
        new_x = []
        n_t = x.shape[1]
        n_vars = x.shape[2]

        for i in range(n_vars):
            new_x.append(x[:,:,i:i+1])

        return  new_x

    def fit(self, x, y, x_test, y_test, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        mini_batch_size = 25
        nb_epochs = 200

        x_train, x_val, y_train, y_val = \
            train_test_split(x, y, test_size=0.33)

        x_test = self.prepare_input(x_test)
        x_train = self.prepare_input(x_train)
        x_val = self.prepare_input(x_val)

        start_time = time.time()
        Y_train = np_utils.to_categorical(y_train)
        Y_test = np_utils.to_categorical(y_val)
        
        Y_true = np_utils.to_categorical(y_test)

        print(y_val.shape, y_train.shape)

        history = self.model.fit(x_train, Y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, Y_test), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        y_pred = model.predict(x_test)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, history, y_pred, y_true, duration)

        keras.backend.clear_session()

        print(history.history.keys())
        plt.figure(1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        file_name = self.output_directory+'epochs_validation.png'
        plt.savefig(file_name)

        plt.figure(2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        file_name = self.output_directory+'epochs_loss.png'
        plt.savefig(file_name)

        score = self.model.evaluate(x_test, Y_true, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        #save_logs(self.output_directory, history, y_pred, y_true, 0.0)

        
        cm_plot_labels = [ '0','1']


        conf_matrix = confusion_matrix(y_true, y_pred)

        self.plot_confusion_matrix(cm          =  conf_matrix, 
                            normalize    = False,
                            target_names = cm_plot_labels,
                            title        = "Confusion Matrix")

    def predict(self, x_test,y_true,x_train,y_train,y_test,return_df_metrics = True):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(self.prepare_input(x_test))
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred



    def plot_confusion_matrix(self, cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

            accuracy = np.trace(cm) / float(np.sum(cm))
            misclass = 1 - accuracy

            if cmap is None:
                cmap = plt.get_cmap('Blues')

            plt.figure(3)
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            if target_names is not None:
                tick_marks = np.arange(len(target_names))
                plt.xticks(tick_marks, target_names, rotation=45)
                plt.yticks(tick_marks, target_names)
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            thresh = cm.max() / 1.5 if normalize else cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")


            plt.tight_layout()
            plt.ylabel('True classes')
            plt.xlabel('Predicted classes\n \nAccuracy={:0.4f}; Misclass={:0.4f}'.format(accuracy, misclass))
            file_name = self.output_directory+'Confusion_matrix.png'
            plt.savefig(file_name)


