import keras 
import numpy as np 
import sklearn
from utils.utils import save_logs
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
import itertools
from sklearn.model_selection import cross_val_score

class Classifier_RESNET: 

    def __init__(self, output_directory, input_shape, nb_classes,nb_prototypes,classes,
                 verbose=False,load_init_weights = False):
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        self.nb_prototypes = nb_prototypes
        self.classes = classes
        if(verbose==True):
            self.model.summary()
        self.verbose = True
        if load_init_weights == True: 
            self.model.load_weights(self.output_directory.
                                    replace('resnet_augment','resnet')
                                    +'/model_init.hdf5')
        else:
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)
        
        # BLOCK 1 

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum 
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2 

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3 

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL 
        
        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(nb_classes, activation='sigmoid')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), 
            metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory+'best_model.hdf5' 

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
            save_best_only=True)

        self.callbacks = [reduce_lr,model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_test,y_true):
        self.enc = sklearn.preprocessing.OneHotEncoder()
        self.enc.fit(np.concatenate((y_train,y_true),axis=0).reshape(-1,1))
        y_train_int = y_train 
        y_train = self.enc.transform(y_train.reshape(-1,1)).toarray()
        y_test = self.enc.transform(y_true.reshape(-1,1)).toarray()
        batch_size = 10

        nb_epochs = 200

        mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

        if len(x_train)>4000: # for ElectricDevices
            mini_batch_size = 128

        history=self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                verbose=self.verbose, validation_data=(x_test,y_test) ,callbacks=self.callbacks)
        
        model = keras.models.load_model(self.output_directory+'best_model.hdf5')

        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred , axis=1)
       
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

        score = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        save_logs(self.output_directory, history, y_pred, y_true, 0.0)

        
        cm_plot_labels = [ '0','1']


        conf_matrix = confusion_matrix(y_true, y_pred)

        self.plot_confusion_matrix(cm          =  conf_matrix, 
                            normalize    = False,
                            target_names = cm_plot_labels,
                            title        = "Confusion Matrix")

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
