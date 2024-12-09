import keras as keras
import tensorflow as tf
import numpy as np
import time

import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt 

from utils.utils import save_logs
from utils.utils import calculate_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score 
import itertools
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from numpy.testing import assert_allclose
from tensorflow.keras.optimizers import SGD

class Classifier_MLP:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=False,build=True):
		self.output_directory = output_directory
		if build == True:
			self.model = self.build_model(input_shape, nb_classes)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			self.model.save_weights(self.output_directory + 'model_init.hdf5')
		return

	def build_model(self, input_shape, nb_classes):

		batch_size = 105
		nb_epochs = 200

		model = Sequential()
		model.add(Dense(32, input_dim=input_shape))
		model.add(Activation('relu'))
		model.add(Dropout(0.4))

		model.add(Dense(32))
		model.add(Activation('relu'))

		model.add(Dense(2))
		model.add(Activation('sigmoid'))


		#sgd = SGD(learning_rate)

		checkpointer = ModelCheckpoint(filepath='ANN.hdf5', monitor='val_loss',
                                   verbose=0, save_best_only=True, 
                                  save_weights_only=False)


		model.compile(loss='binary_crossentropy', optimizer="Adam",
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

		file_path = self.output_directory+'best_model.hdf5' 

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model

	def fit(self, x_train, y_train, x_val, y_val,y_true):
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		batch_size = 105
		nb_epochs = 200

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time()
		Y_train = np_utils.to_categorical(y_train)
		Y_test = np_utils.to_categorical(y_val)

		history = self.model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,Y_test), callbacks=self.callbacks)
		
		duration = time.time() - start_time

		self.model.save(self.output_directory + 'last_model.hdf5')

		model = keras.models.load_model(self.output_directory+'best_model.hdf5')

		y_pred = model.predict(x_val)
		

		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)
		print(y_pred)

		#save_logs(self.output_directory, history, y_pred, y_true, duration)

		keras.backend.clear_session()
		print(history.history.keys())
		plt.figure(1)
		plt.plot(history.history['accuracy'])
		plt.plot(history.history['val_accuracy'])
		plt.title('Model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		file_name = self.output_directory+'epochs_validation.png'
		plt.savefig(file_name)

		plt.figure(2)
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['train', 'test'], loc='upper left')
		file_name = self.output_directory+'epochs_loss.png'
		plt.savefig(file_name)

		score = self.model.evaluate(x_val, Y_test, verbose=1)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

        #save_logs(self.output_directory, history, y_pred, y_true, 0.0)

        
		cm_plot_labels = [ '0','1']


		conf_matrix = confusion_matrix(y_true, y_pred)

		self.plot_confusion_matrix(cm          =  conf_matrix, 
                            normalize    = False,
                            target_names = cm_plot_labels,
                            title        = "Confusion Matrix")
		return y_pred


	def predict(self, x_test, y_true,x_train,y_train,y_test,return_df_metrics = True):
		model_path = self.output_directory + 'best_model.hdf5'
		model = keras.models.load_model(model_path)
		y_pred = model.predict(x_test)
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

            plt.figure(8)
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


