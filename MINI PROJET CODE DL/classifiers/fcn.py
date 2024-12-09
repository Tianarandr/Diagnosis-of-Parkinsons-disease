import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 

from utils.utils import save_logs
from utils.utils import calculate_metrics

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from numpy.testing import assert_allclose
from keras.layers import Conv2D, BatchNormalization, MaxPooling1D, Conv1D
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score 
import itertools
from keras.utils import np_utils


class Classifier_FCN:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=False,build=True):
		self.output_directory = output_directory
		if build == True:
			self.model = self.build_model(input_shape, nb_classes)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			self.model.save_weights(self.output_directory+'model_init.hdf5')
		return

	def build_model(self, input_shape, nb_classes):
		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
		conv1 = keras.layers.BatchNormalization()(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)

		conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
		conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)

		gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

		output_layer = keras.layers.Dense(nb_classes, activation='sigmoid')(gap_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='binary_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)

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
		batch_size = 106
		nb_epochs = 200

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 
		
		Y_train = np_utils.to_categorical(y_train)
		Y_test = np_utils.to_categorical(y_val)
		print(y_val.shape, y_train.shape)

		history = self.model.fit(x_train, Y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,Y_test), callbacks=self.callbacks)
		
		duration = time.time() - start_time

		self.model.save(self.output_directory+'last_model.hdf5')

		model = keras.models.load_model(self.output_directory+'best_model.hdf5')

		y_pred = model.predict(x_val)

		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

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

		score = model.evaluate(x_val, Y_test, verbose=1)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])

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