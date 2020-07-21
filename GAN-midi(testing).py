from __future__ import print_function, division
import os
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  
from music21 import converter, instrument, note, chord
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, CuDNNLSTM, Reshape, Activation, Input, LeakyReLU, BatchNormalization, Bidirectional
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint 


def get_notes():
	
	notes = []

	for file in glob.glob("midi_files/*.mid"):
		midi = converter.parse(file) #read midi file

		print("Parsing %s" % file)

		notes_to_parse = None

		try: 
			s2 = instrument.partitionByInstrument(midi)
			notes_to_parse = s2.parts[0].recurse() 
		except: 
			notes_to_parse = midi.flat.notes

		for element in notes_to_parse:
			if isinstance(element, note.Note):
				notes.append(str(element.pitch))
			elif isinstance(element, chord.Chord):
				notes.append('.'.join(str(n) for n in element.normalOrder))

	return notes


def pre_seq(notes, n_alpha):#prepare sequences
	

	seq_length = 100 #100 previous notes help prediction
	#get pitch name
	pitchnames = sorted(set(item for item in notes))
	#mapping pitch to int
	note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

	net_input = []
	net_output = []
	#create input sequences and its output
	for i in range(0, len(notes) - seq_length, 1):
		seq_in = notes[i:i + seq_length]
		seq_out = notes[i + seq_length]
		net_input.append([note_to_int[char] for char in seq_in])
		net_output.append(note_to_int[seq_out])

	n_patterns = len(net_input)
	#reshape
	net_input = np.reshape(net_input, (n_patterns, seq_length, 1))
	#normalize
	net_input = (net_input - float(n_alpha)/2)/(float(n_alpha)/2)

	net_output = np_utils.to_categorical(net_output) #one hot 

	return (net_input, net_output)


def generate_notes(model, net_input, n_alpha):

	start = np.random.randint(0, len(net_input)-1)

	pitchnames = sorted(set(item for item in notes))
	#mapping int to pitch
	int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

	pattern = net_input[start]

	predict_output = []

	#generate notes
	for note_index in range(500):
		predict_input = np.reshape(pattern, (1, len(pattern), 1))
		predict_input = predict_input / float(n_alpha)

		prediction = model.predict(predict_input, verbose=0)

		index = np.argmax(prediction)
		result = int_to_note[index]
		predict_output.append(result)

		pattern = np.append(pattern,index)
		#pattern.append(index)
		pattern = pattern[1:len(pattern)]

	return predict_output


def create_midi(predict_output, filename):

	offset = 0
	output_notes = []

	for item in predict_output:
		pattern = item[0]
		#pattern is chord
		if ('.' in pattern) or pattern.isdigit():
			notes_in_chord = pattern.split('.')
			notes = []
			for current_note in notes_in_chord:
				new_note = note.Note(int(current_note))
				new_note.storedInstrument = instrument.Piano()
				notes.append(new_note)
			new_chord = chord.Chord(notes)
			new_chord.offset = offset
			output_notes.append(new_chord)
		#pattern is note
		else:
			new_note = note.Note(pattern)
			new_note.offset = offset
			new_note.storedInstrument = instrument.Piano()
			output_notes.append(new_note)

		offset += 0.5

	midi_stream = stream.Stream(output_notes)

	midi_stream.write('midi', fp='test_output.mid')


class GAN():

	def __init__(self, rows):

		self.seq_length = rows
		self.seq_shape = (self.seq_length , 1)
		self.latent_dim = 1000
		self.disc_loss = []
		self.gen_loss = []

		optimizer = Adam(lr=0.0002, beta_1=0.5)

		#build discriminator
		self.discriminator = self.discriminator_net()
		self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

		#build generator
		self.generator = self.generator_net()

		#generator input (noise)
		z = Input(shape=(self.latent_dim,))
		generated_seq = self.generator(z)

		#Stop training discriminator while training combined model
		self.discriminator.trainable = False

		validity = self.discriminator(generated_seq)

		#train combined model
		self.combined = Model(z, validity)
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


	def discriminator_net(self):

		model = Sequential()
		model.add(CuDNNLSTM(512, input_shape=self.seq_shape, return_sequences=True))
		model.add(Bidirectional(CuDNNLSTM(512)))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(256))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()

		seq = Input(shape=self.seq_shape)
		validity = model(seq)

		return Model(seq, validity)


	def generator_net(self):

		model = Sequential()
		model.add(Dense(256, input_dim=self.latent_dim))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(np.prod(self.seq_shape), activation='tanh'))
		model.add(Reshape(self.seq_shape))
		model.summary()

		noise = Input(shape=(self.latent_dim,))
		seq = model(noise)

		return Model(noise, seq)


	def train(self, epochs, batch_size=128, sample_interval=50):

		notes = get_notes()
		n_alpha = len(set(notes)) #note quantity
		X_train, y_train = pre_seq(notes, n_alpha)

		real = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		#Training model
		for epoch in range(epochs):

			# Training the discriminator(Select a random batch of note sequences)
			idx = np.random.randint(0, X_train.shape[0], batch_size)
			real_seqs = X_train[idx]

			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
			# Generate a batch of new note sequences
			gen_seqs = self.generator.predict(noise)
			# Train the discriminator
			d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
			d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


			#  Training the Generator
			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
			# Train the generator (discriminator label samples as real)
			g_loss = self.combined.train_on_batch(noise, real)

			# Print progress & save loss lists
			if epoch % sample_interval == 0:
				print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
				self.disc_loss.append(d_loss[0])
				self.gen_loss.append(g_loss)

		self.generate(notes)
		self.plot_loss()


	def generate(self, input_notes):
		# Get pitch names and store in a dictionary
		notes = input_notes
		pitchnames = sorted(set(item for item in notes))
		int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

		# Use random noise to generate sequences
		noise = np.random.normal(0, 1, (1, self.latent_dim))
		predictions = self.generator.predict(noise)

		pred_notes = [x*242+242 for x in predictions[0]]
		pred_notes = [int_to_note[int(x)] for x in pred_notes]

		create_midi(pred_notes, 'gan_final')

	def plot_loss(self):
		plt.plot(self.disc_loss, c='red')
		plt.plot(self.gen_loss, c='blue')
		plt.title("GAN loss per epoch")
		plt.legend(['Discriminator', 'Generator'])
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.savefig('GAN_loss_per_epoch_final.png', transparent=True)
		plt.close()

if __name__ == '__main__':
	gan = GAN(rows=100)
	gan.train(epochs=200, batch_size=32, sample_interval=1)
