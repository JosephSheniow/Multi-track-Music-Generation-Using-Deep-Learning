import os
import glob
import pickle
import numpy
import matplotlib.pyplot as plt
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint 


def train_net():
    
    notes = get_notes()

    n_alpha = len(set(notes)) #note quantity

    net_input, net_output = pre_seq(notes, n_alpha)

    model = create_net(net_input, n_alpha)

    train(model, net_input, net_output)


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

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath) #save notes to file

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
    net_input = numpy.reshape(net_input, (n_patterns, seq_length, 1))
    #normalize
    net_input = net_input / float(n_alpha)

    net_output = np_utils.to_categorical(net_output) #one hot 
    
    return (net_input, net_output)


def create_net(net_input, n_alpha):

    model = Sequential()
    model.add(LSTM(512,input_shape=(net_input.shape[1], net_input.shape[2]),return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(n_alpha)) #match nodes amount
    model.add(Activation('softmax'))
    adam = Adam(lr=0.0001)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['acc']) #define loss, optimizer
    return model


def train(model, net_input, net_output):

    filepath = "weights-epoch-{epoch:02d}-loss-{loss:.4f}.hdf5"
    #save model after each epoch
    checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose=0,save_best_only=True,mode='min')
    callbacks_list = [checkpoint]
    history = model.fit(net_input, net_output, epochs=500, batch_size=64, callbacks=callbacks_list)#, validation_split=0.2)

    #plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    #plt.title('Model accuracy')
    #plt.ylabel('Accuracy')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()

    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('Model loss')
    #plt.ylabel('Loss')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()

if __name__ == '__main__':
    train_net()