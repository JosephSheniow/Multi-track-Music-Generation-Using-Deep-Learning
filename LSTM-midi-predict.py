import tensorflow as tf
import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.optimizers import Adam


def generator():
    
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath) #load  notes from file

    pitchnames = sorted(set(item for item in notes))
   
    n_alpha = len(set(notes))

    net_input, normalized_input = pre_seq(notes, pitchnames, n_alpha)

    model = create_net(normalized_input, n_alpha)

    predict_output = generate_notes(model, net_input, pitchnames, n_alpha)

    create_midi(predict_output)


def pre_seq(notes, pitchnames, n_alpha):
    #mapping pitch to int
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    seq_length = 100
    net_input = []
    output = []

    for i in range(0, len(notes) - seq_length, 1):
        seq_in = notes[i:i + seq_length]
        seq_out = notes[i + seq_length]
        net_input.append([note_to_int[char] for char in seq_in])
        output.append(note_to_int[seq_out])

    n_patterns = len(net_input)
    #reshape
    normalized_input = numpy.reshape(net_input, (n_patterns, seq_length, 1))
    #normalize
    normalized_input = normalized_input / float(n_alpha)

    return (net_input, normalized_input)


def create_net(net_input, n_alpha):

    model = Sequential()
    model.add(LSTM(512,input_shape=(net_input.shape[1], net_input.shape[2]),return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(n_alpha))
    adam = Adam(lr=0.0001)
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam)

    model.load_weights('weights.hdf5')
    
    return model


def generate_notes(model, net_input, pitchnames, n_alpha):

    start = numpy.random.randint(0, len(net_input)-1)
    #mapping int to pitch
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = net_input[start]

    predict_output = []

    #generate notes
    for note_index in range(300):
        predict_input = numpy.reshape(pattern, (1, len(pattern), 1))
        predict_input = predict_input / float(n_alpha)

        prediction = model.predict(predict_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        predict_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return predict_output


def create_midi(predict_output):

    offset = 0
    output_notes = []

    for pattern in predict_output:
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


#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  


if __name__ == '__main__':
    generator()