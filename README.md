# Multi-track-Music-Generation-Using-Deep-Learning
> LSTM neural network application using Keras

In this repository, we gather several kinds of music data belonging to the same music genre, and use Long Short-Term Memory(LSTM) model for training our digital music data, which is recorded in Musical Instrument Digital Interface(MIDI) format. The purpose of this experiment is to build an efficiency training model for single music track and multi music tracks.

## Getting Started

1. Create a folder named `data` to store the note information for parsing midi files.
2. Create a folder named `midi_files` and put in all the midi files as training data.
3. Train the LSTM network with [LSTM-midi-net.py](#) and obtain the weights stored as filename called `weights-hdf5`.
4. Generate single track midi track with [LSTM-midi-predict.py](#) and the midi file will be save to `test_output-mid`.
5. Import the midi files into DAW such as CUBASE (Windows) of LOGIC (OS X) for adding sofware instrunments and export MP3 music files.


![image](https://github.com/JosephSheniow/Multi-track-Music-Generation-Using-Deep-Learning/blob/master/image/Multi-track-DAW.png)


## LSTM Network Model


![image](https://github.com/JosephSheniow/Multi-track-Music-Generation-Using-Deep-Learning/blob/master/image/LSTM-Network-Model.png)

## Music Generation Flow
### Single-track

![image](https://github.com/JosephSheniow/Multi-track-Music-Generation-Using-Deep-Learning/blob/master/image/Single-track-music-generation.png)

### Multi-track

![image](https://github.com/JosephSheniow/Multi-track-Music-Generation-Using-Deep-Learning/blob/master/image/Multi-track-music-generation.png)

