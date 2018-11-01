import soundfile as sf

import numpy as np

from pydub import AudioSegment


#Extracting array from audio file,

data1, samplerate1=sf.read('inputfile.wav')

print(data1.shape,samplerate1)

data2, samplerate2=sf.read('noise.wav')

print(data2.shape, samplerate2)

#Merging noise and data

sound1=AudioSegment.from_file('inputfile.wav')
sound2=AudioSegment.from_file('noise.wav')

combined=sound1.overlay(sound2)

combined.export('combine.wav', format='wav')

data3, samplerate3 = sf.read('combine.wav')

print(data3.shape, samplerate3)

#Initial weights randomly

np.random.seed(1)

WEIGHTS = 2*np.random.random((2,1))-1

print( "Random Weights before training", WEIGHTS)


error=[]

output_data=[]

#Activation Functions

def linearfunc(x):
    if (x >= 0):
        return x
    else:
        return 0


def step(x):
    if (x > 0):
        return 1
    else:
        return -1;

def sigmoid(x):
    return 1/(1+np.exp(-x))


learning_rate=0.05

#Learning Process - Training

for iter in range(100):

    for originalvalue,noisedata, combined in zip(data1,data2,data3):


        ada_output=(combined*WEIGHTS[0])+(noisedata*WEIGHTS[1])

        ada_output=linearfunc(ada_output)

        # Delta Rule Implementation

        error_value=originalvalue-ada_output


        error.append(error_value)

        WEIGHTS[0]=WEIGHTS[0]+learning_rate*error_value*combined
        WEIGHTS[1] = WEIGHTS[1] + learning_rate * error_value * noisedata



print("Weights after training ", WEIGHTS)

#Appending Output data

for noisedata, combined in zip(data2, data3):
    finaloutput=(combined*WEIGHTS[0])+(noisedata*WEIGHTS[1])

    finaloutput=linearfunc(finaloutput)

    output_data.append(finaloutput)



output_data=np.asarray(output_data)
print(output_data.shape)

output_data=output_data.reshape(output_data.shape[0],1)

print(output_data.shape)

#Writing the sound output

sf.write('outputfile.wav', output_data,samplerate1)







