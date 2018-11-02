import soundfile as sf

import numpy as np

from pydub import AudioSegment

import time

start_time = time.time()



#Extracting array from audio file,  ################################################################

data1, samplerate1=sf.read('inputfile.wav')

print(data1.shape,samplerate1)

data2, samplerate2=sf.read('noise.wav')

print(data2.shape, samplerate2)

#Merging noise and data  ############################################################################

sound1=AudioSegment.from_file('inputfile.wav')
sound2=AudioSegment.from_file('noise.wav')

combined=sound1.overlay(sound2)

combined.export('combine.wav', format='wav')

data3, samplerate3 = sf.read('combine.wav')

print(data3.shape, samplerate3)

#Initialize weights randomly  ########################################################################

np.random.seed(1)

WEIGHTS = np.random.random((2,1))-1

print( "Random Weights before training", WEIGHTS)




output_data=[]

#Activation Functions

def linearfunc(x):
    if (x >= 0):
        return x
    else:
        return 0.01*x


learning_rate=0.01

#Learning Process - Training Data

for iter in range(100):

    for originalvalue,noisedata, combined in zip(data1,data2,data3):


        ada_output=(combined*WEIGHTS[0])+(noisedata*WEIGHTS[1])

        ada_output=linearfunc(ada_output)

        error_value=originalvalue-ada_output

        # Delta Rule Implementation

        WEIGHTS[0]=WEIGHTS[0]+learning_rate*error_value*combined
        WEIGHTS[1] = WEIGHTS[1] + learning_rate * error_value * noisedata

    print("--- epoch %s---"%iter)

    print("Weights  ", WEIGHTS)

#Appending Output data

for noisedata, combined in zip(data2, data3):

    finaloutput=(combined*WEIGHTS[0])+(noisedata*WEIGHTS[1])

    output_data.append(finaloutput)



output_data=np.asarray(output_data)

print(output_data.shape)

#Writing the sound output

sf.write('outputfile.wav', output_data,samplerate1)


print("--- %s seconds ---" % (time.time() - start_time))








