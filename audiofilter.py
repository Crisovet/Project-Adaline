import soundfile as sf

import numpy as np

from pydub import AudioSegment

import time

import scipy

from scipy.io.wavfile import read



import matplotlib.pyplot as plt

import wave
import sys


start_time = time.time()

#SNR ratio function##################

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


#Plotting function ########################################

def plotaudio(file):
    spf = wave.open(file,'r')

    #Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')


    #If Stereo
    if spf.getnchannels() == 2:
        print ('Just mono files')
        sys.exit(0)

    plt.figure(1)
    if(file=="inputfile.wav"):
        plt.title('Input Wave')
    elif(file=="noise.wav"):
        plt.title('Noise wave')
    elif(file=="combine.wav"):
        plt.title('Combined sound Wave')
    elif(file=="outputfile.wav"):
        plt.title('Output Wave')
    plt.plot(signal)
    plt.show()



#Extracting array from audio file,  ################################################################

data1, samplerate1=sf.read('inputfile.wav')

print(data1.shape,samplerate1)


plotaudio('inputfile.wav')


data2, samplerate2=sf.read('noise.wav')

print(data2.shape, samplerate2)

plotaudio('noise.wav')

#Merging noise and data  ############################################################################

sound1=AudioSegment.from_file('inputfile.wav')
sound2=AudioSegment.from_file('noise.wav')

combined=sound1.overlay(sound2)

combined.export('combine.wav', format='wav')

data3, samplerate3 = sf.read('combine.wav')

print(data3.shape, samplerate3)

plotaudio("combine.wav")

print(signaltonoise(data3))

#Initialize weights randomly  ########################################################################

np.random.seed(1)

WEIGHTS = np.random.random((2,1))-1

# data4=np.random.normal(0,0.1, data1.shape[0])
#
# sf.write('gaussiannoise.wav', data4, samplerate1)
#
# sound3=AudioSegment.from_file('inputfile.wav')
# sound4=AudioSegment.from_file('gaussiannoise.wav')
#
# combinedgau=sound3.overlay(sound4)
#
# combinedgau.export('combinegau.wav', format='wav')
#
# data5, samplerate5 = sf.read('combinegau.wav')









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

for iter in range(50):

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

print("SNR", signaltonoise(output_data))

#Writing the sound output

sf.write('outputfile.wav', output_data,samplerate1)

plotaudio("outputfile.wav")


#On new data ##############################################


sound1=AudioSegment.from_file('newinput.wav')
sound2=AudioSegment.from_file('noise.wav')

combined=sound1.overlay(sound2)

combined.export('newcombine.wav', format='wav')

data4, samplerate3 = sf.read('newcombine.wav')

print(data4.shape, samplerate3)

newoutputaudio=[]

for noisedata, combined in zip(data2, data4):

    finaloutput=(combined*WEIGHTS[0])+(noisedata*WEIGHTS[1])

    newoutputaudio.append(finaloutput)

newoutputaudio=np.asarray(newoutputaudio)

sf.write('newoutputfile.wav', newoutputaudio,samplerate1)


print("--- %s seconds ---" % (time.time() - start_time))








