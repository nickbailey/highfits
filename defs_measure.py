#!/usr/bin/env python

import pyaudio
import numpy

#RATE = 96000
#ALSA_BUFFER_SIZE = 4096
#PROCESS_BUFFER_SIZE = 8192

OUTPUT_RATE = 44100
#INPUT_RATE = 96000
#INPUT_RATE = 48000
INPUT_RATE = 44100

#ALSA_BUFFER_SIZE = 2048
#ALSA_BUFFER_SIZE = 256
#ALSA_BUFFER_SIZE = 512
ALSA_BUFFER_SIZE = 1024
#PROCESS_BUFFER_SIZE = 16384
#ALSA_BUFFER_SIZE_MULTIPLIER = 1

# don't change these!  portaudio and/or pyaudio is really finicky
# about sample formats :(
FORMAT_ALSA = pyaudio.paFloat32
FORMAT_NUMPY = numpy.float32
#FORMAT_ALSA = pyaudio.paInt16
#FORMAT_NUMPY = numpy.int16
FORMAT_ALSA_OUT = pyaudio.paInt16
FORMAT_NUMPY_OUT = numpy.int16


