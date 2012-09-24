#!/usr/bin/env python

import sys

import numpy
numpy.seterr(under='ignore')
import scipy.io
import scipy.fftpack
import scipy.signal
import pylab


import stft


def spectral_subtraction_arrays(data_array, noise_array, sample_rate):
    #data_array = data_array / float(numpy.iinfo(data_array.dtype).max)
    #noise_array = data_array / float(numpy.iinfo(noise_array.dtype).max)
    length = len(data_array)

    smallsize = int(len(noise_array) / length)
    noise_data = noise_array[:length*smallsize]
    noise_data = noise_data.reshape(smallsize, length)

    #window = scipy.signal.get_window("blackmanharris", length)
    window = scipy.signal.get_window("hamming", length)
    noise_data *= window
    noise_ffts = scipy.fftpack.fft(noise_data, axis=1)
    noise_ffts_abs = abs(noise_ffts)
    noise_power = noise_ffts_abs**2

    means_power = numpy.mean(noise_power, axis=0)

    # power to dB
    noise_db = stft.amplitude2db( numpy.sqrt(means_power[:len(means_power)/2+1])/ length )
    freqs = [stft.bin2hertz(i, sample_rate, length) for i in range(len(noise_db))]


    fft = scipy.fftpack.fft(data_array*window)
    fft_abs = abs(fft)[:len(fft)/2+1]

    fft_db = stft.amplitude2db( fft_abs / length )

    #reconstructed = fft - mins_full
    #reconstructed = fft
    #reconstructed = numpy.zeros(len(fft), dtype=complex)
    #for i in range(len(reconstructed)):
    theta = numpy.angle(fft)
    alpha = 1.0
    beta = 0.01
    r = numpy.zeros(len(fft))
    for i in range(len(fft)):
        r[i] = (abs(fft[i])**2 - alpha * means_power[i])
        if r[i] < beta*means_power[i]:
            r[i] = beta * means_power[i]
#        else:
#            r[i] = numpy.sqrt(r[i])
        #print r_orig[i], means_full[i], r[i]
    r = numpy.sqrt(r)

    reconstructed = ( r * numpy.cos(theta)
        + r * numpy.sin(theta)*1j);

    rec_abs = abs(reconstructed)[:len(reconstructed)/2+1]
    #print r_abs

    rec_db = stft.amplitude2db( rec_abs / length )


    reconstructed_sig = scipy.fftpack.ifft(reconstructed )

    reconstructed_sig /= window
    reconstructed_sig = numpy.real(reconstructed_sig)

    median_sig = scipy.signal.medfilt(rec_db, 5)
    median_sig_orig = scipy.signal.medfilt(fft_db, 5)

    if False:
    #if True:
        #pylab.plot(freqs, noise_db, label="mean noise")
        pylab.plot(freqs, fft_db, label="sig orig")
        pylab.plot(freqs, rec_db,
            label="reconstructed")
        pylab.plot(freqs, median_sig_orig,
            label="median orig")
        pylab.plot(freqs, median_sig,
            label="median rec.")
        pylab.legend()
        pylab.show()


    return reconstructed_sig 

def spectral_subtraction(wav_filename, noise_filename):
    sample_rate, wav_data = scipy.io.wavfile.read(wav_filename)
    wav_data = wav_data / float(numpy.iinfo(wav_data.dtype).max)
    hopsize = len(wav_data)

    ##noise, freqs, means, mins, stds = calc_noise.get_noise(
    ##    #wav_filename, noise_filename, recalc=True)
    #    wav_filename, noise_filename, recalc=False,
    #    bins=len(wav_data))

    sample_rate, noise_data = scipy.io.wavfile.read(noise_filename)
    noise_data = noise_data / float(numpy.iinfo(noise_data.dtype).max)
    smallsize = int(len(noise_data) / hopsize)
    noise_data = noise_data[:hopsize*smallsize]
    noise_data = noise_data.reshape(smallsize, hopsize)
    #window = scipy.signal.get_window("blackmanharris", hopsize)
    window = scipy.signal.get_window("hamming", hopsize)
    noise_data *= window
    noise_ffts = scipy.fftpack.fft(noise_data, axis=1)
    noise_ffts_abs = abs(noise_ffts)
    noise_power = noise_ffts_abs**2

    #mins = numpy.min(noise_ffts_abs, axis=0)
    #mins_full = mins
    #mins = mins[:len(mins)/2+1]
    means_power = numpy.mean(noise_power, axis=0)
    #means_abs = numpy.abs(means) [:len(means)/2+1]

    noise_db = stft.amplitude2db( numpy.sqrt(means_power[:len(means_power)/2+1])/hopsize )
    #noise_db = 10*numpy.log10(means_power[:len(means_power)/2+1] / hopsize )
    freqs = [float(i)*(sample_rate) / hopsize for i in range(len(noise_db))]

    #for i in range(len(means_abs)):
    #    print means_abs[i]
    #print means_abs
    #print means_abs.shape
    #pylab.semilogy(mins)

    fft = scipy.fftpack.fft(wav_data*window)
    #fft = scipy.fftpack.fft(wav_data)
    fft_abs = abs(fft)[:len(fft)/2+1]

    #pylab.plot(noise_ffts_abs[0][:len(fft)/2+1])
    #pylab.plot(noise_ffts_abs[1][:len(fft)/2+1])
    #pylab.plot( numpy.sqrt(means_power[:len(fft)/2+1]) )
    #pylab.plot(fft_abs)
    #pylab.show()

    #fft_power = fft_abs**2
#    print len(fft)
#    print len(mins)

    fft_db = stft.amplitude2db( fft_abs / hopsize )

    #reconstructed = fft - mins_full
    #reconstructed = fft
    #reconstructed = numpy.zeros(len(fft), dtype=complex)
    #for i in range(len(reconstructed)):
    theta = numpy.angle(fft)
    alpha = 1.0
    beta = 0.1
    r = numpy.zeros(len(fft))
    for i in range(len(fft)):
        r[i] = (abs(fft[i])**2 - alpha * means_power[i])
        if r[i] < 0:
            r[i] = beta * means_power[i]
        else:
            r[i] = numpy.sqrt(r[i])
        #print r_orig[i], means_full[i], r[i]

    reconstructed = ( r * numpy.cos(theta)
        + r * numpy.sin(theta)*1j);

    rec_abs = abs(reconstructed)[:len(reconstructed)/2+1]
    #print r_abs

    rec_db = stft.amplitude2db( rec_abs / hopsize )

    reconstructed_sig = scipy.fftpack.ifft(reconstructed )

    reconstructed_sig /= window
    reconstructed_sig = numpy.real(reconstructed_sig)

    #pylab.figure()
    #pylab.plot(reconstructed_sig)


    # FIXME: don't normalize
    #reconstructed_sig /= max(reconstructed_sig)

    big = numpy.int16(reconstructed_sig * numpy.iinfo(numpy.int16).max)
    #pylab.figure()
    #pylab.plot(big)

    scipy.io.wavfile.write("foo.wav", sample_rate, big)

    if True:
        pylab.plot(freqs, noise_db, label="mean noise")
        pylab.plot(freqs, fft_db, label="sig orig")
        pylab.plot(freqs, rec_db,
            label="reconstructed")
        pylab.legend()
        pylab.show()



if __name__ == "__main__":
    wav_filename = sys.argv[1]
    noise_filename = sys.argv[2]
    spectral_subtraction(wav_filename, noise_filename)


