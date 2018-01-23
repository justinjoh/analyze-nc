import numpy as np
from scipy.io import netcdf
from scipy.stats import expon
from matplotlib.pyplot import *
import os


# Read nc file (this function was written by Chase Goddard)
def read_netcdf(directory, f_name):
    """Read in a netcdf (.nc) file from disk and return the data contained
    in the form of a numpy structured array.
    The file to be read is (directory + f_name)
    The result is in the format:
        array([(time0, E0, channel0), (time1, E1, channel1), ...])
    The result is a structured array, so (e.g.) Energy data can be accessed by
    result['E']. The same holds for other columns of the data"""

    f = netcdf.netcdf_file(directory + f_name, 'r', mmap=False)  # Load in netcdf file
    tmp_data = f.variables['array_data']  # get relevant data
    data = tmp_data[:].copy().astype('uint16')  # copy data into memory from disk
    f.close()  # close netcdf file

    data = data.ravel()  # flatten data

    # Get number of events. 2^16 scales word 67 according to xMAP file format
    # word 66 and 67 are the header words that contain the number of events
    num_events = data[66].astype('int32') + \
                 (2 ** 16) * (data[67].astype('int32'))
    time_constant = 20e-9  # conversion factor from xMAP time units to seconds

    # size of header, in words
    offset = np.array([256]).astype('int32')

    # set up vectors to store data
    E = np.zeros(num_events)
    channel = np.zeros(num_events)
    time = np.zeros(num_events, dtype='float64')

    # keep track of how much data we have processed
    # start by skipping the header
    dynamic_offset = offset

    for i in range(0, num_events):
        dynamic_offset = offset + 3 * i  # we process 3 words each iteration

        # xMAP stores data in specific bits of these words
        word1 = data[dynamic_offset]
        word2 = data[dynamic_offset + 1]
        word3 = data[dynamic_offset + 2]

        # extract channel bits (13-15 of word 1)
        channel[i] = np.bitwise_and(np.right_shift(word1, 13), 3)
        # extract energy bits (0-12 of word 1)
        E[i] = np.bitwise_and(word1, 0x1fff)
        # extract time bits
        time[i] = (word3 * (2 ** 16) + word2) * time_constant

    # package data into table format
    return np.array(list(zip(time, E, channel)),
                    dtype={'names':  ['time', 'E', 'channel'],
                           'formats': ['float64', 'int32', 'int32']})

def count_fluo(spectrum, ROI_start, ROI_end):
    # return number of counts in bins between ROI_start and ROI_end.
    # Be sure that are using units correctly, i.e. detector bins vs. electron volts
    count = 0
    for i in range(0, len(spectrum)):
        num = spectrum[i]
        if (num <= ROI_end and num >= ROI_start):
            count += 1
    return count
    #for i in range(0, len(spectrum)):
#
#        num = spectrum[i]
#        try:
#            num = np.int64(num)
#        except:
#            pass
#        if (num <= ROI_end & num >= ROI_start):
#            print num
#            count += 1
#    return count


def subtract(spectrum, toSubtract):
    # subtract toSubtract from spectrum
    numChannels = 2048
    new = np.zeros(numChannels)
    for i in range(0, numChannels):
        new[i] = spectrum[i]-toSubtract[i]
    return new


def convert_to_eV(spectrum, numChannels, maxEnergy):
    # return spectrum but in units of energy, not channel
    lenSpectrum = len(spectrum)
    newSpectrum = np.zeros(lenSpectrum)
    factor = maxEnergy/numChannels
    for i in range(0, lenSpectrum):
        newSpectrum[i] = spectrum[i]*factor
    return newSpectrum


def plot_spectrum(spectrum, bins, range):
    # show a Counts vs. Energy (eV) histogram
    hist(spectrum, bins, range)
    xlabel('Energy (eV)')
    ylabel('Counts')
    show()

if __name__ == '__main__':
    # commandline use: $ python analyze_nc.py filename.nc

    numChannels = 2048
    maxEnergy = 70000

    try:
        data = read_netcdf("", sys.argv[1])
    except:
        data = read_netcdf("", sys.argv[1] + ".nc")
    energies = data['E']

    energieseV = convert_to_eV(energies, numChannels, maxEnergy) # convert the energies to electron volts for easier visualization

    hist(energieseV, bins=numChannels,range=(0,maxEnergy))

    ROI_start = 0
    ROI_end = 99999
    print("Counts between " + str(ROI_start) + " and " + str(ROI_end) + ": " + str(count_fluo(energieseV, ROI_start, ROI_end)))

    # plot_spectrum(energieseV, bins=numChannels, range=(0, maxEnergy))

    # for each Am + Al spectrum, subtract detector alone and add --> corrected1
    corrected1 = np.zeros(numChannels)

    for filename in os.listdir(os.getcwd() + "/AmAl/"):
    # Use this to add different spectra for now. Eventually want to use for creating "weighted averages"
        dir = os.getcwd() + "/AmAl/"
        totalPhotons = 0 # total number of photons
        allAmAl = np.zeros(0)
        if filename.endswith(".nc"):
            energies = read_netcdf(dir,  filename)['E']
            numPhotons = len(energies)
            # for photon in file, add to allAmAl bin
            for i in range(0, numPhotons):
                np.append(allAmAl, energies[i])
    plot_spectrum(allAmAl, bins=numChannels, range=(0, maxEnergy))

    # for the corrected1, calculate k-alpha fluorescence rate (np array) --> corrected2

    # for each Am + Al + Cu spectrum
    # subtract corresponding corrected2