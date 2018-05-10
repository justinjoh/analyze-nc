from scipy.io import netcdf
from matplotlib.pyplot import *
import numpy as np
from numpy import *
import os


without_sample_foldername = "without_sample"
with_sample_foldername = "with_sample"

numBins = 2048
maxEnergy = 70000
energyBinRatio = float(maxEnergy)/float(numBins) # number of eV per bin

# start and end of fluorescence counting region
fluo_start = 7600
fluo_end = 8200

# Read in cross-section data. Do this for all metals.
metals_dict = {"Al":"", "Ti":"", "Fe":"", "Ni":"", "Cu":"", "Zr":"", "Ag":""}
for key in metals_dict:
    temp_csv = np.genfromtxt(str(key) + "_data.csv", delimiter=",")
    sub_dict = {"energies_eV": "", "photo": "", "total": "", "kshell":"", "rhoNum" : "", "rhoMass" :"", "coef_kshell":"", "thickness":""}

    sub_dict["energies_eV"] = temp_csv[:, 0] * 1000 # convert from keV to eV, for convenience
    sub_dict["photo"] = temp_csv[:, 3]
    sub_dict["total"] = temp_csv[:, 5]
    sub_dict["kshell"] = temp_csv[:, 6]

    metals_dict[key] = sub_dict
# So, that loop takes care of what we need from the NIST data tables. Add in the rest manually.
metals_dict["Al"]["rhoNum"] = "" # TODO need to fill in all of these, may take quite a while.

metals_dict["Cu"]["rhoNum"] = 6.51*10**22  # copper number density (atoms*cm^-3)
metals_dict["Cu"]["rhoMass"] = 8.96  # copper mass density (grams*cm^-3)
metals_dict["Cu"]["coef_kshell"] = 52.6  # x-ray absorption of characteristic fluorescence
metals_dict["Cu"]["thickness"] = 40*10**(-7)  # metal thickness (centimeters). for Jan. test, was .0254 cm


csv = np.genfromtxt('Cu_data.csv', delimiter=",")
energies_eV = csv[:, 0] * 1000  # convert from keV to eV, for convenience
photo = csv[:, 3]
total = csv[:, 5]
kshell = csv[:, 6]

# other numerical constants
rhoNum = 6.51*10**22  # copper number density (atoms*cm^-3)
rhoMass = 8.96  # copper mass density (grams*cm^-3)
coef_kshell = 52.6  # mu/rho for photoabsorption of 8keV, do not need to interpolate w/ NIST values
copper_thickness = .0254  # copper thickness (centimeters)


# Read in an nc file (this function was written by Chase Goddard)
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
    energies = np.zeros(num_events)
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
        energies[i] = np.bitwise_and(word1, 0x1fff)
        # extract time bits
        time[i] = (word3 * (2 ** 16) + word2) * time_constant

    # package data into table format
    return np.array(list(zip(time, energies, channel)),
                    dtype={'names':  ['time', 'E', 'channel'],
                           'formats': ['float64', 'int32', 'int32']})


def list_to_spectrum(some_phot_list):
    """Takes input of photon list (i.e. result of read_netcdf) and returns a <numBins>x1 numpy array.
    Each of those <numBins. entries stores the number of photons in that energy bin."""

    new_spectrum = np.zeros((numBins, 1))  # initialize spectrum as list of <numBins> zeros
    # then iterate through photon list
    num_photons = len(some_phot_list)
    for i in range(0, num_photons-1):
        this_bin = int(some_phot_list[i])
        this_index = min(int(this_bin), numBins-1)
        new_spectrum[this_index] += 1  # increment the number of photons in the corresponding bin by 1

    return new_spectrum


def add_spectra_from_folder(folder_name):
    """Return a spectrum format consisting of all folders in folder <foldername>
    Folder  <foldername> should be in the same folder as this python file."""

    this_dir = os.getcwd() + "/" + folder_name + "/"
    added_spectra = np.zeros((2048, 1))

    for filename in os.listdir(this_dir):
        if filename.endswith(".nc"):
            temp_phot_list = read_netcdf(this_dir, filename)['E']  # read in this .nc as photon list
            single_spectrum = list_to_spectrum(temp_phot_list)  # put it in spectrum form
            added_spectra = single_spectrum + added_spectra  # add to total spectrum

    return added_spectra


def count_fluo(some_spectrum):
    """"Return number of photons in some_spectrum with energy (eV) between ROI_start and ROI_end."""
    assert numBins == len(some_spectrum), "This spectrum does not have <numBins> entries; ensure that" \
        "the photon list from the netcdf file is properly sorted into spectrum format."

    count = 0  # running count of photons in desired range
    bin_start = int(fluo_start/energyBinRatio)
    bin_end = int(fluo_end/energyBinRatio)

    for i in range(bin_start, bin_end):
        count += int(some_spectrum[i])
    return count


def plot_spectrum(some_spectrum):
    assert len(some_spectrum) == numBins, "length of spectrum is not <numBins>"
    bar(range(0, numBins), height=some_spectrum)
    show()


def stdev_from_reposition():
    """Implement later"""
    pass


def plot_spectrum_with_bars():
    """Implement later"""
    pass


def beamtime_expected_fluo(thickness_nm, this_energy, incident_flux, runtime, detector_type):
    """
    thickness_nm: in nm
    this_energy: in eV
    incident_flux: in photons/second
    runtime: in seconds

    Return number of expected fluorescence photons
    Uses simplified formula, for low self-absorption of sample:
    factor = (sig_fl)(number density)(sample thickness)
    """

    assert int(detector_type) == 1 or int(detector_type) == 4, "Must type 1 or 4 for 1 or 4 element detector"

    print("Thickness in nanometers: " + thickness_nm)
    thickness_nm = int(thickness_nm)
    print("Energy of incident beam in eV: " + this_energy)
    this_energy = int(this_energy)
    print("Incident flux in photons/second: " + incident_flux)
    incident_flux = int(incident_flux)
    print("Runtime in seconds: " + str(runtime))
    runtime = int(runtime)
    # interpolation process:

    i = 0
    while int(energies_eV[i]) < int(this_energy):

        i += 1

    E1 = energies_eV[i-1]
    E2 = energies_eV[i]
    print E1
    print E2
    if kshell[i-1] == 0:
        presig_fl = 0
    else:
        b1 = math.log(kshell[i]/kshell[i-1])/math.log(E2/E1)
        presig_fl = float(kshell[i-1])*(this_energy/E1)**b1

    # presig_fl needs to be converted to cross-section. convert to barns/atom to cm^2/atom
    sig_fl = presig_fl * 1.055*10**2 * 10**(-24)  # to barns, then to cm^2
    print ("fluorescence cross section cm^2/atom: " + str(sig_fl))

    thickness_cm = thickness_nm * 10**(-7)

    if int(detector_type) == 1:
        SA = .0084
    elif int(detector_type) == 4:
        SA = 0
        print("need to add solid angle for 4-element")
    expect_fluo = SA * .44/(4*3.141592) * sig_fl * rhoNum * thickness_cm * incident_flux * runtime
    print("Expected fluo counts: " + str(expect_fluo))
    other_fluo =  SA * .44/(4*3.141592) * (sig_fl*rhoNum/(presig_fl*rhoMass))*(1-math.exp(-presig_fl*rhoMass*thickness_cm)) * incident_flux * runtime
    print other_fluo
    return expect_fluo



def fluorate_from_bin(this_energy):
    """Return \"fluorate\" as such: fluorate*(number of photons in bin) = total number of fluorescence photons
    expected to be created by the photons in this bin incident on copper sample"""

    # interpolation process:
    i = 0
    while energies_eV[i] < this_energy:
        i += 1
    # Now have overshoto with i, so energieseV[i-1] and energieseV[i] are on either side

    # begin linear interpolation
    slope = (this_energy-energies_eV[i-1])/(energies_eV[i]-energies_eV[i-1])

    presig_fl = kshell[i-1] + (kshell[i]-kshell[i-1]) * slope
    coef_energy = photo[i-1] + (photo[i]-photo[i-1])*slope

    # end linear interpolation

    # begin power law interpolation

    E1 = energies_eV[i-1]

    E2 = energies_eV[i]
    # print(math.log(E2/E1))
    # print(kshell[i-1])
#    print(kshell[i]/kshell[i-1])
    # presig_fl:
    if kshell[i-1] == 0:
        presig_fl = 0
    else:
        b1 = math.log(kshell[i]/kshell[i-1])/math.log(E2/E1)
        presig_fl = kshell[i-1]*(this_energy/E1)**b1

    # coef_energy:
    if photo[i-1] == 0:
        coef_energy = 0
    else:
        b2 = math.log(photo[i]/photo[i-1])/math.log(E2/E1)
        coef_energy = photo[i-1]*(this_energy/E1)**b2

    # end power law interpolation

    sig_fl = presig_fl*1.05521*100*10**(-24)  # unit conversion
    # ust for convenience/legibility, define a, b, z as following:
    a = sig_fl*rhoNum
    b = coef_kshell * rhoMass
    z = -coef_energy*rhoMass

    factor = (a/(z+b))*(math.exp(z * copper_thickness) - math.exp(-b * copper_thickness))
    # print("factor non-flex: " + str(factor))
    return factor  # remember, multiply this by number of photons in the bin (back in copper_fluo_rate)


def fluorate_from_bin_flex(this_energy, metal, *thickness):
    """Return \"fluorate\" as such: fluorate*(number of photons in bin) = total number of fluorescence photons
        expected to be created by the photons in this bin incident on metal sample"""

    # For convenience, since working with same metal throughout function:
    this_energieseV = metals_dict[metal]["energies_eV"]
    this_kshell = metals_dict[metal]["kshell"]
    this_photo = metals_dict[metal]["photo"]
    this_rhoNum = metals_dict[metal]["rhoNum"]
    this_rhoMass = metals_dict[metal]["rhoMass"]
    this_coef_kshell = metals_dict[metal]["coef_kshell"]
    if type(thickness) == int:
        this_thickness = thickness
    else:
        this_thickness = metals_dict[metal]["thickness"]

    # start of power law interpolation process
    i = 0
    while this_energieseV[i] < this_energy:
        i += 1
    # Now have overshot with i, so energieseV[i-1] and energieseV[i] are on either side.

    E1 = this_energieseV[i-1]
    E2 = this_energieseV[i]
    if this_kshell[i-1] == 0:
        presig_fl = 0
    else:
        b1 = math.log(this_kshell[i] / this_kshell[i - 1]) / math.log(E2 / E1)
        presig_fl = this_kshell[i - 1] * (this_energy / E1) ** b1

    # get coef_energy:
    if this_photo[i-1] == 0:
        coef_energy = 0
    else:
        b2 = math.log(this_photo[i]/this_photo[i-1])/math.log(E2/E1)
        coef_energy = this_photo[i-1]*(this_energy/E1)**b2
    # end of power law interpolation process
    sig_fl = presig_fl * 1.05521 * 100 * 10 ** (-24)  # unit conversion
    a = sig_fl * this_rhoNum
    b = this_coef_kshell * this_rhoMass
    z = -coef_energy * this_rhoMass

    factor = (a / (z + b)) * (math.exp(z * this_thickness) - math.exp(-b * this_thickness))

    return factor  # remember, multiply this by number of photons in the bin


def copper_fluo_rate(some_spectrum):
    """Returns the total number of 8keV photons expected from a spectrum incident on a copper sample.
    Does this by calling method fluorate_from_bin for each bin.
    continue writing appropriate method spec here"""

    # TODO need to add parameter for which detector (and incorporate Arthur's solid angle calculations)

    total_expected_8kev = 0  #

    for i in range(0, numBins):
        this_energy = energyBinRatio*(i+1)  # energy corresponding to this bin
        # find total absorption coefficient here with linear interpolation?
        # TODO ensure that geometric factors are in assignment to "additional"
        # .25 is from solid angle calculation (this is in physical lab book, based on
        #   information from manufacturer)
        additional = .146*fluorate_from_bin(this_energy) * some_spectrum[i]
        total_expected_8kev += additional
    return total_expected_8kev


def fluo_rate_metal(some_spectrum, metal):
    """Returns the total number of k-shell fluorescence photons expected from a spectrum incident on a metal sample.
    Does this by calling method fluorate_from_bin for each bin."""

    total_expected_kshell = 0

    for i in range(0, numBins):
        this_energy = energyBinRatio*(i+1)  # energy corresponding to this bin
        # TODO this is where to put in solid angle factor
        SA_factor = 0  # change this
        additional = SA_factor*fluorate_from_bin_flex(this_energy, metal) * some_spectrum[i]
        total_expected_kshell += additional
    return total_expected_kshell


def print3(num):
    print "    {:0.3f}".format(num)

if __name__ == '__main__':

    if len(sys.argv) == 3 and sys.argv[1].lower() == "single":
        """Single-file inspection mode: histogram, print total number of photons.
        Example cmdline usage:
        $ python analyze_nc.py single list_00129.nc """

        print("Single-file inspection of \"" + str(sys.argv[2]) + "\"")

        phot_list = read_netcdf("", sys.argv[2])['E']
        spectrum = list_to_spectrum(phot_list)

        plot_spectrum(spectrum)
        total_photons = len(phot_list)
        print("Number of photons in this .nc file: " + str(total_photons))
        total_fluo = count_fluo(spectrum)
        print("Fluo counts in this .nc file: " + str(total_fluo))
        print("Fluo divided by total: " + str(float(total_fluo)/float(total_photons)))

    elif len(sys.argv) == 2 and sys.argv[1].lower() == "repositioning":
        print("Repositioning analysis not yet implemented")
        pass

    elif len(sys.argv) == 2 and type(sys.argv[1] == int):
        """Multi-file inspection mode: for analyzing spectrum, fluorescence counts etc.
        Example cmdline usage:
        $ python analyze_nc.py """

        print("Running multi-file inspection mode")

        in_nm = float(sys.argv[1])

        copper_thickness = float(in_nm)*10**(-7)
        metals_dict["Cu"]["thickness"] = float(in_nm)*10 ** (-7)

        print("Running with copper thickness " + str(in_nm) + " nanometers")

        # Step 1: for each without-sample spectrum, add together
        without_sample_spectrum = add_spectra_from_folder(without_sample_foldername)

        # Step 2: calculate expected k-alpha fluo. rate (looks at each energy bin, because different incident energies
        #   interact differently with the copper sample
        expected_fluo = copper_fluo_rate(without_sample_spectrum)
        # divide by total number of photons to "normalize" for comparison

        norm_expected_fluo = float(expected_fluo)/float(len(without_sample_spectrum))
        print("Normalized expected fluorescence: ")
        print3(norm_expected_fluo)

        # Step 3: add up with-sample spectrum count, and count_flu_photons (normalize)
        with_sample_spectrum = add_spectra_from_folder(with_sample_foldername)

        try:
            # Count all fluorescence photons from with-sample files
            observed_fluo = count_fluo(with_sample_spectrum)
            print("Observed fluorescence photons: \n    " + str(observed_fluo))

            # Count the number of fluorescence-range photons seen WITHOUT the Cu target
            # "normalize" for comparison
            background_observed_fluo = count_fluo(without_sample_spectrum)
            normalized_background_observed_fluo = float(background_observed_fluo)/float(len(without_sample_spectrum))
            print("Normalized background observed fluo: ")
            print3(normalized_background_observed_fluo)

            # Divide by total number of photons to "normalize" for comparison
            normalized_observed_fluo = float(observed_fluo)/float(len(with_sample_spectrum))
            print("normalized observed fluo: ")
            print3(normalized_observed_fluo)
            final_normalized_fluo = normalized_observed_fluo - normalized_background_observed_fluo
            print("Subtracted normalized observed fluo (final): ")
            print3(final_normalized_fluo)
            percent_error = 100*float(final_normalized_fluo - norm_expected_fluo)/float(norm_expected_fluo)
            print("percent error: ")
            print3(percent_error)

            if(percent_error < 0):
                print ("Seeing fewer fluorescence photons than expected")
            elif percent_error > 0:
                print ("Seeing more fluorescence photons than expected")

            plot_spectrum(with_sample_spectrum)

        except ZeroDivisionError:
            print("The folder \"" + str(with_sample_foldername) + "\" appears to be empty")

    elif len(sys.argv) == 3 and sys.argv[1].lower() == "calc":
        run_energy = int(sys.argv[2])
        # run_thickness = int(sys.argv[3])
        print(str(run_energy) + " eV")
        prefactor = (fluorate_from_bin_flex(run_energy, "Cu"))
        factor1 = prefactor * .00875  # .11/4pi
        factor4 = prefactor * .02976  # .374/4pi

        print factor1
        print factor4

    elif len(sys.argv) == 2 and sys.argv[1].lower() == "test_equation":

        for i in (.1, 1, 16):  # thicknesses in microns
            t = i*10**(-4)  # microns to cm
            metals_dict["Cu"]["thickness"] = t
            mat_x = np.zeros((20, 1))
            mat_y = np.zeros((20, 1))
            counter = 0
            for j in range(0, 20000, 1000):  # over steps of energies
                mat_x[counter] = j
                r = fluorate_from_bin_flex(j, "Cu", t)
                mat_y[counter] = r
                counter += 1
            print mat_y
            plot(mat_x, mat_y, label=str("thickness " + str(t)))
        legend()
        xlabel("energy (eV)")
        ylabel("fluo factor")
        show()

        for energy in (10000, 20000, 30000, 40000, 50000):

            mat_x1 = np.zeros((8000, 1))
            mat_y1 = np.zeros((8000, 1))
            counter1 = 0
            for i in range(0, 80000, 10):  # thickness, in nm
                t = i*10**(-7)  # convert nm to cm thickness
                metals_dict["Cu"]["thickness"] = t
                mat_x1[counter1] = i
                r = fluorate_from_bin_flex(energy, "Cu")  # at some set energy
                mat_y1[counter1] = r
                counter1 += 1
            plot(mat_x1, mat_y1, label=str(energy) + " eV")
        legend()
        xlabel("thickness (nm)")
        ylabel("fluo factor")
        # show()

        pass

    elif len(sys.argv) == 6:
        #
        a = beamtime_expected_fluo(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        #beamtime_expected_fluo(thickness_nm, this_energy, incident_flux, runtime)
