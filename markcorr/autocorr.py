import numpy as np
import os
import logging
from . import jackknife_generator, auto_angular, auto_threeD, auto_projected, utils
from multiprocessing import Pool, cpu_count
from astropy.table import Table
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def _process_jackknife(args):

    jki, cfTypeArg, realTabArg, realPropertiesArg, randTabArg, sepMinArg, sepNbinsArg, sepBinWidthArg, sep2NbinsArg, sep2BinWidthArg, doRankingArg, realRaColArg, realDecColArg, realZColArg, randRaColArg, randDecColArg, randZColArg, jackknifeSamplesArg, workingDir, cosmology_H0_Om0Arg = args

    try:
        if jki == 0:
            realTabi, randTabi = realTabArg, randTabArg
            resultFile = os.path.join(workingDir, 'results', 'CFReal.txt')
            print("Working on the real sample: Nreal = %d, Nrand = %d" %(len(realTabi), len(randTabi)))
        else:
            realTabi, randTabi = jackknifeSamplesArg[jki - 1]
            resultFile = os.path.join(workingDir, 'results', 'jackknifes', 'CFJackknife_jk%d.txt' %jki)
            print("Working on the jackknife sample %d: Nreal = %d, Nrand = %d" %(jki, len(realTabi), len(randTabi)))

        resulti = None

        if cfTypeArg == 'angular':
            resulti = auto_angular.do_compute(realTabi, realPropertiesArg, randTabi, sepMinArg, sepNbinsArg, sepBinWidthArg, doRankingArg, realRaColArg, realDecColArg, randRaColArg, randDecColArg)
        elif cfTypeArg == '3d_redshift':
            resulti = auto_threeD.do_compute(realTabi, realPropertiesArg, randTabi, sepMinArg, sepNbinsArg, sepBinWidthArg, doRankingArg, realRaColArg, realDecColArg, realZColArg, randRaColArg,
                                            randDecColArg, randZColArg, cosmology_H0_Om0Arg)
        elif cfTypeArg == '3d_projected':
            resulti = auto_projected.do_compute(realTabi, realPropertiesArg, randTabi, sepMinArg, sepNbinsArg, sepBinWidthArg, sep2NbinsArg, sep2BinWidthArg, doRankingArg,
                                              realRaColArg, realDecColArg, realZColArg, randRaColArg, randDecColArg, randZColArg, cosmology_H0_Om0Arg)

        np.savetxt(resultFile, resulti, delimiter="\t",fmt='%f')

    except Exception as e:
        logging.exception("Error processing jk_i = %d: %s", jki, e)
        return 1

    return 0

def compute_cf(cfType, realTab=None, randTab=None, sepMin=0.1, sepMax=10.0, sepNbins=None, sepBinWidth=None, sep2Min=None, sep2Max=None, sep2Nbins=None, sep2BinWidth=None, nJacksRa=0, nJacksDec=0, workingDir=os.getcwd(), realRaCol='RA',realDecCol='DEC', realZCol=None, randRaCol='RA', randDecCol='Dec', randZCol=None, doParallel=False, cosmology_H0_Om0=None, doMCF=False, realProperties=None, doRanking=True, makePlots=True):

    cfAutoCrossLabel = 'auto'

    if cosmology_H0_Om0 is None:
        cosmology_H0_Om0=[70.0, 0.3]

    # validating parameters consistency with type of CF

    validCfTypes = ['angular', '3d_redshift', '3d_projected']
    if cfType not in validCfTypes:
        raise ValueError("Invalid cfType '%s'. Must be one of: %s." %(cfType, ', '.join(validCfTypes)))

    if '3d' in cfType and (realZCol is None or randZCol is None):
        raise ValueError("Redshift column should be given for cfType '%s'" %cfType)

    if cfType != '3d_projected':
        if any(param is not None for param in [sep2Min, sep2Max, sep2Nbins, sep2BinWidth]):
            print("Warning: sep2 parameters are ignored for cfType %s" %cfType)

    if not doMCF:
        realProperties = []
    if doMCF and realProperties is None:
        raise ValueError("Real properties should be given for marked %s correlation function" %cfType)

    # reading datatables if not given

    if realTab is None:
        GalFile = os.path.join(workingDir, 'real_galaxies')
        if os.path.exists(GalFile):
            realTab = Table.read(GalFile, format='ascii')
        else:
            realTab = None

    if randTab is None:
        GalFile = os.path.join(workingDir, 'random_galaxies')
        if os.path.exists(GalFile):
            randTab = Table.read(GalFile, format='ascii')
        else:
            randTab = None

    if realTab is None or randTab is None:
        raise ValueError("Real and random catalogues are to be given")

    # setting bins in th, s, or rp (in log scale)

    if sepMin <= 0 or sepMax <= sepMin:
        raise ValueError("sepMin and sepMax must be > 0 and sepMax > sepMin for logarithmic binning.")

    if sepNbins is not None and sepBinWidth is not None: # fixing dsep even if it is not consistent with given min, max, nbins
        sepBinWidth = (np.log10(sepMax) - np.log10(sepMin)) / sepNbins
    elif sepNbins is None and sepBinWidth is None:
        sepNbins = 10 # default 10 bins in rp
        sepBinWidth = (np.log10(sepMax) - np.log10(sepMin)) / sepNbins
    elif sepNbins is None:
        sepNbins = int((np.log10(sepMax) - np.log10(sepMin)) / sepBinWidth)
    elif sepBinWidth is None:
        sepBinWidth = (np.log10(sepMax) - np.log10(sepMin)) / sepNbins

    # setting bins in pi using (in linear scale)

    if cfType == '3d_projected':

        sep2Min = 0.0 if sep2Min is None else sep2Min

        sep2Max = 40.0 if sep2Max is None else sep2Max

        if sep2Min < 0 or sep2Max < sep2Min:
            raise ValueError("sep2Min and sep2Max must be > 0 and sep2Max > sep2Min.")

        if sep2Nbins is not None and sep2BinWidth is not None:
            sep2BinWidth = (sep2Max - sep2Min) / sep2Nbins
        elif sep2Nbins is None and sep2BinWidth is None:
            sep2Nbins = int(10)  # default to 10 linear bins
            sep2BinWidth = (sep2Max - sep2Min) / sep2Nbins
        elif sep2Nbins is None:
            sep2Nbins = int((sep2Max - sep2Min) / sep2BinWidth)
        elif sep2BinWidth is None:
            sep2BinWidth = (sep2Max - sep2Min) / sep2Nbins

    else:
        sep2Min, sep2Max, sep2Nbins, sep2BinWidth = None, None, None, None

    # Total number of jackknife regions
    nJacks = nJacksRa * nJacksDec

    # Printing parameters --------------------------------------------------------------------------------------------------

    sepLabelDict = {'angular': 'theta [deg]', '3d_redshift': 's [Mpc/h]', '3d_projected': 'r_p [Mpc/h]'}
    cfTypeLabelDict = {'angular': 'Angular', '3d_redshift': '3D reshift-space', '3d_projected': '3D projected'}

    print("\n")
    print("---------- COMPUTING %s TWO-POINT AUTO-CORRELATION FUNCTION ------------------" %cfTypeLabelDict[cfType].upper())
    print("\n")

    print("Working directory: ", workingDir)
    print("Minimum %s = %0.2f \nMaximum %s = %0.2f\n Nr. of bins in %s = %d\n Binwidth (log) in %s = %0.2f" %(sepLabelDict[cfType], sepMin, sepLabelDict[cfType], sepMax, sepLabelDict[cfType], sepNbins, sepLabelDict[cfType], sepBinWidth))
    if cfType == '3d_projected':
        print("Minimum pi = %0.2f \nMaximum pi = %0.2f\n Nr. of bins in pi = %d\n Binwidth (log) in pi = %0.2f" %(sep2Min, sep2Max, sep2Nbins, sep2BinWidth))
    print("Number of jackknife regions: %d (%d RA x %d Dec)" %(nJacks, nJacksRa, nJacksDec))

    # ------------------------------------------------------------------------------------------------------------------------

    biproductDirName = os.path.join(workingDir, "biproducts")
    os.makedirs(biproductDirName,  exist_ok=True)

    resultsJKDirName = os.path.join(workingDir, "results", "jackknifes")
    os.makedirs(resultsJKDirName, exist_ok=True)

    original_working_dir = os.getcwd()
    os.chdir(workingDir)

    if doMCF:
        with open(os.path.join(biproductDirName, 'real_properties.txt'), 'w') as f:
            for realProp in realProperties:
                f.write('%s\n' % realProp)

    summary_path = os.path.join(workingDir, 'biproducts', 'process_summary.txt')
    utils.write_process_summary(
        summary_path,
        cfType,
        cfAutoCrossLabel,
        cfTypeLabelDict[cfType],
        sepLabelDict[cfType],
        {
            'workingDir': workingDir,
            'sepMin': sepMin,
            'sepMax': sepMax,
            'sepNbins': sepNbins,
            'sepBinWidth': sepBinWidth,
            'sep2Min': sep2Min,
            'sep2Max': sep2Max,
            'sep2Nbins': sep2Nbins,
            'sep2BinWidth': sep2BinWidth,
            'nJacksRa': nJacksRa,
            'nJacksDec': nJacksDec,
            'nJacks': nJacks,
            'cosmology': cosmology_H0_Om0,
            'doMCF': doMCF,
            'doRanking': doRanking,
            'doParallel': doParallel
        }
    )

    jackknifeSamples = jackknife_generator.make_JK_samples(realTab, randTab, nJacksRa, nJacksDec, realRaCol, realDecCol, randRaCol, randDecCol, plot=False)

    if makePlots:
        fig, ax = plt.subplots()
        fig.set_size_inches(7, 7)

        # Scatter plot for real and random
        ax.scatter(randTab[randRaCol], randTab[randDecCol], s=1, color='blue', label='Random')
        ax.scatter(realTab[realRaCol], realTab[realDecCol], s=1, color='red', label='Real')

        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')

        # If jackknife boundaries should be drawn
        if nJacksRa > 0 and nJacksDec > 0:
            ra_min, ra_max = min(randTab[randRaCol]), max(randTab[randRaCol])
            dec_min, dec_max = min(randTab[randDecCol]), max(randTab[randDecCol])

            # Vertical boundaries (RA)
            ra_step = (ra_max - ra_min) / nJacksRa
            for i in range(1, nJacksRa):
                ax.axvline(x=ra_min + i * ra_step, color='black', linestyle='--', linewidth=0.8)

            # Horizontal boundaries (Dec)
            dec_step = (dec_max - dec_min) / nJacksDec
            for j in range(1, nJacksDec):
                ax.axhline(y=dec_min + j * dec_step, color='black', linestyle='--', linewidth=0.8)

        ax.legend(markerscale=5, loc=0)  # Scale legend points

        plt.savefig(os.path.join(workingDir, "ra_dec.png"), dpi=300, bbox_inches='tight')
        plt.close()

        if '3d' in cfType:
            fig, ax = plt.subplots()
            fig.set_size_inches(7, 7)

            # Scatter plot for real and random
            ax.hist(randTab[randZCol], color='blue', alpha=0.5, density=True, label='Random')
            ax.hist(realTab[realZCol], color='red', alpha=0.5, density=True, label='Real')

            ax.set_xlabel('Redshift')
            ax.set_ylabel('Normalised count')

            ax.legend()

            plt.savefig(os.path.join(workingDir, "z_hist.png"), dpi=300, bbox_inches='tight')
            plt.close()


    processOutcomes = []

    if doParallel:
        numProcesses = cpu_count()
        print("Parallelizing with %d processes..." %numProcesses)

        tasks = []
        for jki in range(nJacks + 1):
            argsToPass = (jki, cfType, realTab, realProperties, randTab, sepMin, sepNbins, sepBinWidth, sep2Nbins, sep2BinWidth, doRanking, realRaCol, realDecCol, realZCol, randRaCol, randDecCol, randZCol, jackknifeSamples, workingDir, cosmology_H0_Om0)
            tasks.append(argsToPass)

        with Pool(processes=numProcesses) as pool:
            processOutcomes = pool.map(_process_jackknife, tasks)
    else:
        for jki in range(nJacks+1):
            argsToPass = (jki, cfType, realTab, realProperties, randTab, sepMin, sepNbins, sepBinWidth, sep2Nbins, sep2BinWidth, doRanking, realRaCol, realDecCol, realZCol, randRaCol, randDecCol, randZCol, jackknifeSamples, workingDir, cosmology_H0_Om0)
            outcome = _process_jackknife(argsToPass)
            processOutcomes.append(outcome)

    os.chdir(original_working_dir)

    if any(outcome != 0 for outcome in processOutcomes):
        print("Warning: Some jackknife computations failed.")
        return 1

    print("All computations completed successfully.")
    return 0
