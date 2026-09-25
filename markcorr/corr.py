from multiprocessing import Pool, cpu_count
import numpy as np
import os
import logging
from . import utils, jackknife_generator, gen_angular
from astropy.table import Table

logging.basicConfig(level=logging.INFO)

def _process_jackknife(args):

    jki, cfTypeArg, realTab1Arg, realTab2Arg, randTabArg, sepMinArg, sepNbinsArg, sepBinWidthArg, sep2NbinsArg, sep2BinWidthArg, \
        doRankingArg, realRaCol1Arg, realDecCol1Arg, realZCol1Arg, randRaColArg, randDecColArg, randZColArg, \
             realRaCol2Arg, realDecCol2Arg, realZCol2Arg, jackknifeSamples1Arg, jackknifeSamples2Arg, workingDir, cosmology_H0_Om0Arg, \
                 weightCol1Arg, weightCol2Arg, realPropertiesArg, doParallelGundam, doBoot = args

    resulti = None

    nReal1 = len(realTab1Arg)
    nReal2 = len(realTab2Arg) if realTab2Arg is not None else 0
    nRand = len(randTabArg)

    try:
        if jki == 0:
            realTab1i, realTab2i, randTabi = realTab1Arg, realTab2Arg, randTabArg
            resultFile = os.path.join(workingDir, 'results', 'CFReal.txt')
            print("Working on the real sample: Nreal_1 = %d, Nreal_2 = %d, Nrand = %d" %(nReal1, nReal2, nRand))
        else:
            realTab1i, randTabi = jackknifeSamples1Arg[jki - 1]
            if jackknifeSamples2Arg is not None:
                realTab2i, _ = jackknifeSamples2Arg[jki - 1]
            else:
                realTab2i = None
            resultFile = os.path.join(workingDir, 'results', 'jackknifes', 'CFJackknife_jk%d.txt' %jki)
            print("Working on the jackknife sample %d: Nreal_1 = %d, Nreal_2 = %d, Nrand = %d" %(jki, nReal1, nReal2, nRand))


        if cfTypeArg == 'angular':
            resulti = gen_angular.do_compute(realTab1i, realTab2i, randTabi, sepMinArg, sepNbinsArg, sepBinWidthArg, doRankingArg, 
                                             realRaCol1Arg, realDecCol1Arg, realRaCol2Arg, realDecCol2Arg, randRaColArg, randDecColArg,
                                             doBoot=doBoot, weight_col1 = weightCol1Arg, weight_col2=weightCol2Arg, realProperties=realPropertiesArg, 
                                             doParallelGundam=doParallelGundam)
        elif cfTypeArg == '3d_redshift':
            raise ValueError("3d_reshift crosscf is not implemented yet!")
        elif cfTypeArg == '3d_projected':
            raise ValueError("3d_projected crosscf is not implemented yet!")

        np.savetxt(resultFile, resulti, delimiter="\t",fmt='%f')

    except Exception as e:
        logging.exception("Error processing jk_i = %d: %s", jki, e)
        return 1

    return 0

def compute_cf(cfType, realTab1=None, realTab2=None, randTab=None, sepMin=0.1, sepMax=10.0, sepNbins=None, 
               sepBinWidth=None, sep2Min=0.0, sep2Max=40.0, sep2Nbins=None, sep2BinWidth=None, 
               nJacksRa=0, nJacksDec=0, workingDir=os.getcwd(), 
               realRaCol1='RA',realDecCol1='DEC', realZCol1=None,  
               realRaCol2='RA',realDecCol2='DEC', realZCol2=None,
               randRaCol='RA', randDecCol='Dec', randZCol=None, 
               doParallel=False, cosmology_H0_Om0=None, doMCF=False, realProperties=None, doRanking=True, weightCol1=None,
               weightCol2=None, doParallelGundam=False, doBoot=False):

    cfAutoCrossLabel = 'cross'

    if cosmology_H0_Om0 is None:
        cosmology_H0_Om0=[70.0, 0.3]

    # validating parameters consistency with type of CF

    validCfTypes = ['angular', '3d_redshift', '3d_projected']
    if cfType not in validCfTypes:
        raise ValueError("Invalid cfType '%s'. Must be one of: %s." %(cfType, ', '.join(validCfTypes)))

    if '3d' in cfType and (realZCol1 is None or randZCol is None or realZCol2 is None):
        raise ValueError("Redshift columns should be given for cfType '%s'" %cfType)

    if cfType == '3d_projected':
        if any(param is not None for param in [sep2Min, sep2Max, sep2Nbins, sep2BinWidth]):
            print(f"Warning: sep2 parameters are ignored for cfType %s" %cfType)

    # reading datatables if not given

    if realTab1 is None:
        GalFile = os.path.join(workingDir, 'real_galaxies1')
        if os.path.exists(GalFile):
            realTab1 = Table.read(GalFile, format='ascii')
        else:
            realTab1 = None

    if realTab2 is None:
        GalFile = os.path.join(workingDir, 'real_galaxies2')
        if os.path.exists(GalFile):
            realTab2 = Table.read(GalFile, format='ascii')
        else:
            realTab2 = None

    if randTab is None:
        GalFile = os.path.join(workingDir, 'random_galaxies')
        if os.path.exists(GalFile):
            randTab = Table.read(GalFile, format='ascii')
        else:
            randTab = None

    if realTab1 is None or randTab is None:
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
            sep2Nbins = int(40)  # default to 10 linear bins
            sep2BinWidth = (sep2Max - sep2Min) / sep2Nbins
        elif sep2Nbins is None:
            sep2Nbins = int((sep2Max - sep2Min) / sep2BinWidth)
        elif sep2BinWidth is None:
            sep2BinWidth = (sep2Max - sep2Min) / sep2Nbins

    # Total number of jackknife regions
    nJacks = nJacksRa * nJacksDec

    # Printing parameters --------------------------------------------------------------------------------------------------

    sepLabelDict = {'angular': 'theta [deg]', '3d_redshift': 's [Mpc/h]', '3d_projected': 'r_p [Mpc/h]'}
    cfTypeLabelDict = {'angular': 'Angular', '3d_redshift': '3D reshift-space', '3d_projected': '3D projected'}

    print("\n")
    print("---------- COMPUTING %s CROSS-CORRELATION FUNCTION ------------------" %cfTypeLabelDict[cfType].upper())
    print("\n")

    print("Working directory: ", workingDir)
    print("Minimum %s = %0.2f \nMaximum %s = %0.2f\n Nr. of bins in %s = %d\n Binwidth (log) in %s = %0.2f" %(sepLabelDict[cfType], sepMin, sepLabelDict[cfType], sepMax, sepLabelDict[cfType], sepNbins, sepLabelDict[cfType], sepBinWidth))
    if cfType == '3d_projected':
        print("Minimum pi = %0.2f \nMaximum pi = %0.2f\n Nr. of bins in pi = %d\n Binwidth (log) in pi = %0.2f" %(sep2Min, sep2Max, sep2Nbins, sep2BinWidth))
    print("Number of jackknife regions: %d (%d RA x %d Dec)" %(nJacks, nJacksRa, nJacksDec))

    # ------------------------------------------------------------------------------------------------------------------------

    original_working_dir = os.getcwd()

    os.chdir(workingDir)
    os.makedirs(workingDir+os.path.sep+'biproducts',  exist_ok=True)
    os.makedirs(workingDir+os.path.sep+'results/jackknifes',  exist_ok=True)

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

    jackknifeSamples1 = jackknife_generator.make_JK_samples(realTab1, randTab, nJacksRa, nJacksDec, realRaCol1, realDecCol1, randRaCol, randDecCol, plot=False)
    if realTab2 is not None:
        jackknifeSamples2 = jackknife_generator.make_JK_samples(realTab2, randTab, nJacksRa, nJacksDec, realRaCol2, realDecCol2, randRaCol, randDecCol, plot=False)
    else:
        jackknifeSamples2 = None

    processOutcomes = []
    tasks = []
    
    numProcesses = cpu_count()
    print(f"Parallelizing with %d processes..." %numProcesses)

    for jki in range(nJacks + 1):
        argsToPass = (jki, cfType, realTab1, realTab2, randTab, sepMin, sepNbins, sepBinWidth, sep2Nbins, sep2BinWidth, doRanking, 
                        realRaCol1, realDecCol1, realZCol1, randRaCol, randDecCol, randZCol, realRaCol2, realDecCol2, realZCol2, 
                        jackknifeSamples1, jackknifeSamples2, workingDir, cosmology_H0_Om0, 
                        weightCol1, weightCol2, realProperties, 
                        doParallelGundam, doBoot)

        if doParallel:
            tasks.append(argsToPass)
        else:
            outcome = _process_jackknife(argsToPass)
            processOutcomes.append(outcome)

    if doParallel:
        numProcesses = cpu_count()
        print("Parallelizing with %d processes..." % numProcesses)

        with Pool(processes=numProcesses) as pool:
            processOutcomes = pool.map(_process_jackknife, tasks)
    #     for jki in range(nJacks + 1):
            
            

    #     with Pool(processes=numProcesses) as pool:
    #         processOutcomes = pool.map(_process_jackknife, tasks)
    # else:
    #     for jki in range(nJacks+1):
    #         argsToPass = (jki, cfType, realTab1, realTab2, randTab, sepMin, sepNbins, sepBinWidth, sep2Nbins, sep2BinWidth, doRanking, 
    #                       realRaCol1, realDecCol1, realZCol1, randRaCol, randDecCol, randZCol, realRaCol2, realDecCol2, realZCol2, 
    #                       jackknifeSamples1, jackknifeSamples2, workingDir, cosmology_H0_Om0,
    #                       weightCol1Arg, weightCol2Arg, realPropertiesArg, doParallelGundam, doBoot)
    #         outcome = _process_jackknife(argsToPass)
    #         processOutcomes.append(outcome)

    os.chdir(original_working_dir)

    if any(outcome != 0 for outcome in processOutcomes):
        print("Warning: Some jackknife computations failed.")
        return 1

    print("All computations completed successfully.")
    return 0

