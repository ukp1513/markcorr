import os
import datetime

def write_process_summary(filepath, cfType, cfTypeLabel, cfAutoCrossLabel, sepLabel, params):
    with open(filepath, 'w') as f:
        f.write("---------- COMPUTATION SUMMARY ----------\n\n")
        f.write("Timestamp: %s\n\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        f.write("Correlation function type: %s %s\n" % (cfAutoCrossLabel, cfTypeLabel))
        f.write("Working directory: %s\n\n" % params['workingDir'])

        f.write("Separation binning (%s):\n" % sepLabel)
        f.write(" - sepMin: %.2f\n" % params['sepMin'])
        f.write(" - sepMax: %.2f\n" % params['sepMax'])
        f.write(" - sepNbins: %d\n" % params['sepNbins'])
        f.write(" - sepBinWidth (log): %.4f\n\n" % params['sepBinWidth'])

        if cfType == '3d_projected':
            f.write("Pi binning (line-of-sight):\n")
            f.write(" - sep2Min: %.2f\n" % params['sep2Min'])
            f.write(" - sep2Max: %.2f\n" % params['sep2Max'])
            f.write(" - sep2Nbins: %d\n" % params['sep2Nbins'])
            f.write(" - sep2BinWidth (linear): %.4f\n\n" % params['sep2BinWidth'])

        f.write("Jackknife configuration:\n")
        f.write(" - RA regions: %d\n" % params['nJacksRa'])
        f.write(" - Dec regions: %d\n" % params['nJacksDec'])
        f.write(" - Total regions: %d\n\n" % params['nJacks'])

        f.write("Cosmology (H0, Om0): %s\n" % str(params['cosmology']))
        f.write("Do MCF? %s\n" % str(params['doMCF']))
        f.write("Do ranking? %s\n" % str(params['doRanking']))
        f.write("Do parallel? %s\n" % str(params['doParallel']))

