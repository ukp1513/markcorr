import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import os
import shutil
from scipy.optimize import curve_fit
import bokeh.palettes as bp
from . import integral_constrain
from matplotlib.ticker import FormatStrFormatter
from astropy.table import Table

def _covmat_to_corrmat(covariance):
    std_devs = np.sqrt(np.diag(covariance))
    outer_std = np.outer(std_devs, std_devs)
    correlation = covariance / outer_std
    correlation[covariance == 0] = 0
    return correlation, std_devs

def _corrmat_to_covmat(correlation, std_devs):
    outer_std = np.outer(std_devs, std_devs)
    covariance = correlation * outer_std
    return covariance

def _invcorrmat_to_invcovmat(inv_correlation, std_devs):
    D_inv = np.diag(1.0 / std_devs)
    inv_covariance = D_inv @ inv_correlation @ D_inv
    return inv_covariance


def _angularCF_model(theta, A, gam):
    return A*pow(theta, 1-gam)

def _redshift3dCF_model(s, s0, gam):
    return pow((s/s0), (-1*gam))

def _projected3dCF_model(rp, r0, gam):
    return rp * pow((r0/rp), gam) * gamma(0.5) * gamma(0.5*(gam-1)) / gamma(0.5*gam)


def do_analyze(cfType, sepMin, sepMax, sepMinToFit, sepMaxToFit, doFit2pcf=True, useFullCovar=True, doSvdFilter=False, doHartlapCorr=False, doMCF=True, realProperties=None, dirName=os.getcwd(), plotXScale='log', plotYScale='log', ignoreNegatives = True, computeIC = False):

    if doMCF and realProperties is None:
        raise ValueError("realProperties list has to be given if doMCF is True")

    biproductDirName = os.path.join(dirName, "biproducts")
    resultsDirName = os.path.join(dirName, "results")

    validCfTypes = ['angular', '3d_redshift', '3d_projected']

    if cfType == 'angular':

        cfXLabel = r"$\theta \,\,\, (deg)$"
        cfYLabel = r"$\omega(\theta)$"
        mcfXLabel = r"$\theta \,\,\, (deg)$"
        mcfYLabel = r"$M (\theta)$"
        cfFigName = dirName+os.path.sep+"fig_angularCF.png"
        mcfFigName = dirName+os.path.sep+"fig_angularMCF.png"

    elif cfType == '3d_redshift':

        cfXLabel = r"$s \,\,\, (Mpc/h)$"
        cfYLabel = r"$\xi(s)$"
        mcfXLabel = r"$s \,\,\, (Mpc/h)$"
        mcfYLabel = r"$M (s)$"
        cfFigName = dirName+os.path.sep+"fig_3DRedshiftCF.png"
        mcfFigName = dirName+os.path.sep+"fig_3DRedshiftMCF.png"

    elif cfType == '3d_projected':

        cfXLabel = r"$r_p \,\,\, (Mpc/h)$"
        cfYLabel = r"$\omega_p(r_p) \,\,\, (Mpc/h)$"
        mcfXLabel = r"$r_p \,\,\, (Mpc/h)$"
        mcfYLabel = r"$M_p (r_p)$"
        cfFigName = dirName+os.path.sep+"fig_3DProjectedCF.png"
        mcfFigName = dirName+os.path.sep+"fig_3DProjectedMCF.png"

    else:
        raise ValueError("Invalid cfType '%s'. Must be one of: %s." %(cfType, ', '.join(validCfTypes)))

    if not doSvdFilter:
        print("\nSVD correction NOT done!")
    else:
        print("\nSVD correction done!")

    if not doHartlapCorr:
        print("\nHartlap correction NOT done!")
    else:
        print("\nHartlap correction done!")


    # REMOVING PREVIOUS INV COVAR FILES IF EXIST
    invCovMatFileName = os.path.join(biproductDirName ,"inv_cov_mat.txt")
    invCovMatSVDFileName = os.path.join(biproductDirName ,"inv_corr_mat_SVD.txt")

    if os.path.exists(invCovMatFileName):
        os.remove(invCovMatFileName)
    if os.path.exists(invCovMatSVDFileName):
        os.remove(invCovMatSVDFileName)

    cwdFullPath = os.getcwd()
    cwdSplit=cwdFullPath.split('/')
    sampleName=cwdSplit[-1].upper()

    print("\n--------------------------------------------")
    print("\nFitting sample %s....\n" %(sampleName))
    print("--------------------------------------------\n")

    # CREATING CFRealAll.txt for JK copies

    CFRealFilePath = os.path.join(resultsDirName,'CFReal.txt')
    CFJKDirPath = os.path.join(resultsDirName,'jackknifes')
    if not os.path.exists(CFRealFilePath):
        raise FileNotFoundError("CF file of real galaxy not found")

    CFRealResult = np.loadtxt(CFRealFilePath)
    sep = CFRealResult[:,0]
    CFReal = CFRealResult[:,1]

    totalNBins = len(sep)
    nCopies=len([f for f in os.listdir(CFJKDirPath) if not f.startswith('.')])

    print('Nr. of total bins: ', totalNBins)
    print('Nr. of jackknife copies: ', nCopies)

    CFRealAll = np.empty((totalNBins, nCopies + 2), dtype=float)

    CFRealAll[:, 0] = sep       # First column ← sep
    CFRealAll[:, 1] = CFReal    # Second column ← CFReal

    for copy in range(nCopies):
        CFJK = np.loadtxt(os.path.join(CFJKDirPath, 'CFJackknife_jk%d.txt' % (copy + 1)))[:, 1]
        CFRealAll[:, copy + 2] = CFJK


    np.savetxt(os.path.join(resultsDirName, "CFRealAll.txt"), CFRealAll,delimiter="\t",fmt='%f')

    if doMCF:
        # COLLECTING MCFs

        marks = realProperties
        nMarks = len(marks)
        print("Marks : ", marks)

        mcfAllMarks = {} # dictionary to carry all the MCFs (including jackknifes) for all marks

        nCopiesTmp = nCopies
        for marki, mark in enumerate(marks):
            mcfReal = CFRealResult[:,marki+2]
            mcfRealAllMarki = np.empty((totalNBins, nCopies+2), dtype=float)
            mcfRealAllMarki[:, 0] = sep
            mcfRealAllMarki[:, 1] = mcfReal
            for copy in range(nCopiesTmp):
                try:
                    mcfJK = np.loadtxt(os.path.join(CFJKDirPath, 'CFJackknife_jk%d.txt' %(copy+1)))[:,marki+2]
                    mcfRealAllMarki[:, copy+2] = mcfJK
                except FileNotFoundError:
                    print("Jackknife copy %d not found." %(copy+1))
                    nCopies = nCopies-1

            # saving each MCF values
            np.savetxt(os.path.join(resultsDirName, "mcfRealAll_%s.txt" %(mark)), mcfRealAllMarki, delimiter="\t",fmt='%f')

            mcfAllMarks[mark] = mcfRealAllMarki

    # FILTERING NAN AND INF VALUES

    filterIndexCF = []

    for i in range(totalNBins):
        sep_val = CFRealAll[i, 0]
        cf_val = CFRealAll[i, 1]

        # Filter by sepMin/sepMax
        if sep_val < sepMin or sep_val > sepMax:
            filterIndexCF.append(i)
            continue  # no need to check further if already excluded

        # Always check NaN/Inf in CF column
        if np.isnan(cf_val) or np.isinf(cf_val):
            filterIndexCF.append(i)
            continue

        # If ignoreNegatives is True, also check for negative values
        if ignoreNegatives and cf_val < 0.:
            filterIndexCF.append(i)
            continue

        # Always check NaN/Inf in MCF columns

        if doMCF:
            for j in range(nMarks):
                mcf_val = CFRealAll[i, j]
                if np.isnan(mcf_val) or np.isinf(mcf_val):
                    filterIndexCF.append(i)
                    break  # no need to check further marks for this row


    filterIndexCF=list(set(filterIndexCF))

    #REMOVING NAN BINS FROM CF FILE
    CFRealAllFiltered=np.delete(CFRealAll, filterIndexCF, axis=0)

    nBinsCFFiltered=totalNBins-len(filterIndexCF)

    print("Number of CF bins with non-nan values:", nBinsCFFiltered)
    np.savetxt(os.path.join(resultsDirName, 'CFRealAll_filtered.txt'), CFRealAllFiltered, delimiter='\t', fmt='%f')

    if nBinsCFFiltered == 0:
        raise ValueError("There are no bins with reliable CF within the fitting range")

    #REMOVING NAN BINS FROM mcf FILE

    if doMCF:
        for mark, mcfRealAll in mcfAllMarks.items():
            mcfRealAllFiltered = np.delete(mcfRealAll, filterIndexCF, axis=0)
            np.savetxt(os.path.join(resultsDirName, 'mcfRealAll_%s_filtered.txt' %mark), mcfRealAllFiltered, delimiter='\t', fmt='%f')

    #FILTERING TO FIT BINS

    filterIndexCFToFit = []
    for i in range(0, nBinsCFFiltered):
        if(CFRealAllFiltered[i,0] < sepMinToFit or CFRealAllFiltered[i,0] > sepMaxToFit):
            filterIndexCFToFit.append(i)
    filterIndexCFToFit = list(set(filterIndexCFToFit))
    nBinsCFToFit = nBinsCFFiltered - len(filterIndexCFToFit)

    print("Number of bins used for CF fitting ", nBinsCFToFit)

    CFRealAllToFit=np.delete(CFRealAllFiltered, filterIndexCFToFit, axis=0)

    np.savetxt(os.path.join(resultsDirName, 'CFRealAll_filtered_tofit.txt'), CFRealAllToFit, delimiter='\t', fmt='%f')

    if nBinsCFToFit == 0:
        raise ValueError("There are no bins with reliable CF within the fitting range")

    # COMPUTING COVARIANCE MATRIX FOR ALL FILTERED BINS AND PLOTTING ALL BINS

    sepToPlot = CFRealAllFiltered[:,0]
    CFToPlot = CFRealAllFiltered[:,1]

    if nCopies > 0:

        allJKCFs = CFRealAllFiltered[:, 2:nCopies+2]

        covMat = np.cov(allJKCFs, bias=True)	# C = (Njk-1)/Njk x SUM in case of jackknife
        covMat = (nCopies - 1) * covMat

        CFErrToPlot = np.sqrt(np.diag(covMat))

    else:
        CFErrToPlot = [0 for i in range(len(sepToPlot))]

    fig,ax_now=plt.subplots(nrows=1,ncols=1,sharex=False)
    fig.set_size_inches(5,5)
    plt.errorbar(sepToPlot, CFToPlot, CFErrToPlot, ls='none', capsize=5,ms=10,marker='o',mew=1.0,mec='black',mfc='white',ecolor='black',elinewidth=1)

    finalDirPath = os.path.join(dirName, 'finals')

    if not os.path.exists(finalDirPath):
        os.makedirs(finalDirPath)
    else:
        shutil.rmtree(finalDirPath)           # Removes all the subdirectories!
        os.makedirs(finalDirPath)

    np.savetxt(finalDirPath+os.path.sep+'final_CF_toPlot.txt', np.transpose([sepToPlot, CFToPlot, CFErrToPlot]), fmt='%f', delimiter='\t')

    # fitting 2pCF

    if doFit2pcf:

        covariancingSuccess = True
        SVDDone = False
        covMatSVD = None

        sepToFit = CFRealAllToFit[:, 0]
        CFToFit = CFRealAllToFit[:, 1]

        nBinsEff=nBinsCFToFit # effective bins used for fitting

        if nCopies == 0:
            print("No JK copies found. Unable to estimate CF errors - setting errorbars to zero.")
            covariancingSuccess = False

        else:

            allJKCFsToFit = CFRealAllToFit[:, 2:nCopies+2]

            covMatToFit = np.cov(allJKCFsToFit, bias=True)	# C = (Njk-1)/Njk x SUM in case of jackknife
            covMatToFit = (nCopies-1)*covMatToFit

            corrMat, stdDevs = _covmat_to_corrmat(covMatToFit)

            U, Dvector, UT = np.linalg.svd(corrMat)	# C = U D UT

            DinvVec = []

            for lambdai in Dvector:
                if doSvdFilter:
                    if lambdai < np.sqrt(2./nCopies):
                        DinvVec.append(0.)
                    else:
                        DinvVec.append(1./lambdai)
                    SVDDone = True
                else:
                    DinvVec.append(1./lambdai)

            Dinv = np.diag(DinvVec)

            invCorrMatSVD = np.matmul(U, np.matmul(Dinv,UT))
            invCovMatSVD = _invcorrmat_to_invcovmat(invCorrMatSVD, stdDevs)

            if doHartlapCorr:
                hartlapFactor = (nCopies - nBinsCFToFit - 2)/(nCopies-1)
                print("Hartlap factor: ",hartlapFactor,"\n")
                invCovMatSVD = hartlapFactor*invCovMatSVD

            try:
                covMatSVD = np.linalg.inv(invCovMatSVD)
            except np.linalg.LinAlgError:
                print("\nIssue taking inverse of inverse covariance matrix! Setting errorbars are zero")
                covariancingSuccess = False

        with open(os.path.join(biproductDirName, "effective_bins.txt"), "w", encoding="utf-8") as fEff:
            fEff.write(str(nBinsEff))


        np.savetxt(biproductDirName+os.path.sep+"cov_mat.txt",np.transpose(covMatSVD),delimiter="\t",fmt='%f')

        if covariancingSuccess:
            CFErrToFit = np.sqrt(np.diag(covMatSVD))
        else:
            CFErrToFit = [0 for i in range(len(sepToFit))]

        np.savetxt(finalDirPath+os.path.sep+'final_CF.txt', np.transpose([sepToFit, CFToFit, CFErrToFit]), fmt='%f', delimiter='\t')

        # FIT USING CURVE_FIT

        if useFullCovar:
            sigmaToFit = covMatSVD
        else:
            sigmaToFit = CFErrToFit

        if cfType == 'angular':

            try:
                popt, pcov, _, _, _ = curve_fit(_angularCF_model, sepToFit, CFToFit, sigma=sigmaToFit)
            except Exception as e:
                print(f"Error fitting: {e}")
                return

            AFit, AErrFit, gammaFit, gammaErrFit = popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1])
            angularFitParams = (AFit, AErrFit, gammaFit, gammaErrFit)
            print('Curve fitting parameters:\nA = %0.2lf +/- %0.2lf\ngamma = %0.2lf +/- %0.2lf\n' %angularFitParams)
            bestFitModelCurve = _angularCF_model(sepToPlot, AFit, gammaFit)
            plt.errorbar(sepToFit, CFToFit, CFErrToFit,ls='none',capsize=5,ms=10,marker='o',mew=1.0,mec='black',mfc='black',ecolor='black',elinewidth=1)
            plotLabel = (r"$\omega(\theta)=A \theta^{1-\gamma}$" + "\n" + r"$A = %0.2f \pm %0.2f$" + "\n" + r"$\gamma = %0.2f \pm %0.2f$")
            if SVDDone:
                plotLabel = "SVD Done" + "\n" + plotLabel
            plt.plot(sepToPlot, bestFitModelCurve, color='red',label=plotLabel %angularFitParams)


            # WRITING TO FILES

            np.savetxt(finalDirPath+os.path.sep+'CF_fit_params_covariance.txt', pcov, fmt='%f')
            np.savetxt(finalDirPath+os.path.sep+'CF_fit_params.txt', [angularFitParams], fmt='%f', delimiter='\n')
            np.savetxt(finalDirPath+os.path.sep+'sepFitRange.txt', [sepMinToFit, sepMaxToFit], fmt='%f', delimiter='\n')


            if computeIC:
                if randTab is None:
                    randTab = Table.read(dirName+os.path.sep+'random_galaxies', format='ascii')
                for col in randTab.colnames:
                    randTab.rename_column(col, col.upper())

                IC = integral_constrain.computeICAngular(randgalaxies=randTab, A=AFit, gamma=gammaFit, randracol='RA', randdeccol='DEC')
                print("Integral Constrain = %f" %IC)
                with open(finalDirPath+os.path.sep+'IC.txt', 'w', encoding="utf-8") as file:
                    file.write(str(IC))


        elif cfType == '3d_redshift':

            try:
                popt, pcov = curve_fit(_redshift3dCF_model, sepToFit, CFToFit, p0=[5.0, 1.8], sigma=sigmaToFit)
            except Exception as e:
                print(f"Error fitting: {e}")
                return

            s0Fit, s0ErrFit, gammaFit, gammaErrFit = popt[0],np.sqrt(pcov[0,0]),popt[1],np.sqrt(pcov[1,1])
            threeDFitParams = (s0Fit, s0ErrFit, gammaFit, gammaErrFit)
            print('Curve fitting parameters:\ns0 = %0.2lf +/- %0.2lf\ngamma = %0.2lf +/- %0.2lf\n' %threeDFitParams)
            bestFitModelCurve = _redshift3dCF_model(sepToPlot, s0Fit, gammaFit)
            plt.errorbar(sepToFit, CFToFit, CFErrToFit, ls='none',capsize=5,ms=10,marker='o',mew=1.0,mec='black',mfc='black',ecolor='black',elinewidth=1)
            plotLabel = r"$\xi(s)=(s/s_0)^{-\gamma}$" + "\n" + r"$s_0 = %0.2f \pm %0.2f$" + "\n" + r"$\gamma = %0.2f \pm %0.2f$"
            if SVDDone:
                plotLabel = "SVD Done" + "\n" + plotLabel
            plt.plot(sepToPlot, bestFitModelCurve, color='red',label=plotLabel %threeDFitParams)

            # WRITING TO FILES

            np.savetxt(finalDirPath+os.path.sep+'CF_fit_params_covariance.txt', pcov, fmt='%f')
            np.savetxt(finalDirPath+os.path.sep+'CF_fit_params.txt', [threeDFitParams], fmt='%f', delimiter='\n')
            np.savetxt(finalDirPath+os.path.sep+'sepFitRange.txt', [sepMinToFit, sepMaxToFit], fmt='%f', delimiter='\n')

            if computeIC:
                print("IC Computation is not coded for 3d...") #TODO

        elif cfType == '3d_projected':

            try:
                popt, pcov = curve_fit(_projected3dCF_model, sepToFit, CFToFit, p0=[5.0, 1.8], sigma=sigmaToFit)
            except Exception as e:
                print(f"Error fitting: {e}")
                return

            r0Fit, r0ErrFit, gammaFit, gammaErrFit = popt[0],np.sqrt(pcov[0,0]),popt[1],np.sqrt(pcov[1,1])
            projectedFitParams = (r0Fit, r0ErrFit, gammaFit, gammaErrFit)
            print('Curve fitting parameters:\nr0 = %0.2lf +/- %0.2lf\ngamma = %0.2lf +/- %0.2lf\n' %projectedFitParams)
            bestFitModelCurve = _projected3dCF_model(sepToPlot, r0Fit, gammaFit)
            plt.errorbar(sepToFit, CFToFit, CFErrToFit, ls='none',capsize=5,ms=10,marker='o',mew=1.0,mec='black',mfc='black',ecolor='black',elinewidth=1)
            plotLabel = r"$\xi(r)=(r/r_0)^{-\gamma}$" + "\n" + r"$r_0 = %0.2f \pm %0.2f$" + "\n" + r"$\gamma = %0.2f \pm %0.2f$"
            if SVDDone:
                plotLabel = "SVD Done" + "\n" + plotLabel
            plt.plot(sepToPlot, bestFitModelCurve, color='red',label=plotLabel %projectedFitParams)

            # WRITING TO FILES

            np.savetxt(finalDirPath+os.path.sep+'CF_fit_params_covariance.txt', pcov, fmt='%f')
            np.savetxt(finalDirPath+os.path.sep+'CF_fit_params.txt', [projectedFitParams], fmt='%f', delimiter='\n')
            np.savetxt(finalDirPath+os.path.sep+'sepFitRange.txt', [sepMinToFit, sepMaxToFit], fmt='%f', delimiter='\n')


            if computeIC:
                print("IC Computation is not coded for 3d...") #TODO


    plt.xscale(plotXScale)
    plt.yscale(plotYScale)
    plt.xlabel(cfXLabel,labelpad=10)
    plt.ylabel(cfYLabel,labelpad=0.5)
    plt.legend()
    plt.savefig(cfFigName , dpi=300, bbox_inches = 'tight')
    plt.close()

    # PLOTTING MCF

    if doMCF:

        if len(marks) >= 10:
            colors = bp.d3['Category10'][10]+bp.d3['Category10'][10]
        elif len(marks) >= 3:
            colors=bp.d3['Category10'][len(marks)]
        elif len(marks) ==2:
            colors=['#ff7f0e','#2ca02c']
        else:
            colors=['black']

        markers=['s','H','v','+','x','d','s','^','p','D','o','h','*','H','v','+','x','d']

        fig,ax_now=plt.subplots(nrows=1,ncols=1,sharex=False)
        fig.set_size_inches(5,5)

        for mark_i, mark in enumerate(marks):
            mcfRealAllToPlot = np.loadtxt(resultsDirName+os.path.sep+'mcfRealAll_%s_filtered.txt' %mark)
            sepMcf = mcfRealAllToPlot[:,0]
            markedCf = mcfRealAllToPlot[:,1]

            allCopiesMcfs = mcfRealAllToPlot[:, 2:nCopies+2]

            mcfErr = np.std(allCopiesMcfs, axis=1)

            np.savetxt(finalDirPath+os.path.sep+'final_mcf_%s.txt' %mark, np.transpose([sepMcf, markedCf, mcfErr]), fmt='%f', delimiter='\t')

            ax_now.errorbar(sepMcf, markedCf, mcfErr, color=colors[mark_i],capsize=3,ms=6,marker=markers[mark_i],mew=1.0,mec=colors[mark_i],mfc=colors[mark_i],ecolor=colors[mark_i],elinewidth=1,lw=1.0,label="%s" %(marks[mark_i]))

            ax_now.axhline(y=1, color='black', linestyle='dashed')

        plt.xscale(plotXScale)
        plt.xlabel(mcfXLabel,labelpad=10)
        plt.ylabel(mcfYLabel,labelpad=0.5)
        ax_now.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax_now.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        plt.legend(numpoints=1,frameon=False,loc=0)

        plt.grid(False)
        plt.subplots_adjust(hspace=0.0,wspace=0.2)
        plt.savefig(mcfFigName, dpi=300, bbox_inches = 'tight')
        plt.close()

    return None
