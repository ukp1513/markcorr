import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy.optimize import curve_fit
import bokeh.palettes as bp
from . import integral_constrain
from matplotlib.ticker import FormatStrFormatter
from astropy.table import Table
import emcee
from matplotlib.gridspec import GridSpec
import seaborn as sns

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
    from scipy.special import gamma as Gamma
    return rp * pow((r0/rp), gam) * Gamma(0.5) * Gamma(0.5*(gam-1)) / Gamma(0.5*gam)

def log_likelihood_fullcovar(CFparams, sep, CFObs, covMat, modelFunc):
    CFModel = modelFunc(sep, *CFparams)
    delta = CFObs - CFModel
    invCovMat = np.linalg.inv(covMat)
    chi2 = delta @ invCovMat @ delta
    return -0.5 * chi2

def log_likelihood_diagcovar(CFparams, sep, CFObs, CFSigma, modelFunc):
    CFModel = modelFunc(sep, *CFparams)
    delta = CFObs - CFModel
    chi2 = np.sum(delta**2 / CFSigma**2)
    return -0.5 * chi2

def log_prior(CFparams, cfType):
    if cfType == 'angular':
        A, gam = CFparams
        if 0 < A < 10 and 0.5 < gam < 3.5:
            return 0.0
    else:
        r0, gam = CFparams
        if 0 < r0 < 50 and 0.5 < gam < 3.5:
            return 0.0
    return -np.inf

def log_posterior(CFparams, sep, cfObs, sigmaFit, modelFunc, cfType, useFullCovar=True):
    lp = log_prior(CFparams, cfType)
    if not np.isfinite(lp):
        return -np.inf
    if useFullCovar:
        return lp + log_likelihood_fullcovar(CFparams, sep, cfObs, sigmaFit, modelFunc)
    else:
        return lp + log_likelihood_diagcovar(CFparams, sep, cfObs, sigmaFit, modelFunc)

def _run_mcmc(cfType, sepToFit, CFToFit, sigmaFit, nWalkers=50, nSteps=5000, useFullCovar=True):

    # Select model
    if cfType == 'angular':
        modelFunc = _angularCF_model
        initial = [0.01, 1.8]
    elif cfType == '3d_redshift':
        modelFunc = _redshift3dCF_model
        initial = [5.0, 1.8]
    elif cfType == '3d_projected':
        modelFunc = _projected3dCF_model
        initial = [5.0, 1.8]
    else:
        raise ValueError("Unknown cfType")

    nDim = len(initial)
    pos = initial + 1e-4 * np.random.randn(nWalkers, nDim)

    sampler = emcee.EnsembleSampler(nWalkers, nDim, log_posterior, args=(sepToFit, CFToFit, sigmaFit, modelFunc, cfType, useFullCovar))
    sampler.run_mcmc(pos, nSteps, progress=True)

    return sampler

def model_curve_plot_label(sep, param, cfType):
    if cfType == 'angular':
        modelFunc = _angularCF_model
        plotLabel = r"$\omega(\theta)=A \theta^{1-\gamma}$" + "\n" + r"$A = %0.2f \pm %0.2f$" + "\n" + r"$\gamma = %0.2f \pm %0.2f$"
    elif cfType == '3d_redshift':
        modelFunc = _redshift3dCF_model
        plotLabel = r"$\xi(s)=(s/s_0)^{-\gamma}$" + "\n" + r"$s_0 = %0.2f \pm %0.2f$" + "\n" + r"$\gamma = %0.2f \pm %0.2f$"
    elif cfType == '3d_projected':
        modelFunc = _projected3dCF_model
        plotLabel = r"$\xi(r)=(r/r_0)^{-\gamma}$" + "\n" + r"$r_0 = %0.2f \pm %0.2f$" + "\n" + r"$\gamma = %0.2f \pm %0.2f$"
    else:
        print("Wrong cfType")
        return 1


    return modelFunc(sep, param[0], param[2]), plotLabel

def plot_posterior(samples, param1Lab, param2Lab=r'$\gamma$', figFileName=None):

    param1Samples = samples[:, 0]
    param2Samples = samples[:, 1]

    # Median and std
    best = np.median(samples, axis=0)
    errs = np.std(samples, axis=0)
    p1_med, p2_med = best
    p1_std, p2_std = errs

    fig = plt.figure(figsize=(5, 5))
    gs = GridSpec(2, 2, figure=fig)

    # Top-left: KDE for param1
    ax0 = fig.add_subplot(gs[0, 0])
    sns.kdeplot(x=param1Samples, ax=ax0, fill=True, color="skyblue", bw_adjust=0.8)
    ax0.axvline(p1_med, color='blue', linestyle='--', label='Median', lw=1.5)
    ax0.axvline(p1_med-p1_std, color='blue', linestyle='dotted', lw=1.0)
    ax0.axvline(p1_med+p1_std, color='blue', linestyle='dotted', lw=1.0)
    ax0.set_xticklabels([])

    # Empty top-right
    fig.add_subplot(gs[0, 1]).axis('off')

    # Bottom-left: 2D KDE with 1σ, 2σ, 3σ contours
    ax1 = fig.add_subplot(gs[1, 0])
    sns.kdeplot(
        x=param1Samples, y=param2Samples, ax=ax1,
        fill=True, cmap="Blues", levels=10,
        thresh=0, bw_adjust=0.8, cumulative=False, common_norm=False
    )
    ax1.plot(p1_med, p2_med, 'o', color='black', label='Median')
    ax1.set_xlabel(param1Lab)
    ax1.set_ylabel(param2Lab)

    # Bottom-right: KDE for param2 (vertical)


    ax2 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(y=param2Samples, ax=ax2, fill=True, color="salmon", bw_adjust=0.8)
    ax2.axhline(p2_med, color='red', linestyle='--', label='Median', lw=1.5)
    ax2.axhline(p2_med-p2_std, color='red', linestyle='dotted', lw=1.0)
    ax2.axhline(p2_med+p2_std, color='red', linestyle='dotted', lw=1.0)
    ax2.set_yticklabels([])

    plt.tight_layout()
    if figFileName:
        plt.savefig(figFileName, dpi=300, bbox_inches='tight')
    plt.close()
    return None

def plot_error_ellipse(popt, pcov, param1Lab, param2Lab=r'$\gamma$', figFileName=None):

    from scipy.stats import norm
    from matplotlib.patches import Ellipse

    p1, p2 = popt[0], popt[1]
    p1_std, p2_std = np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1])

    fig = plt.figure(figsize=(5, 5))
    gs = GridSpec(2, 2, figure=fig)

    # Top-left: 1D Gaussian for param1
    ax0 = fig.add_subplot(gs[0, 0])
    x1 = np.linspace(p1 - 4*p1_std, p1 + 4*p1_std, 500)
    y1 = norm.pdf(x1, loc=p1, scale=p1_std)
    ax0.plot(x1, y1, color='blue')
    ax0.axvline(p1, color='blue', linestyle='--', lw=1.5, label='Mean')
    ax0.fill_between(x1, 0, y1, where=((x1 >= p1 - p1_std) & (x1 <= p1 + p1_std)),
                     color='blue', alpha=0.2, label='±1σ')
    ax0.set_xticklabels([])
    ax0.set_ylim(0,)

    # Empty top-right
    fig.add_subplot(gs[0, 1]).axis('off')

    # Bottom-left: 2D error ellipse contour
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_xlabel(param1Lab)
    ax1.set_ylabel(param2Lab)

    # Plot the best-fit point
    ax1.plot(p1, p2, 'o', color='black', label='Best fit')

    # Compute and plot the 1σ error ellipse
    vals, vecs = np.linalg.eigh(pcov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Width and height of ellipse = 2 * sqrt(eigenvalues) for 1σ
    width, height = 2 * np.sqrt(vals)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    ellipse = Ellipse(xy=(p1, p2), width=width, height=height, angle=angle,
                      edgecolor='blue', facecolor='none', lw=2, label='1σ error ellipse')
    ax1.add_patch(ellipse)

    # Set limits around the ellipse
    ax1.set_xlim(p1 - 4*p1_std, p1 + 4*p1_std)
    ax1.set_ylim(p2 - 4*p2_std, p2 + 4*p2_std)

    ax1.legend()

    # Bottom-right: 1D Gaussian for param2 (vertical)
    ax2 = fig.add_subplot(gs[1, 1])
    y2 = np.linspace(p2 - 4*p2_std, p2 + 4*p2_std, 500)
    p2_pdf = norm.pdf(y2, loc=p2, scale=p2_std)
    ax2.plot(p2_pdf, y2, color='red')
    ax2.axhline(p2, color='red', linestyle='--', lw=1.5, label='Mean')
    ax2.fill_betweenx(y2, 0, p2_pdf, where=((y2 >= p2 - p2_std) & (y2 <= p2 + p2_std)),
                      color='red', alpha=0.2, label='±1σ')
    ax2.set_yticklabels([])
    ax2.set_xlim(0,)

    plt.tight_layout()
    if figFileName:
        plt.savefig(figFileName, dpi=300, bbox_inches='tight')
    plt.close()


def do_analyze(cfType, sepMin=None, sepMax=None, sepMinToFit=None, sepMaxToFit=None, doFit2pcf=True, useFullCovar=True, doSvdFilter=False, doHartlapCorr=False, doMCF=True, realProperties=None, dirName=os.getcwd(), plotXScale='log', plotYScale='log', ignoreNegatives = True, computeIC = False, doCurveFit=True, doMCMC=False):

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
        param1Lab = r"$A$"
        param2Lab = r"$\gamma$"

    elif cfType == '3d_redshift':

        cfXLabel = r"$s \,\,\, (Mpc/h)$"
        cfYLabel = r"$\xi(s)$"
        mcfXLabel = r"$s \,\,\, (Mpc/h)$"
        mcfYLabel = r"$M (s)$"
        cfFigName = dirName+os.path.sep+"fig_3DRedshiftCF.png"
        mcfFigName = dirName+os.path.sep+"fig_3DRedshiftMCF.png"
        param1Lab = r"$r_0$"
        param2Lab = r"$\gamma$"

    elif cfType == '3d_projected':

        cfXLabel = r"$r_p \,\,\, (Mpc/h)$"
        cfYLabel = r"$\omega_p(r_p) \,\,\, (Mpc/h)$"
        mcfXLabel = r"$r_p \,\,\, (Mpc/h)$"
        mcfYLabel = r"$M_p (r_p)$"
        cfFigName = dirName+os.path.sep+"fig_3DProjectedCF.png"
        mcfFigName = dirName+os.path.sep+"fig_3DProjectedMCF.png"
        param1Lab = r"$r_0$"
        param2Lab = r"$\gamma$"

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

    dirNameSplit=dirName.split('/')
    sampleName=dirNameSplit[-1].upper()

    print("\n--------------------------------------------")
    print("\nFitting sample %s....\n" %(sampleName))
    print("--------------------------------------------\n")

    # CREATING CFRealAll.txt for JK copies

    CFRealFilePath = os.path.join(resultsDirName,'CFReal.txt')
    CFJKDirPath = os.path.join(resultsDirName,'jackknifes')
    if not os.path.exists(CFRealFilePath):
        print("CF file of real galaxy not found")
        return 1

    CFRealResult = np.loadtxt(CFRealFilePath)
    sep = CFRealResult[:,0]
    CFReal = CFRealResult[:,1]

    totalNBins = len(sep)
    nCopies=len([f for f in os.listdir(CFJKDirPath) if not f.startswith('.')])

    print('Nr. of total bins: ', totalNBins)
    print('Nr. of jackknife copies: ', nCopies)

    CFRealAll = np.empty((totalNBins, nCopies + 2), dtype=float)

    CFRealAll[:, 0] = sep
    CFRealAll[:, 1] = CFReal

    # setting min and max seps to analyze if not given
    sepMin = sep[0] if sepMin is None else sepMin
    sepMax = sep[1] if sepMax is None else sepMax
    sepMinToFit = sepMin if sepMinToFit is None else sepMinToFit
    sepMaxToFit = sepMax if sepMaxToFit is None else sepMaxToFit

    for copy in range(nCopies):
        CFJK = np.loadtxt(os.path.join(CFJKDirPath, 'CFJackknife_jk%d.txt' % (copy + 1)))[:, 1]
        CFRealAll[:, copy + 2] = CFJK


    np.savetxt(os.path.join(resultsDirName, "CFRealAll.txt"), CFRealAll,delimiter="\t",fmt='%f')

    if doMCF:
        # COLLECTING MCFs
        if not realProperties:
            realPropFilePath = biproductDirName+os.path.sep+'real_properties.txt'
            with open(realPropFilePath, 'r') as f:
                realProperties = [line.strip() for line in f]

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

    filterIndexCF=list(set(filterIndexCF))

    #REMOVING NAN BINS FROM CF FILE
    CFRealAllFiltered=np.delete(CFRealAll, filterIndexCF, axis=0)

    nBinsCFFiltered=totalNBins-len(filterIndexCF)

    print("Number of CF bins with non-nan values:", nBinsCFFiltered)
    np.savetxt(os.path.join(resultsDirName, 'CFRealAll_filtered.txt'), CFRealAllFiltered, delimiter='\t', fmt='%f')

    if nBinsCFFiltered == 0:
        print("There are no bins with reliable CF within the fitting range")
        return

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

    if CFRealAllToFit.shape[0] < 2:
        print("\nCannot compute covariance: need at least two valid bins!")
        return None

    np.savetxt(os.path.join(resultsDirName, 'CFRealAll_filtered_tofit.txt'), CFRealAllToFit, delimiter='\t', fmt='%f')

    if nBinsCFToFit == 0:
        print("There are no bins with reliable CF within the fitting range")
        return

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




        if covariancingSuccess:
            np.savetxt(biproductDirName+os.path.sep+"cov_mat.txt",np.transpose(covMatSVD),delimiter="\t",fmt='%f')
            CFErrToFit = np.sqrt(np.diag(covMatSVD))
        else:
            CFErrToFit = [0 for i in range(len(sepToFit))]

        np.savetxt(finalDirPath+os.path.sep+'final_CF.txt', np.transpose([sepToFit, CFToFit, CFErrToFit]), fmt='%f', delimiter='\t')

        if useFullCovar:
            sigmaToFit = covMatSVD
            print("\nFitting using full covariance matrix...\n")
        else:
            sigmaToFit = CFErrToFit
            print("\nFitting using only diagonal elements of the covariance matrix...\n")

        #FIT USING MCMC

        # setup MCMC

        if doMCMC:

            sampler = _run_mcmc(cfType, sepToFit, CFToFit, sigmaToFit, nWalkers=32, nSteps=5000, useFullCovar=useFullCovar)

            samples = sampler.get_chain(discard=1000, thin=10, flat=True)

            plot_posterior(samples, param1Lab, param2Lab, figFileName=dirName+os.path.sep+"fig_param_covariance.png")

            best = np.median(samples, axis=0)
            errs = np.std(samples, axis=0)

            CFFitParams = (best[0], errs[0], best[1], errs[1])

            if cfType == 'angular':
                A, gamma = best
                print("MCMC fit: A = %.3f ± %.3f, gamma = %.3f ± %.3f" % (A, errs[0], gamma, errs[1]))
            else:
                r0, gamma = best
                print("MCMC fit: r0 = %.3f ± %.3f, gamma = %.3f ± %.3f" % (r0, errs[0], gamma, errs[1]))

            np.savetxt(finalDirPath+os.path.sep+"CF_MCMC_chain.txt", samples)

        # FIT USING CURVE_FIT

        if doCurveFit:

            if cfType == 'angular':

                try:
                    popt, pcov = curve_fit(_angularCF_model, sepToFit, CFToFit, sigma=sigmaToFit)
                except Exception as e:
                    print(f"Error fitting: {e}")
                    return

                AFit, AErrFit, gammaFit, gammaErrFit = popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1])
                CFFitParams = (AFit, AErrFit, gammaFit, gammaErrFit)
                print('Curve fitting parameters:\nA = %0.2lf +/- %0.2lf\ngamma = %0.2lf +/- %0.2lf\n' %CFFitParams)
                bestFitModelCurve = _angularCF_model(sepToPlot, AFit, gammaFit)

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

                if computeIC:
                    print("IC Computation is not coded for 3d...") #TODO

            plot_error_ellipse(popt, pcov, param1Lab, param2Lab, figFileName=dirName+os.path.sep+"fig_param_covariance.png")

        # PLOTTING

        plt.errorbar(sepToFit, CFToFit, CFErrToFit,ls='none',capsize=5,ms=10,marker='o',mew=1.0,mec='black',mfc='black',ecolor='black',elinewidth=1)

        bestFitModelCurve, plotLabel = model_curve_plot_label(sepToPlot, CFFitParams, cfType)
        if SVDDone is True:
            plotLabel = "SVD Done" + "\n" + plotLabel
        plt.plot(sepToPlot, bestFitModelCurve, color='red',label=plotLabel %CFFitParams)

    plt.title(sampleName)
    plt.xscale(plotXScale)
    plt.yscale(plotYScale)
    plt.xlabel(cfXLabel,labelpad=10)
    plt.ylabel(cfYLabel,labelpad=0.5)
    plt.legend()
    plt.savefig(cfFigName , dpi=300, bbox_inches = 'tight')
    plt.close()

     # WRITING TO FILES

    #np.savetxt(finalDirPath+os.path.sep+'CF_fit_params_covariance.txt', pcov, fmt='%f')
    np.savetxt(finalDirPath+os.path.sep+'CF_fit_params.txt', [CFFitParams], fmt='%f', delimiter='\n')
    np.savetxt(finalDirPath+os.path.sep+'sepFitRange.txt', [sepMinToFit, sepMaxToFit], fmt='%f', delimiter='\n')



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

        plt.title(sampleName)
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
