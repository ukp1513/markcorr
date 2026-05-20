import numpy as np
from scipy.stats import rankdata
from astropy.table import Table
from nugundam import acf, AngularAutoConfig, CatalogColumns, AngularBinning, AngularGridSpec, WeightSpec
import os

def omega_theta(raReal, decReal, raRand, decRand, thMin, thNBins, thBinWidth, 
                doBoot=False, weighted=False, weightReal = None, doParallelGundam=False):

    gals = Table([raReal, decReal], names=('ra', 'dec'))
    rans = Table([raRand, decRand], names=('ra', 'dec'))

    if weighted is True:
        if weightReal is None:
            print("Error: weightReal must be provided when weighted is True.")
            return None, None, None
        gals['wei'] = weightReal
        weights = WeightSpec(weight_mode="weighted")
    else:
        weights = WeightSpec(weight_mode="unweighted")

    if doParallelGundam is True:
        ncores = os.cpu_count() or 1
        nthreads = max(1, int(0.8 * ncores))
    else:
        nthreads = 1

    config = AngularAutoConfig(estimator = "LS", 
                               columns_data=CatalogColumns(ra='ra', dec='dec'),
                               columns_random=CatalogColumns(ra='ra', dec='dec'),
                               binning=AngularBinning.from_binsize(nsep=int(thNBins),
                                                                   sepmin=thMin,
                                                                   dsep=thBinWidth,
                                                                   logsep=True),
                               grid=AngularGridSpec(autogrid=True, pxorder="cell-dec"),
                               weights = weights,
                               nthreads=nthreads,
    )

    result = acf(gals, rans, config)
    th = result.theta_centers
    omega = result.wtheta
    omegaErr = result.wtheta_err if doBoot else None

    return th, omega, omegaErr

def weighted_omega_theta(raReal, decReal, weightReal, raRand, decRand, thMin, thNBins, 
                         thBinWidth, doBoot=False, doParallelGundam=False):

    gals = Table([raReal, decReal], names=('ra', 'dec'))
    rans = Table([raRand, decRand], names=('ra', 'dec'))

    gals['wei'] = weightReal # nugundam normalizes the weight inside it.
    rans['wei'] = 1.

    if doParallelGundam is True:
        ncores = os.cpu_count() or 1
        nthreads = max(1, int(0.8 * ncores))

    config = AngularAutoConfig(estimator = "LS", 
                               columns_data=CatalogColumns(ra='ra', dec='dec'),
                               columns_random=CatalogColumns(ra='ra', dec='dec'),
                               binning=AngularBinning.from_binsize(nsep=int(thNBins),
                                                                   sepmin=thMin,
                                                                   dsep=thBinWidth,
                                                                   logsep=True),
                               grid=AngularGridSpec(autogrid=True, pxorder="cell-dec"),
                               weights = WeightSpec(weight_mode="weighted", data_col="wei"),
                               nthreads=nthreads,
    )

    result = acf(gals, rans, config)    
    th = result.theta_centers
    weightedOmega = result.wtheta
    weightedOmegaErr = result.wtheta_err if doBoot else None

    return th, weightedOmega, weightedOmegaErr

def mcf_theta(omegaTh, weightedOmegaTh):
    MTheta = (1 + weightedOmegaTh)/(1 + omegaTh)
    return MTheta

def do_compute(realTab, realProperties, randTab, thMin, thNBins, thBinWidth, doRanking=True, realRaCol='RA',realDecCol='DEC',randRaCol='RA', 
               randDecCol='Dec', 
               doBoot=False, weight_w_theta = False, weight_col = None, doParallelGundam=False):
    
    raReal = realTab[realRaCol]
    decReal = realTab[realDecCol]
    weightReal = realTab[weight_col] if weight_w_theta else None

    raRand = randTab[randRaCol]
    decRand = randTab[randDecCol]

    th, omega, _ = omega_theta(raReal, decReal, weightReal, raRand, decRand, thMin, thNBins, thBinWidth, doBoot=doBoot)
    
    
    thOmegaMcfs = np.empty((len(th), 0))

    thOmegaMcfs = np.hstack((thOmegaMcfs, th.reshape(len(th), 1)))
    thOmegaMcfs = np.hstack((thOmegaMcfs, omega.reshape(len(th), 1)))

    if len(realProperties) >= 1:

        for prop_i in realProperties:

            propNow = np.array(realTab[prop_i])

            if doRanking:
                propNowRanked = rankdata(propNow)
                weightRealForMCF = propNowRanked
            else:
                weightRealForMCF = propNow

            th, weightedOmega, _ = weighted_omega_theta(raReal, decReal, weightReal=weightRealForMCF, raRand=raRand, decRand=decRand, 
                                                         thMin=thMin, thNBins=thNBins, thBinWidth=thBinWidth, 
                                                        doBoot=doBoot)

            MThetaArray = np.array(mcf_theta(omega, weightedOmega)).reshape(len(th), 1)

            thOmegaMcfs = np.hstack((thOmegaMcfs, MThetaArray))

    return thOmegaMcfs
