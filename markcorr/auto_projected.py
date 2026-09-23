import numpy as np
from scipy.stats import rankdata
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from nugundam import pcf, ProjectedAutoConfig, ProjectedCatalogColumns, ProjectedBinning, ProjectedGridSpec, WeightSpec, DistanceSpec
import os

def comoving_distance_Mpch(redshift, cosmologyH0Om0):
    """
    Computes the comoving distance in units of Mpc/h for a given redshift.

    Args:
        redshift: Scalar or array of redshift values.

        cosmologyH0Om0: List containing cosmological parameters in the form
            ``[H0, Om0]``.

    Returns:
        comDist:
            Comoving distance in units of Mpc/h.
    """
        
    H0, Om0 = cosmologyH0Om0
    cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)

    littleH = H0/100.
    comDist = cosmology.comoving_distance(redshift).value*littleH
    return comDist

def omegap_rp(raReal, decReal, zReal, raRand, decRand, zRand, rpMin, rpNBins, rpBinWidth, piNBins, piBinWidth, 
              cosmologyH0Om0, doParallelGundam=False, doBoot=False):
    """
    Computes the projected two-point correlation function w_p(r_p).

    Args:
        raReal: Right ascension values of the real catalogue in degrees.
        decReal: Declination values of the real catalogue in degrees.
        zReal: Redshift values of the real catalogue.
        raRand: Right ascension values of the random catalogue in degrees.
        decRand: Declination values of the random catalogue in degrees.
        zRand: Redshift values of the random catalogue.
        rpMin: Minimum projected separation in Mpc/h.
        rpNBins: Number of logarithmic projected separation bins.
        rpBinWidth: Logarithmic projected separation bin width.
        piNBins: Number of line-of-sight separation bins.
        piBinWidth: Line-of-sight separation bin width.
        cosmologyH0Om0: List containing cosmological parameters in the form ``[H0, Om0]``.
        doParallelGundam: If ``True``, enables multithreading in nugundam.
        doBoot: If ``True``, returns bootstrap uncertainties from nugundam.

    Returns:
        rp: Array of projected separation bin centres in Mpc/h.
        omegaP: Array of projected correlation-function values.
        omegaPErr: Array of projected correlation-function uncertainties.
    """

    H0, OmegaM = cosmologyH0Om0

    gals = Table([raReal, decReal, zReal], names=('ra', 'dec', 'z'))
    rans = Table([raRand, decRand, zRand], names=('ra', 'dec', 'z'))

    gals['wei'] = 1.
    rans['wei'] = 1.

    if doParallelGundam is True:
        ncores = os.cpu_count() or 1
        nthreads = max(1, int(0.8 * ncores))
    else:
        nthreads = 1

    config = ProjectedAutoConfig(estimator = "LS", 
                               columns_data=ProjectedCatalogColumns(ra='ra', dec='dec', redshift='z'),
                               columns_random=ProjectedCatalogColumns(ra='ra', dec='dec', redshift='z'),
                               binning=ProjectedBinning.from_binsize(nsepp=int(rpNBins),
                                                                   seppmin=rpMin,
                                                                   dsepp=rpBinWidth,
                                                                   logsepp=True,
                                                                   nsepv=int(piNBins),
                                                                   dsepv=piBinWidth),
                               grid=ProjectedGridSpec(autogrid=True, pxorder="natural"),
                               weights = WeightSpec(weight_mode="unweighted"),
                               nthreads=nthreads,
                               distance=DistanceSpec(calcdist=True, h0=H0, omegam=OmegaM, omegal=(1-OmegaM))
    )

    result = pcf(gals, rans, config)
    rp = result.rp_centers*(H0/100.) # convert to Mpc/h
    omegaP = result.wp*(H0/100.)
    omegaPErr = result.wp_err if doBoot else np.zeros_like(rp)

    return rp, omegaP, omegaPErr

def weighted_omegap_rp(raReal, decReal, zReal, weightReal, raRand, decRand, zRand, rpMin, rpNBins, rpBinWidth, piNBins, piBinWidth, 
              cosmologyH0Om0, doParallelGundam=False, doBoot=False):
    """
    Computes the weighted projected two-point correlation function.

    Args:
        raReal: Right ascension values of the real catalogue in degrees.
        decReal: Declination values of the real catalogue in degrees.
        zReal: Redshift values of the real catalogue.
        weightReal: Array of weights assigned to sources in the real catalogue.
        raRand: Right ascension values of the random catalogue in degrees.
        decRand: Declination values of the random catalogue in degrees.
        zRand: Redshift values of the random catalogue.
        rpMin: Minimum projected separation in Mpc/h.
        rpNBins: Number of logarithmic projected separation bins.
        rpBinWidth: Logarithmic projected separation bin width.
        piNBins: Number of line-of-sight separation bins.
        piBinWidth: Line-of-sight separation bin width.
        cosmologyH0Om0: List containing cosmological parameters in the form ``[H0, Om0]``.
        doParallelGundam: If ``True``, enables multithreading in nugundam.
        doBoot: If ``True``, returns bootstrap uncertainties from nugundam.

    Returns:
        rp: Array of projected separation bin centres in Mpc/h.
        weightedOmegaP: Array of weighted projected correlation-function values.
        weightedOmegaPErr: Array of weighted projected correlation-function uncertainties.
    """

    H0, OmegaM = cosmologyH0Om0

    gals = Table([raReal, decReal, zReal], names=('ra', 'dec', 'z'))
    rans = Table([raRand, decRand, zRand], names=('ra', 'dec', 'z'))

    gals['wei'] = weightReal
    rans['wei'] = 1.

    if doParallelGundam is True:
        ncores = os.cpu_count() or 1
        nthreads = max(1, int(0.8 * ncores))
    else:
        nthreads = 1

    config = ProjectedAutoConfig(estimator = "LS", 
                               columns_data=ProjectedCatalogColumns(ra='ra', dec='dec', redshift='z'),
                               columns_random=ProjectedCatalogColumns(ra='ra', dec='dec', redshift='z'),
                               binning=ProjectedBinning.from_binsize(nsepp=int(rpNBins),
                                                                   seppmin=rpMin,
                                                                   dsepp=rpBinWidth,
                                                                   logsepp=True,
                                                                   nsepv=int(piNBins),
                                                                   dsepv=piBinWidth),
                               grid=ProjectedGridSpec(autogrid=True, pxorder="natural"),
                               weights = WeightSpec(weight_mode="weighted", data_col="wei"),
                               nthreads=nthreads,
                               distance=DistanceSpec(calcdist=True, h0=H0, omegam=OmegaM, omegal=(1-OmegaM))
    )

    result = pcf(gals, rans, config)
    rp = result.rp_centers*(H0/100.)
    weightedOmegaP = result.wp*(H0/100.)
    weightedOmegaPErr = result.wp_err if doBoot else np.zeros_like(rp)

    return rp, weightedOmegaP, weightedOmegaPErr

def mcf_rp(rp, omegaP, weightedOmegaP):
    """
    Computes the projected marked correlation function.

    Args:
        rp: Array of projected separation values in Mpc/h.
        omegaP: Array of unweighted projected correlation-function values.
        weightedOmegaP: Array of weighted projected correlation-function values.

    Returns:
        MpRp: Array containing the projected marked correlation function.
    """
    MpRp = (1 + (weightedOmegaP/rp))/(1 + (omegaP/rp))
    return MpRp

def do_compute(realTab, realProperties, randTab, rpMin, rpNBins, rpBinWidth, piNBins, piBinWidth, doRanking, 
               realRaCol='RA', realDecCol='Dec', realZCol='Z', randRaCol='RA', randDecCol='Dec', randZCol='Z', 
               cosmologyH0Om0=None, doBoot=False):
    """
    Computes the projected two-point correlation function and optional
    marked correlation functions.

    Args:
        realTab: Astropy table containing the real galaxy catalogue.
        realProperties: List of source-property column names used for marked correlation functions.
        randTab: Astropy table containing the random catalogue.
        rpMin: Minimum projected separation in Mpc/h.
        rpNBins: Number of logarithmic projected separation bins.
        rpBinWidth: Logarithmic projected separation bin width.
        piNBins: Number of line-of-sight separation bins.
        piBinWidth: Line-of-sight separation bin width.
        doRanking: If ``True``, rank-transforms the source properties before computing marked correlation functions.
        realRaCol: Column name of right ascension in the real catalogue.
        realDecCol: Column name of declination in the real catalogue.
        realZCol: Column name of redshift in the real catalogue.
        randRaCol: Column name of right ascension in the random catalogue.
        randDecCol: Column name of declination in the random catalogue.
        randZCol: Column name of redshift in the random catalogue.
        cosmologyH0Om0: List containing cosmological parameters in the form ``[H0, Om0]``.
        doBoot: If ``True``, returns bootstrap uncertainties from nugundam.

    Returns:
        rpOmegaMcfs: Two-dimensional array containing projected separation values, projected correlation-function values, 
        and marked correlation-function values for each supplied property.
    """

    if cosmologyH0Om0 is None:
        cosmologyH0Om0=[70.0, 0.3]

    raReal = realTab[realRaCol]
    decReal = realTab[realDecCol]
    zReal = realTab[realZCol]

    raRand = randTab[randRaCol]
    decRand = randTab[randDecCol]
    zRand = randTab[randZCol]

    rp, omegaP, _ = omegap_rp(raReal, decReal, zReal, raRand, decRand, zRand, rpMin, rpNBins, rpBinWidth, piNBins, piBinWidth, 
                              cosmologyH0Om0, doBoot=doBoot)

    rpOmegaMcfs = np.empty((len(rp), 0))

    rpOmegaMcfs = np.hstack((rpOmegaMcfs, rp.reshape(len(rp), 1)))
    rpOmegaMcfs = np.hstack((rpOmegaMcfs, omegaP.reshape(len(rp), 1)))

    if len(realProperties) >= 1:

        for prop_i in realProperties:

            propNow = np.array(realTab[prop_i])

            if doRanking:
                weightReal = rankdata(propNow)
            else:
                weightReal = propNow

            rp, weightedOmega, _ = weighted_omegap_rp(raReal, decReal, zReal, weightReal, raRand, decRand, zRand, 
                                                      rpMin, rpNBins, rpBinWidth, piNBins, piBinWidth, cosmologyH0Om0, 
                                                      doBoot=doBoot)

            MpRpArray = np.array(mcf_rp(rp, omegaP, weightedOmega)).reshape(len(rp), 1)

            rpOmegaMcfs = np.hstack((rpOmegaMcfs, MpRpArray))

    return rpOmegaMcfs
