import numpy as np
from scipy.stats import rankdata
from astropy.table import Table
from nugundam import acf, accf, AngularAutoConfig, AngularCrossConfig, CatalogColumns, AngularBinning, AngularGridSpec, WeightSpec

def omega_theta(raReal1, decReal1, weightReal1=None, raRand=None, decRand=None, 
                raReal2=None, decReal2=None, weightReal2=None, 
                thMin=None, thNBins=None, thBinWidth=None, doBoot=False, crossCF=False):

    gals1 = Table([raReal1, decReal1], names=('ra', 'dec'))
    rans1 = Table([raRand, decRand], names=('ra', 'dec'))
    rans1['wei'] = 1.
    rans2 = rans1.copy()  # For cross-correlation, we need a second random catalog

    if weightReal1 is None:
        gals1['wei'] = 1.
    else:
        gals1['wei'] = weightReal1 #/np.mean(weightReal1)

    if crossCF is True:
        gals2 = Table([raReal2, decReal2], names=('ra', 'dec'))
        if weightReal2 is None:
            gals2['wei'] = 1.
        else:
            gals2['wei'] = weightReal2/np.mean(weightReal2)

        config = AngularCrossConfig(estimator="LS",
                                        columns_data1=CatalogColumns(ra="ra", dec="dec", weight='wei'), 
                                        columns_random1=CatalogColumns(ra="ra", dec="dec", weight='wei'), 
                                        columns_data2=CatalogColumns(ra="ra", dec="dec", weight='wei'),
                                        columns_random2=CatalogColumns(ra="ra", dec="dec", weight='wei'),
                                        binning = AngularBinning.from_binsize(nsep=int(thNBins), 
                                                                            sepmin=thMin, 
                                                                            dsep=thBinWidth, 
                                                                            logsep=True),
                                        grid=AngularGridSpec(autogrid=True, pxorder="cell-dec"),
                                        weights=WeightSpec(weight_mode="weighted"),
                                        nthreads=4)

        result = accf(gals1, gals2, config, random1=rans1, random2=rans2)

        th = result.theta_centers
        d1d2 = result.counts.d1d2
        d1r2 = result.counts.d1r2
        r1d2 = result.counts.r1d2
        r1r2 = result.counts.r1r2

        d1d2normfactor = sum(gals1['wei']) * sum(gals2['wei'])
        d1r2normfactor = sum(gals1['wei']) * sum(rans2['wei'])
        r1d2normfactor = sum(rans1['wei']) * sum(gals2['wei'])
        r1r2normfactor = sum(rans1['wei']) * sum(rans2['wei'])

        d1d2normalized = d1d2 / d1d2normfactor
        r1r2normalized = r1r2 / r1r2normfactor
        d1r2normalized = d1r2 / d1r2normfactor
        r1d2normalized = r1d2 / r1d2normfactor    

        omega = (d1d2normalized - d1r2normalized - r1d2normalized + r1r2normalized) / r1r2normalized

        print("am:", d1d2[0], d1r2[0], r1d2[0], r1r2[0])
        print("am:", d1d2normfactor, d1r2normfactor, r1d2normfactor, r1r2normfactor)
        print("am:", d1d2normalized[0], d1r2normalized[0], r1d2normalized[0], r1r2normalized[0], omega[0])

        

        
        omegaErr = result.wtheta_err if doBoot else None

    else:
        config = AngularAutoConfig(estimator = "LS",
                                    columns_data=CatalogColumns(ra='ra', dec='dec', weight='wei'),
                                    columns_random=CatalogColumns(ra='ra', dec='dec', weight='wei'),
                                    binning=AngularBinning.from_binsize(nsep=int(thNBins),
                                                                           sepmin=thMin,
                                                                           dsep=thBinWidth,
                                                                           logsep=True),
                                    grid=AngularGridSpec(autogrid=True, pxorder="cell-dec"),
                                    weights = WeightSpec(weight_mode="weighted"),
                                    nthreads = 4,
        )
        
        result = acf(gals1, rans1, config)

        th = result.theta_centers
        dd = result.counts.dd
        dr = result.counts.dr
        rr = result.counts.rr

        ddnormfactor = sum(gals1['wei']) * (sum(gals1['wei']) - 1) * 0.5
        rrnormfactor = len(rans1) * (len(rans1) - 1) * 0.5
        drnormfactor = sum(gals1['wei']) * len(rans1)

        ddnormalized = dd / ddnormfactor
        rrnormalized = rr / rrnormfactor
        drnormalized = dr / drnormfactor    

        omega = (ddnormalized - 2*drnormalized + rrnormalized) / rrnormalized
        omegaErr = result.wtheta_err if doBoot else None

    return th, omega, omegaErr

def mcf_theta(omegaTh, weightedOmegaTh):
    MTheta = (1 + weightedOmegaTh)/(1 + omegaTh)
    return MTheta

def do_compute(realTab1, realTab2=None, randTab=None, thMin=None, thNBins=None, thBinWidth=None, doRanking=True, 
               realRaCol1='RA',realDecCol1='DEC', realRaCol2='RA',realDecCol2='DEC', randRaCol='RA', randDecCol='Dec', 
               doBoot=False, weight_col1 = None, weight_col2=None, realProperties=None, doParallelGundam=False):

    if realTab2 is None:
        crossCF = False
    else:
        crossCF = True

    raReal1 = realTab1[realRaCol1]
    decReal1 = realTab1[realDecCol1]
    if weight_col1 is not None:
        weightReal1 = realTab1[weight_col1]
    else:
        weightReal1 = [1.0] * len(realTab1)  # Default to uniform weights if no weight column is provided

    

    if crossCF is True:
        raReal2 = realTab2[realRaCol2]
        decReal2 = realTab2[realDecCol2]
        if weight_col2 is not None:
            weightReal2 = realTab2[weight_col2]
        else:   
            weightReal2 = [1.0] * len(realTab2)
    else:
        raReal2 = None
        decReal2 = None
        weightReal2 = None

    raRand = randTab[randRaCol]
    decRand = randTab[randDecCol]

    th, omega, _ = omega_theta(raReal1, decReal1, weightReal1, raRand, decRand, raReal2, decReal2, weightReal2, 
                               thMin, thNBins, thBinWidth, doBoot=doBoot, crossCF=crossCF)
    
    thOmegaMcfs = np.empty((len(th), 0))

    thOmegaMcfs = np.hstack((thOmegaMcfs, th.reshape(len(th), 1)))
    thOmegaMcfs = np.hstack((thOmegaMcfs, omega.reshape(len(th), 1)))

    if realProperties is not None:

        for prop_i in realProperties:

            propNow = np.array(realTab1[prop_i])

            if doRanking:
                propNowRanked = rankdata(propNow)
                weightRealForMCF = propNowRanked*weightReal1
            else:
                weightRealForMCF = propNow*weightReal1

            th, weightedOmega, _ = omega_theta(raReal1, decReal1, weightReal=weightRealForMCF, raRand=raRand, decRand=decRand,
                                               raReal2=raReal2, decReal2=decReal2, weightReal2=weightReal2, 
                                                thMin=thMin, thNBins=thNBins, thBinWidth=thBinWidth, 
                                                doBoot=doBoot, crossCF=crossCF)

            MThetaArray = np.array(mcf_theta(omega, weightedOmega)).reshape(len(th), 1)

            thOmegaMcfs = np.hstack((thOmegaMcfs, MThetaArray))

    return thOmegaMcfs
