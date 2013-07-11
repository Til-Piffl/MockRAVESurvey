#!/usr/bin/python

import pylab as plb
from numpy import *
import os, cPickle, time
import ebfpy
import healpy
import transformations as trafo
from TilContainer import *
import RAVEpy
import PyTil

def Til_equal(a1,a2):
    if len(a1) == len(a2):
        if (a1 == a2).all():
            return True
    return False

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class MockRaveSurvey(object):
    def __init__(self,
                 fname,
                 outputFolder = 'as fname',
                 useAbsoluteImagDistribution = False,
                 metal_correction_flag = False,
                 VERBOSE = False,
                 DEBUG = False,
                 seed = 16052011.):
        self.VERBOSE = VERBOSE
        self.EBFfilename = fname
        self.seed = seed
        self.metal_correction = metal_correction_flag

        if outputFolder == 'as fname':
            self.outputDir = fname[:-4]
        elif outputFolder[-1] == '/':
            self.outputDir = outputFolder[:-1]
        else:
            self.outputDir = outputFolder

        self.useAbsoluteImagDistribution = useAbsoluteImagDistribution

        self.loadRealRAVEData()

        self.loadGALAXIAdata()

        if self.VERBOSE:
            print '======================================'
            print 'Start mock survey selection process.'
            print '======================================'
        self.selectMockSurveyStars()
        
        if self.VERBOSE:
            print 'Reducing Galaxia output again via the Survey selection.'
        for tag in self.Survey.keys():
            if not tag in ['/Log','/Center','/Typelist','/Typelist1']:
                self.Survey[tag] = self.Survey[tag][self.Selection]
        
        if 'TwoMASS' in self.__dict__.keys():
            del self.TwoMASS

        self.saveSurvey()

        #self.plotAppMagnitudeDistribution(normed=False)
        #plb.show()

    # =============================================================
    # =============================================================
    # =============================================================
    # FIXED PARAMETERS
    # =============================================================

    ImagResolution = 0.2
    Irange   = arange(5,15,ImagResolution)

    # Field diameter of the 6DF instrument is 5.7 degree (--> RAVE DR1 paper)
    angRad = 2.85 # degree

    # RAVE Color cut near the Galactic disk plane:
    JmK_min = 0.5 # mag
    b_Range = [0.,25.] # degree

    # 2MASS completeness boundaries
    J_Range = [6.,13.5]
    K_Range = [5.,13.]

    # RAVE Imag limits
    I_Range = [8.,13.]

    # HEALPIX
    NSIDE = 32

    # =============================================================
    # =============================================================
    # =============================================================
    # SURVEY CREATION METHODS
    # =============================================================

    def loadRealRAVEData(self):
        '''
        This function loads the RAVE (and if neccesary the 2MASS) data,
        sorts it using 'healpy' and computes the (relative) I magnitude
        distribution in each healpy pixel.
        '''
        if self.VERBOSE:
            print 'Loading RAVE data.'
        #self.RAVE = ebfpy.readEBFObjs('/scratch/01/til/Full_RAVE_plus_AddOns.ebf')
        self.RAVE = ebfpy.readEBFObjs('/scratch/01/til/RAVE_DR4.ebf')

        if self.VERBOSE:
            print 'Applying RAVE DR2 Imag correction.'
        #self.RAVE['corrImag'] = RAVEpy.RAVE_DR2_ImagCorrection(self.RAVE['/idenis'],
        #                                                self.RAVE['/j2mass'],
        #                                                self.RAVE['/k2mass'])
        self.RAVE['corrImag'] = RAVEpy.computeI2MASS(self.RAVE['/jmag_2mass'],
                                                        self.RAVE['/kmag_2mass'])

        # Extract usable data entries
        Nlines = len(self.RAVE['/raveid'])
	ABCorder = argsort(self.RAVE['/raveid'])
        use = zeros(Nlines,dtype=bool)
        use[0] = True
        SN = self.RAVE['/snr_k'][ABCorder[0]]
        iSN = 0
        for i in range(1,Nlines):
            if self.RAVE['/raveid'][ABCorder[i]] == self.RAVE['/raveid'][ABCorder[i-1]]:
                if self.RAVE['/snr_k'][ABCorder[i]] > SN:
                    use[ABCorder[iSN]] = False
                    use[ABCorder[i]] = True
                    SN = self.RAVE['/snr_k'][ABCorder[i]]
                    iSN = i
            else:
                use[ABCorder[i]] = True
                SN = self.RAVE['/snr_k'][ABCorder[i]]
                iSN = i
        goodZeroPoint = array(['*' not in self.RAVE['/zeropointflag'][i] for i in range(Nlines)])

        # ===============================================================================================
        # ===============================================================================================
        # ===============================================================================================
        # Select "good" observations
        matisse = (self.RAVE['/algo_conv_k']==0)|(self.RAVE['/algo_conv_k']==2)
        #matisse_degas = (self.RAVE['/algo_conv_k']!='1')&(self.RAVE['/algo_conv_k']!='4')
        #cb_measured = self.RAVE['/met_c'] < 1e9
        good = use &\
            matisse &\
            (self.RAVE['/correlationcoeff'] >= 10) &\
            (self.RAVE['/spectraflag']=='NaN') &\
            (self.RAVE['/xidqualityflag_2mass']=='A') &\
            goodZeroPoint &\
            (abs(self.RAVE['/correctionrv'])<10.) &\
            (self.RAVE['/c1'] == 'n') &\
            (self.RAVE['/c2'] == 'n') &\
            (self.RAVE['/c3'] == 'n') &\
            (self.RAVE['/c4'] == 'n') &\
            (self.RAVE['/c5'] == 'n') &\
            (self.RAVE['/c6'] == 'n') &\
            ((abs(self.RAVE['/b']) >= self.b_Range[1])|\
             (self.RAVE['/jmag_2mass']-self.RAVE['/kmag_2mass'] >= self.JmK_min))
        STN40 = (self.RAVE['/snr_k']>=40.)
        take = good & STN40

        if not self.useAbsoluteImagDistribution:
            complete2MASS = (self.RAVE['/jmag_2mass'] >= self.J_Range[0])&(self.RAVE['/jmag_2mass'] <= self.J_Range[1]) \
                &(self.RAVE['/kmag_2mass'] >= self.K_Range[0])&(self.RAVE['/kmag_2mass'] <= self.K_Range[1])

            #tmp = self.RAVE['/j2mass'] + 1.103*(self.RAVE['/j2mass']-self.RAVE['/k2mass']) + 0.07
            #tmp = RAVEpy.computeI2MASS(self.RAVE['/j2mass'],self.RAVE['/k2mass'])
            self.RAVE['/i2mass'] = self.RAVE['corrImag']  # I magnitude computed from 2MASS J,K according to Zwitter (priv. com.)
            InMagRange = (self.RAVE['/i2mass'] >= self.I_Range[0])&(self.RAVE['/i2mass'] <= self.I_Range[1])

            take = take & complete2MASS & InMagRange
        else:
            InMagRange = (self.RAVE['corrImag'] >= self.I_Range[0])&(self.RAVE['corrImag'] <= self.I_Range[1])
            take = take & InMagRange
        print 'Number of RAVE stars in sample: ',sum(take)

        # The fields within the Galactic plane and near the Bulge seem to be problematic so we leave them out
        problematic = \
            (abs(self.RAVE['/b']) < 5) |\
            ((abs(self.RAVE['/b']) < 10.)&(self.RAVE['/l']<45.)&(self.RAVE['/l']>315.)) |\
            (((self.RAVE['/l']>330)|(self.RAVE['/l']<30))&(self.RAVE['/b']<25)&(self.RAVE['/b']>0))
        
        # Keep only usable entries
        usable = take & (problematic == False)
        for tag in self.RAVE.keys():
            self.RAVE[tag] = self.RAVE[tag][usable]
        # ===============================================================================================
        # ===============================================================================================
        # ===============================================================================================


        if self.VERBOSE:
            print 'Computing HEALPIX indices for RAVE stars.'
        d2r = pi/180.
        tmp = healpy.ang2pix(self.NSIDE, (90.-self.RAVE['/de'])*d2r,self.RAVE['/ra']*d2r)
        self.RAVE['healpixNum'] = tmp #Id of healpix pixel with NSIDE = self.NSIDE
        x = arange(healpy.nside2npix(self.NSIDE)+1)
        PixDistRAVE = 1.*histogram(tmp,x)[0]
        if self.VERBOSE:
            healpy.mollview(PixDistRAVE,title='Number counts distribution of RAVE')
            plb.show()

        if not self.useAbsoluteImagDistribution:
            if self.VERBOSE:
                print 'Loading 2MASS data.'
            full_reload = False
            tmp_fname = '/store/dpk01/til/TwoMASS-catalogue/tmp2MASS.dat'
            if not os.path.exists(tmp_fname):
                full_reload = True
            if not full_reload:
                f = open(tmp_fname,'rb')
                tmp_nside = cPickle.load(f)
                tmp_Irange = cPickle.load(f)
                tmp_Jrange = cPickle.load(f)
                tmp_Krange = cPickle.load(f)
                tmp_JmK_min = cPickle.load(f)
                f.close()
                if tmp_nside == self.NSIDE and tmp_Irange[0] == self.I_Range[0] and tmp_Irange[1] == self.I_Range[1] \
                        and tmp_Jrange[0] == self.J_Range[0] and tmp_Jrange[1] == self.J_Range[1] \
                        and tmp_Krange[0] == self.K_Range[0] and tmp_Krange[1] == self.K_Range[1] \
                        and tmp_JmK_min == self.JmK_min:
                    f = open(tmp_fname,'rb')
                    tmp_nside = cPickle.load(f)
                    tmp_Irange = cPickle.load(f)
                    tmp_Jrange = cPickle.load(f)
                    tmp_Krange = cPickle.load(f)
                    tmp_JmK_min = cPickle.load(f)
                    self.TwoMASS = {}
                    self.TwoMASS['Imag'] = cPickle.load(f)
                    self.TwoMASS['healpixNum'] = cPickle.load(f)
                    PixDist2MASS = cPickle.load(f)
                    TwoMASSorder = cPickle.load(f)
                    f.close()
                else:
                    full_reload = True

            if full_reload:
                self.TwoMASS = ebfpy.readEBFObjs('/store/dpk01/til/TwoMASS-catalogue/2MASS-catalogue_good.ebf',
                                                 objIDs=['/dec','/ra','/j_m','/k_m'])
                #self.TwoMASS['Imag'] = self.TwoMASS['/j_m'] +1.103*(self.TwoMASS['/j_m']-self.TwoMASS['/k_m']) + 0.07
                self.TwoMASS['Imag'] = RAVEpy.computeI2MASS(self.TwoMASS['/j_m'],self.TwoMASS['/k_m'])
                
                self.TwoMASS['l'],self.TwoMASS['b'] = trafo.eq2galCoords(self.TwoMASS['/ra'],
                                                                         self.TwoMASS['/dec'],
                                                                         units='deg')
                keep = ((abs(self.TwoMASS['b']) < self.b_Range[0]) | \
                            (abs(self.TwoMASS['b']) >= self.b_Range[1]) | \
                            (self.TwoMASS['/j_m'] - self.TwoMASS['/k_m'] >= self.JmK_min)) & \
                            (self.TwoMASS['Imag'] >= self.I_Range[0])&(self.TwoMASS['Imag'] <= self.I_Range[1]) & \
                            (self.TwoMASS['/j_m'] >= self.J_Range[0])&(self.TwoMASS['/j_m'] <= self.J_Range[1]) & \
                            (self.TwoMASS['/k_m'] >= self.K_Range[0])&(self.TwoMASS['/k_m'] <= self.K_Range[1])
                for tag in self.TwoMASS.keys():
                    self.TwoMASS[tag] = self.TwoMASS[tag][keep]
            
                if self.VERBOSE:
                    print 'Computing HEALPIX indices for 2MASS stars.'
                tmp = healpy.ang2pix(self.NSIDE,(90.-self.TwoMASS['/dec'])*d2r,self.TwoMASS['/ra']*d2r)
                PixDist2MASS = 1.*histogram(tmp,x)[0]
                if self.VERBOSE:
                    healpy.mollview(PixDist2MASS,title='Number counts distribution of 2MASS')
                    plb.show()
                self.TwoMASS['healpixNum'] = tmp
                del tmp
                
                TwoMASSorder = argsort(self.TwoMASS['healpixNum'])
                f = open(tmp_fname,'wb')
                cPickle.dump(self.NSIDE,f)
                cPickle.dump(self.I_Range,f)
                cPickle.dump(self.J_Range,f)
                cPickle.dump(self.K_Range,f)
                cPickle.dump(self.JmK_min,f)
                cPickle.dump(self.TwoMASS['Imag'],f)
                cPickle.dump(self.TwoMASS['healpixNum'],f)
                cPickle.dump(PixDist2MASS,f)
                cPickle.dump(TwoMASSorder,f)
                f.close()

            # Compute relative Imag distributions
            self.ImagDist = []
            self.ProblemPixels = zeros(healpy.nside2npix(self.NSIDE))  # Pixels with stars present in RAVE but not
                                                                       # in 2MASS
            self.NmoreAll = 0   # Number of stars present in RAVE but not in 2MASS

            RAVEorder = argsort(self.RAVE['healpixNum'])

            for i in range(healpy.nside2npix(self.NSIDE)):
                if self.VERBOSE:
                    if i%1000 == 0:
                        print i
                if PixDistRAVE[i] == 0:
                    self.ImagDist.append(['none',])
                    continue
                if i == 0:
                    subRAVE  = RAVEorder[:PixDistRAVE[i]]
                    sub2MASS  = TwoMASSorder[:PixDist2MASS[i]]
                else:
                    subRAVE  = RAVEorder[sum(PixDistRAVE[:i-1]):sum(PixDistRAVE[:i])]
                    sub2MASS  = TwoMASSorder[sum(PixDist2MASS[:i-1]):sum(PixDist2MASS[:i])]

                distRAVE  = 1.*histogram(self.RAVE['/i2mass'][subRAVE],self.Irange)[0]
                dist2MASS = 1.*histogram(self.TwoMASS['Imag'][sub2MASS],self.Irange)[0]
                dist = distRAVE/dist2MASS
                dist[dist2MASS == 0] = 0.
                if not (dist2MASS >= distRAVE).all():
                    Nmore = 0
                    for j in range(len(self.Irange)-1):
                        if dist2MASS[j] < distRAVE[j]:
                            dist[j] = -(distRAVE[j] - dist2MASS[j])
                            Nmore += distRAVE[j] - dist2MASS[j]
                    if self.VERBOSE:
                        print 'At pixel %i (ra = %.1f , dec = %.1f):'\
                            %(i,healpy.pix2ang(self.NSIDE,i)[1]/d2r,90. - healpy.pix2ang(self.NSIDE,i)[0]/d2r)
                        print 'RAVE observed %i out of %i stars not present in 2MASS.'\
                            %(Nmore,sum(distRAVE))
                    self.ProblemPixels[i] = Nmore
                    self.NmoreAll += Nmore
                self.ImagDist.append(dist)
            if self.VERBOSE:
                print 'Total number of stars present in RAVE but not in 2MASS: %i'%(self.NmoreAll)
                healpy.mollview(self.ProblemPixels,title='More RAVE stars than 2MASS?')
                plb.show()
        else:
            # Compute absolute Imag distributions
            self.ImagDist = []
            RAVEorder = argsort(self.RAVE['healpixNum'])

            for i in range(healpy.nside2npix(self.NSIDE)):
                if i%100 == 0 and self.VERBOSE:
                    print i
                if PixDistRAVE[i] == 0:
                    self.ImagDist.append(['none',])
                    continue
                if i == 0:
                    subRAVE  = RAVEorder[:PixDistRAVE[i]]
                else:
                    subRAVE  = RAVEorder[sum(PixDistRAVE[:i-1]):sum(PixDistRAVE[:i])]

                distRAVE,tmp  = histogram(self.RAVE['corrImag'][subRAVE],self.Irange)
                self.ImagDist.append(distRAVE)
        return
                
    # =============================================================
    # =============================================================
    # =============================================================

    def loadGALAXIAdata(self):
        if self.VERBOSE:
            print 'Loading Galaxia all-sky data.'

        # Check whether 2MASS photometry is available; if not run Galaxia to add it
        content = ebfpy.getEBFObjIDs(self.EBFfilename)
        if not '/2MASS_Ks' in content:
            if self.metal_correction:
                os.system('TilGalaxia_Scalo86_MetalCorrection -a --psys=2MASS %s'%self.EBFfilename)
            else:
                os.system('TilGalaxia_Scalo86IMF -a --psys=2MASS %s'%self.EBFfilename)

        self.Survey = ebfpy.readEBFObjs(self.EBFfilename, objIDs = \
                                      ['/px','/py','/pz','/vx','/vy','/vz',
                                       '/UBV_I','/2MASS_J','/2MASS_Ks',
                                       '/ExBV_Schlegel','/rad',
                                       '/Center','/age','/partID'])
        if self.VERBOSE:
            print 'EBF file loaded.'
        self.Survey['IDs'] = arange(len(self.Survey['/px']))
        self.Survey['/ExBV_Schlegel'] = SchegelExtinctionCorrection(self.Survey['/ExBV_Schlegel'])

        self.distMod = 5.*log10(self.Survey['/rad']*100.)
        # Factors R = A/E(B-V) from Schafly & Finkbeiner(2011) and Yuan, Liu & Xiang (2011)
        self.Survey['Imag'] = self.Survey['/UBV_I'] + \
            self.distMod + \
            1.555 * self.Survey['/ExBV_Schlegel']     # I_DENIS = Gunn i
        self.Survey['Jmag'] = self.Survey['/2MASS_J'] + \
            self.distMod + \
            0.72 * self.Survey['/ExBV_Schlegel']
        self.Survey['Kmag'] = self.Survey['/2MASS_Ks'] + \
            self.distMod + \
            0.306 * self.Survey['/ExBV_Schlegel']
        ebfpy.addGalacticCoordinates(self.Survey)
        ebfpy.addEquatorialCoordinates(self.Survey)

        if self.VERBOSE:
            print 'Reducing Galaxia output:'
            print '  %.1f < I-mag < %.1f'%(self.I_Range[0],self.I_Range[1])
            print '  Declination < %i degree'%(int(max(self.RAVE['/de']))+1)
            print '  For %.1f < Gal. lattitude < %.1f degree: J-K > %.1f mag'\
                                %(self.b_Range[0],self.b_Range[1],self.JmK_min)
        keep = (self.Survey['DE'] <= int(max(self.RAVE['/de']))+1) \
             & (  (abs(self.Survey['b']) < self.b_Range[0]) \
                | (abs(self.Survey['b']) >= self.b_Range[1]) \
                | (self.Survey['Jmag'] - self.Survey['Kmag'] >= self.JmK_min)) \
             & (self.Survey['Jmag'] >= self.J_Range[0])&(self.Survey['Jmag'] <= self.J_Range[1]) \
             & (self.Survey['Kmag'] >= self.K_Range[0])&(self.Survey['Kmag'] <= self.K_Range[1])
        for tag in self.Survey.keys():
            if not tag in ['/Log','/Center','/Typelist','/Typelist1']:
                self.Survey[tag] = self.Survey[tag][keep]


        # Compute "proper" RAVE I magnitude
        #self.Survey['corrImag'] = RAVEpy.RAVE_DR2_ImagCorrection(self.Survey['Imag'],
        #                                                self.Survey['Jmag'],
        #                                                self.Survey['Kmag'])
        self.Survey['corrImag'] = RAVEpy.computeI2MASS(self.Survey['Jmag'],
                                                        self.Survey['Kmag'])
        if not self.useAbsoluteImagDistribution:
            #self.Survey['i2mass'] = self.Survey['Jmag'] + 1.103*(self.Survey['Jmag']-self.Survey['Kmag']) + 0.07
            self.Survey['i2mass'] = RAVEpy.computeI2MASS(self.Survey['Jmag'],self.Survey['Kmag'])
            keep = (self.Survey['i2mass'] >= self.I_Range[0])&(self.Survey['i2mass'] <= self.I_Range[1])
        else:
            keep = (self.Survey['corrImag'] >= self.I_Range[0])&(self.Survey['corrImag'] <= self.I_Range[1])
        for tag in self.Survey.keys():
            if not tag in ['/Log','/Center','/Typelist','/Typelist1']:
                self.Survey[tag] = self.Survey[tag][keep]

        if self.VERBOSE:
            print 'Computing HEALPIX indices for mock stars.'
        d2r = pi/180.
        self.Survey['healpixNum'] = healpy.ang2pix(self.NSIDE,
                                                   (90.-self.Survey['DE']) * d2r,
                                                   self.Survey['RA'] * d2r)
        return

    # =============================================================
    # =============================================================
    # =============================================================

    def selectMockSurveyStars(self):
        if self.VERBOSE:
            print 'Selecting mock stars according to the Imag distributions.'

        PixDistMock,x = histogram(self.Survey['healpixNum'],
                                  arange(healpy.nside2npix(self.NSIDE)+1))
        if self.VERBOSE:
            healpy.mollview(PixDistMock,title='Number counts distribution of Mock Galaxy')
            plb.show()

        Mockorder = argsort(self.Survey['healpixNum'])

        self.Selection = []
        for i in range(healpy.nside2npix(self.NSIDE)):
            if i%1000 == 0 and self.VERBOSE:
                print i                             # Show how many pixels are already done.
            if PixDistMock[i] == 0 or self.ImagDist[i][0] == 'none': # If no stars in this pixel, go to the next.
                continue
            if i == 0:                  # Load the ids of all stars in the current pixel
                sub  = Mockorder[:PixDistMock[i]]       
            else:
                sub  = Mockorder[sum(PixDistMock[:i-1]):sum(PixDistMock[:i])]
            
            Imag = self.Survey['corrImag'][sub]   # Load the proper magnitude distribution
                
            for j in range(len(self.Irange)-1):     # Loop over all magnitude bins
                IDs_bin = sub[(Imag >= self.Irange[j]) & (Imag < self.Irange[j+1])] # load the ids of all stars in the current mag bin
                if len(IDs_bin) == 0:
                    continue
                random.shuffle(IDs_bin)   # randomize the ordering in order to select a random sub-sample
                IDs0 = self.Survey['/partID'][IDs_bin] == 0
                tmp = copy(IDs_bin)
                IDs_bin[:sum(IDs0)] = tmp[IDs0]
                IDs_bin[sum(IDs0):] = tmp[IDs0==False]
                if self.useAbsoluteImagDistribution:
                    Nbin = self.ImagDist[i][j]       # how many stars are needed from this bin?
                else:
                    if self.ImagDist[i][j] < 0:
                        Nbin = abs(self.ImagDist[i][j])
                    else:
                        Nbin = self.ImagDist[i][j]*len(IDs_bin)
                        if self.ImagDist[i][j] != 0:
                            X = random.rand()
                            if X <= Nbin%int(Nbin): # Random rounding
                                Nbin = int(Nbin) + 1
                            else:
                                Nbin = int(Nbin)
                self.Selection.extend(IDs_bin[:min(Nbin,len(IDs_bin)-1)])  # Store the ids of the selected stars
        return

    def saveSurvey(self):
        if self.VERBOSE:
            print 'Storing survey data into file.'
        if not os.path.exists(self.outputDir):
            assert os.system('mkdir '+self.outputDir) == 0
        self.outputFile = self.outputDir+'/SurveyOutput.dat'

        f= open(self.outputFile,'wb')
        cPickle.dump(self,f,protocol=-1)
        f.close()

    # =============================================================
    # =============================================================
    # =============================================================
    # PLOTTING METHODS
    # =============================================================

    def loadParameter(self,param):
        if not param in self.Survey.keys():
            if not param == 'hRV':
                tmp = ebfpy.readEBFObjs(self.EBFfilename,objIDs = (param,))
                if tmp == -1:
                    if param == '/metal':
                        tmp = ebfpy.readEBFObjs(self.EBFfilename,
                                                objIDs = ('/feh',))
                        if tmp == -1:
                            print 'No metallicity in Aquarius output file!'
                            print 'Returning.'
                            return -1
                        tmp[param] = tmp['/feh']
                    else:
                        print '%s not found in Aquarius output file!'%param
                        print 'Returning.'
                        return -1
                if param == '/teff':
                    self.Survey[param] = 10.**tmp[param][self.Survey['IDs']]
                else:
                    self.Survey[param] = tmp[param][self.Survey['IDs']]
                del tmp
            else:
                ebfpy.addHeliocentricRV(self.Survey)
        return

    def loadParameters(self,params):
        getList = []
        for p in params:
            if p not in self.Survey.keys():
                getList.append(p)
        haveList = ebfpy.getEBFObjIDs(self.EBFfilename)
        if '/metal' in getList:
            if '/metal' not in haveList:
                getList.remove('/metal')
                getList.append('/feh')
        for p in getList:
            if p not in haveList:
                print '%s is not available for this survey file'%p
                getList.remove(p)
        tmp = ebfpy.readEBFObjs(self.EBFfilename,objIDs = getList)
        for p in getList:
            if p == '/teff':
                self.Survey[p] = 10.**tmp[p][self.Survey['IDs']]
            elif p == '/feh':
                self.Survey['/metal'] = tmp[p][self.Survey['IDs']]
            else:
                self.Survey[p] = tmp[p][self.Survey['IDs']]
        del tmp
        return

    def loadApparentMagnitude(self,mag):
        UBV_Schlegel_factors = {
            'U' : 5.434,
            'B' : 4.315,
            'V' : 3.315,
            'R' : 2.673,
            'I' : 1.940,
            'J' : 0.902,
            'H' : 0.576,
            'K' : 0.367,
            }
        if mag not in UBV_Schlegel_factors.keys():
            print '"%s" is not in the UBV filter system. Returning.'%mag
            return -1
        tag = mag+'mag'
        if not tag in self.Survey.keys():
            tmp = ebfpy.readEBFObjs(self.GalaxiaOutputFile,
                                    objIDs = ('/UBV_%s'%(mag),))
            absMag = tmp['/UBV_%s'%mag][self.Survey['IDs']]
            tmp = absMag + \
                  5.*log10(self.Survey['/rad']*100.) + \
                  UBV_Schlegel_factors[mag] * self.Survey['/ExBV_Schlegel']
            self.Survey[tag] = tmp
            self.Survey['/UBV_%s'%mag] = absMag
            del tmp
        return

def load_Survey(OutputFolder):
    f = open('%s/SurveyOutput.dat'%OutputFolder,'rb')
    d = cPickle.load(f)
    f.close()
    print '%s loaded'%OutputFolder
    return d

def introduceObservationalErrors(d,RAVE_sample =['all'],seed=12,\
                                 vel_errors=True,\
                                 distance_error=True,\
                                 external_errors=True, f_ext=1.,\
                                 include_correlations=True):
    random.seed(seed)
    UDF = random.normal
    if RAVE_sample[0] == 'all':
        RAVE_sample = ones(len(d.RAVE['/hrv']),dtype=bool)
    
    # Output
    e = {'seed':seed,\
         'l':d.Survey['l'],\
         'b':d.Survey['b']}
    
    # Sort RAVE data in uncertainty regions
    region = {}
    region['Aa'] = (d.RAVE['/logg_k']>=3.5)&(d.RAVE['/teff_k']>6000)&(d.RAVE['/met_n_k'] < -0.7)&RAVE_sample
    region['Ab'] = (d.RAVE['/logg_k']>=3.5)&(d.RAVE['/teff_k']>6000)&\
                   (d.RAVE['/met_n_k'] >= -0.7)&(d.RAVE['/met_n_k'] < -0.25)&RAVE_sample
    region['Ac'] = (d.RAVE['/logg_k']>=3.5)&(d.RAVE['/teff_k']>6000)&(d.RAVE['/met_n_k'] >= -0.25)&RAVE_sample

    region['Ba'] = (d.RAVE['/logg_k']<3.5)&(d.RAVE['/teff_k']>6000)&(d.RAVE['/met_n_k'] < -0.7)&RAVE_sample
    region['Bb'] = (d.RAVE['/logg_k']<3.5)&(d.RAVE['/teff_k']>6000)&(d.RAVE['/met_n_k'] >= -0.7)&RAVE_sample

    region['Ca'] = (d.RAVE['/logg_k']>=3.5)&(d.RAVE['/teff_k']<=6000)&\
                   (d.RAVE['/teff_k']>5000)&(d.RAVE['/met_n_k'] < -0.7)&RAVE_sample
    region['Cb'] = (d.RAVE['/logg_k']>=3.5)&(d.RAVE['/teff_k']<=6000)&\
                   (d.RAVE['/teff_k']>5000)&(d.RAVE['/met_n_k'] >= -0.7)&(d.RAVE['/met_n_k'] < -0.25)&RAVE_sample
    region['Cc'] = (d.RAVE['/logg_k']>=3.5)&(d.RAVE['/teff_k']<=6000)&\
                   (d.RAVE['/teff_k']>5000)&(d.RAVE['/met_n_k'] >= -0.25)&RAVE_sample

    region['Da'] = (d.RAVE['/logg_k']< 3.5)&(d.RAVE['/teff_k']<=6000)&\
                   (d.RAVE['/teff_k']>5000)&(d.RAVE['/met_n_k'] < -0.7)&RAVE_sample
    region['Db'] = (d.RAVE['/logg_k']< 3.5)&(d.RAVE['/teff_k']<=6000)&\
                   (d.RAVE['/teff_k']>5000)&(d.RAVE['/met_n_k'] >= -0.7)&RAVE_sample

    region['Ea'] = (d.RAVE['/logg_k']<3.5)&(d.RAVE['/teff_k']<=5000)&(d.RAVE['/met_n_k'] < -0.7)&\
                   (d.RAVE['/logg_k']>= (d.RAVE['/teff_k']-3800)/400)&RAVE_sample
    region['Eb'] = (d.RAVE['/logg_k']<3.5)&(d.RAVE['/teff_k']<=5000)&(d.RAVE['/met_n_k'] >= -0.7)&\
                   (d.RAVE['/logg_k']>= (d.RAVE['/teff_k']-3800)/400)&RAVE_sample

    region['Fa'] = (d.RAVE['/logg_k']<3.5)&(d.RAVE['/teff_k']<=5000)&(d.RAVE['/met_n_k'] < -0.7)&\
                   (d.RAVE['/logg_k']< (d.RAVE['/teff_k']-3800)/400)&RAVE_sample
    region['Fb'] = (d.RAVE['/logg_k']<3.5)&(d.RAVE['/teff_k']<=5000)&(d.RAVE['/met_n_k'] >= -0.7)&\
                   (d.RAVE['/logg_k']< (d.RAVE['/teff_k']-3800)/400)&RAVE_sample

    region['Ga'] = (d.RAVE['/logg_k']>=3.5)&(d.RAVE['/teff_k']>6000)&(d.RAVE['/met_n_k'] < -0.7)&RAVE_sample
    region['Gb'] = (d.RAVE['/logg_k']>=3.5)&(d.RAVE['/teff_k']>6000)&\
                   (d.RAVE['/met_n_k'] >= -0.7)&(d.RAVE['/met_n_k'] < -0.25)&RAVE_sample
    region['Gc'] = (d.RAVE['/logg_k']>=3.5)&(d.RAVE['/teff_k']>6000)&(d.RAVE['/met_n_k'] >= -0.25)&RAVE_sample

    # Sort model data in uncertainty regions
    region_m = {}
    region_m['Aa'] = (d.Survey['/grav']>=3.5)&(d.Survey['/teff']>6000)&(d.Survey['/metal'] < -0.7)
    region_m['Ab'] = (d.Survey['/grav']>=3.5)&(d.Survey['/teff']>6000)&\
                     (d.Survey['/metal'] >= -0.7)&(d.Survey['/metal'] < -0.25)
    region_m['Ac'] = (d.Survey['/grav']>=3.5)&(d.Survey['/teff']>6000)&(d.Survey['/metal'] >= -0.25)

    region_m['Ba'] = (d.Survey['/grav']<3.5)&(d.Survey['/teff']>6000)&(d.Survey['/metal'] < -0.7)
    region_m['Bb'] = (d.Survey['/grav']<3.5)&(d.Survey['/teff']>6000)&(d.Survey['/metal'] >= -0.7)

    region_m['Ca'] = (d.Survey['/grav']>=3.5)&(d.Survey['/teff']<=6000)&\
                     (d.Survey['/teff']>5000)&(d.Survey['/metal'] < -0.7)
    region_m['Cb'] = (d.Survey['/grav']>=3.5)&(d.Survey['/teff']<=6000)&\
                     (d.Survey['/teff']>5000)&(d.Survey['/metal'] >= -0.7)&(d.Survey['/metal'] < -0.25)
    region_m['Cc'] = (d.Survey['/grav']>=3.5)&(d.Survey['/teff']<=6000)&\
                     (d.Survey['/teff']>5000)&(d.Survey['/metal'] >= -0.25)

    region_m['Da'] = (d.Survey['/grav']<3.5)&(d.Survey['/teff']<=6000)&\
                     (d.Survey['/teff']>5000)&(d.Survey['/metal'] < -0.7)
    region_m['Db'] = (d.Survey['/grav']<3.5)&(d.Survey['/teff']<=6000)&\
                     (d.Survey['/teff']>5000)&(d.Survey['/metal'] >= -0.7)

    region_m['Ea'] = (d.Survey['/grav']<3.5)&(d.Survey['/teff']<=5000)&\
                     (d.Survey['/metal'] < -0.7)&(d.Survey['/grav']>= (d.Survey['/teff']-3800)/400)
    region_m['Eb'] = (d.Survey['/grav']<3.5)&(d.Survey['/teff']<=5000)&\
                     (d.Survey['/metal'] >= -0.7)&(d.Survey['/grav']>= (d.Survey['/teff']-3800)/400)
    region_m['Fa'] = (d.Survey['/grav']<3.5)&(d.Survey['/teff']<=5000)&\
                     (d.Survey['/metal'] < -0.7)&(d.Survey['/grav']< (d.Survey['/teff']-3800)/400)
    region_m['Fb'] = (d.Survey['/grav']<3.5)&(d.Survey['/teff']<=5000)&\
                     (d.Survey['/metal'] >= -0.7)&(d.Survey['/grav']< (d.Survey['/teff']-3800)/400)
    
    region_m['Ga'] = (d.Survey['/grav']>=3.5)&(d.Survey['/teff']<=5000)&(d.Survey['/metal'] < -0.7)
    region_m['Gb'] = (d.Survey['/grav']>=3.5)&(d.Survey['/teff']<=5000)&\
                     (d.Survey['/metal'] >= -0.7)&(d.Survey['/metal'] < -0.25)
    region_m['Gc'] = (d.Survey['/grav']>=3.5)&(d.Survey['/teff']<=5000)&(d.Survey['/metal'] >= -0.25)

    Ntot = len(d.RAVE['/logg_k'])
    Ntot_m = len(d.Survey['/grav'])
    ids_RAVE = arange(Ntot)
    uncert_id = zeros(Ntot_m,dtype=int)
    for letter in region.keys():
        Nrave = sum(region[letter])
        Nmodel = sum(region_m[letter])
        uncert_id[region_m[letter]] = ids_RAVE[region[letter]][random.randint(0,Nrave,Nmodel)]
    elogg_int = d.RAVE['/elogg_k'][uncert_id]
    eteff_int = d.RAVE['/eteff_k'][uncert_id]
    emet_int  = d.RAVE['/emet_k'][uncert_id]

    e['elogg_int'] = elogg_int
    e['eteff_int'] = eteff_int
    e['emet_int'] = emet_int

    if vel_errors:
        epmRA = d.RAVE['/epmra'][uncert_id]
        epmDE = d.RAVE['/epmde'][uncert_id]
        e['ehRV'] = d.RAVE['/ehrv'][uncert_id]
        # Attach other e_pm if no e_pm is available for a RAVE star
        wi_pm = isfinite(epmRA)
        no_pm = isfinite(epmRA)==False
        N_wi_pm,N_no_pm = sum(wi_pm),sum(no_pm)
        tmp = random.randint(0,N_wi_pm,N_no_pm)
        epmRA[no_pm] = epmRA[wi_pm][tmp]
        epmDE[no_pm] = epmDE[wi_pm][tmp]
        e['epmRA'] =  epmRA
        e['epmDE'] =  epmDE
        for tag in ['hRV','pmRA','pmDE']:
            e['e'+tag][e['e'+tag]<=0] = 1e-9
            e[tag] = UDF(d.Survey[tag],e['e'+tag])
    else:
        e['hRV'] = array(d.Survey['hRV']).reshape(-1)
        e['pmRA'] = array(d.Survey['pmRA']).reshape(-1)
        e['pmDE'] = array(d.Survey['pmDE']).reshape(-1)
        

    elogg_ext,eteff_ext,emet_ext = zeros(Ntot_m),zeros(Ntot_m),zeros(Ntot_m)
    if external_errors:
        # External uncertainties from DR4 paper, table 3
        # Dwarfs
        eteff_ext[(d.Survey['/grav']>3.5)&(d.Survey['/teff']>6000)&(d.Survey['/metal']<-0.5)] = 314.*f_ext
        elogg_ext[(d.Survey['/grav']>3.5)&(d.Survey['/teff']>6000)&(d.Survey['/metal']<-0.5)] = 0.466*f_ext
        emet_ext[(d.Survey['/grav']>3.5)&(d.Survey['/teff']>6000)&(d.Survey['/metal']<-0.5)]  = 0.269*f_ext

        eteff_ext[(d.Survey['/grav']>3.5)&(d.Survey['/teff']>6000)&(d.Survey['/metal']>=-0.5)] = 173.*f_ext
        elogg_ext[(d.Survey['/grav']>3.5)&(d.Survey['/teff']>6000)&(d.Survey['/metal']>=-0.5)] = 0.276*f_ext
        emet_ext[(d.Survey['/grav']>3.5)&(d.Survey['/teff']>6000)&(d.Survey['/metal']>=-0.5)]  = 0.119*f_ext

        eteff_ext[(d.Survey['/grav']>3.5)&(d.Survey['/teff']<=6000)&(d.Survey['/metal']<-0.5)] = 253.*f_ext
        elogg_ext[(d.Survey['/grav']>3.5)&(d.Survey['/teff']<=6000)&(d.Survey['/metal']<-0.5)] = 0.470*f_ext
        emet_ext[(d.Survey['/grav']>3.5)&(d.Survey['/teff']<=6000)&(d.Survey['/metal']<-0.5)]  = 0.197*f_ext

        eteff_ext[(d.Survey['/grav']>3.5)&(d.Survey['/teff']<=6000)&(d.Survey['/metal']>=-0.5)] = 145.*f_ext
        elogg_ext[(d.Survey['/grav']>3.5)&(d.Survey['/teff']<=6000)&(d.Survey['/metal']>=-0.5)] = 0.384*f_ext
        emet_ext[(d.Survey['/grav']>3.5)&(d.Survey['/teff']<=6000)&(d.Survey['/metal']>=-0.5)]  = 0.111*f_ext
        # Giants
        eteff_ext[(d.Survey['/grav']<=3.5)&(d.Survey['/teff']>6000)] = 263.*f_ext
        elogg_ext[(d.Survey['/grav']<=3.5)&(d.Survey['/teff']>6000)] = 0.423*f_ext
        emet_ext[(d.Survey['/grav']<=3.5)&(d.Survey['/teff']>6000)]  = 0.300*f_ext

        eteff_ext[(d.Survey['/grav']<=3.5)&(d.Survey['/teff']<=6000)&(d.Survey['/metal']<-0.5)] = 191.*f_ext
        elogg_ext[(d.Survey['/grav']<=3.5)&(d.Survey['/teff']<=6000)&(d.Survey['/metal']<-0.5)] = 0.725*f_ext
        emet_ext[(d.Survey['/grav']<=3.5)&(d.Survey['/teff']<=6000)&(d.Survey['/metal']<-0.5)]  = 0.217*f_ext

        eteff_ext[(d.Survey['/grav']<=3.5)&(d.Survey['/teff']<=6000)&(d.Survey['/metal']>=-0.5)] = 89.*f_ext
        elogg_ext[(d.Survey['/grav']<=3.5)&(d.Survey['/teff']<=6000)&(d.Survey['/metal']>=-0.5)] = 0.605*f_ext
        emet_ext[(d.Survey['/grav']<=3.5)&(d.Survey['/teff']<=6000)&(d.Survey['/metal']>=-0.5)]  = 0.144*f_ext
    
    fTeff_g,fTeff_d,fmet_g,fmet_d = 1.,1.,1.,1.
    if include_correlations:
        # Adding correlations to the errors:
        dTeff_dlogg_g = 122. # K per dex
        dTeff_dlogg_d = 531. # K per dex
        dMet_dlogg = 0.15
        # Removing corrlations from random errors
        fTeff_g, fmet_g = 0.866, 0.756
        fTeff_d, fmet_d = 0.674, 0.928

        eteff_int[d.Survey['/grav']< 3.5] *= fTeff_g
        eteff_int[d.Survey['/grav']>=3.5] *= fTeff_d
        emet_int[d.Survey['/grav']< 3.5] *= fmet_g
        emet_int[d.Survey['/grav']>=3.5] *= fmet_d
        
    e['elogg'] = sqrt(elogg_int**2 + elogg_ext**2)
    e['eteff'] = sqrt(eteff_int**2 + eteff_ext**2)
    e['emet']  = sqrt(emet_int**2 + emet_ext**2)

    e['logg']  = UDF(d.Survey['/grav'],e['elogg'])
    e['teff']  = UDF(d.Survey['/teff'],e['eteff'])
    e['metal'] = UDF(d.Survey['/metal'],e['emet'])

    dteff = e['teff'] - d.Survey['/teff']
    dmet  = e['metal'] - d.Survey['/metal']

    if include_correlations:
        dlogg = e['logg'] - d.Survey['/grav']
        e['teff'][d.Survey['/grav']< 3.5] += dTeff_dlogg_g * dlogg[d.Survey['/grav']< 3.5]
        e['teff'][d.Survey['/grav']>=3.5] += dTeff_dlogg_d * dlogg[d.Survey['/grav']>=3.5]
        e['metal'] += dMet_dlogg * dlogg

    # Biases visible in DR4 paper, Figure 10
    #logg_bias = interp1d(array([-1.,1.5,2.5,3.5,4.5,6.]),array([-0.74,-0.42,0.16,-0.21,0.18,0.]))
    #model_logg[(d.Survey['/grav']<1)&(d.Survey['/grav']>0)] += -0.42 # -0.76
    #model_logg[(d.Survey['/grav']<2)&(d.Survey['/grav']>1)] += -0.42
    #model_logg[(d.Survey['/grav']<3)&(d.Survey['/grav']>2)] +=  0.16
    #model_logg[(d.Survey['/grav']<4)&(d.Survey['/grav']>3)] += -0.21
    #model_logg[(d.Survey['/grav']<5)&(d.Survey['/grav']>4)] += 0.18

    if distance_error:
        wDist = RAVE_sample & isfinite(d.RAVE['/parallax'])
        Ndist = sum(wDist)
        tmp_logg = d.RAVE['/logg_k'][wDist]
        tmp_teff = d.RAVE['/teff_k'][wDist]
        tmp_epara = (d.RAVE['/e_parallax']/d.RAVE['/parallax'])[wDist]

        t,g = arange(3800,8000,500),arange(-1,6,0.25)
        R_ids = arange(Ndist,dtype=int)
        duncert_id = zeros(Ntot_m,dtype=int)
        idpix_R = PyTil.xy2pix(t,g,tmp_teff,tmp_logg)
        #idpix_M = PyTil.xy2pix(t,g,d.Survey['/teff'],d.Survey['/grav'])
        idpix_M = PyTil.xy2pix(t,g,e['teff'],e['logg'])
        Npix = PyTil.getNpix(t,g)
        pixHist_R = histogram(idpix_R,arange(-0.1,Npix))[0]
        pixHist_M = histogram(idpix_M,arange(-0.1,Npix))[0]
        for i in xrange(Npix):
            ij,inc = i,1
            if pixHist_M[i] == 0:
                continue
            if pixHist_R[i] == 0:
                while pixHist_R[ij] == 0:
                    if ij < 0 or ij > Npix-2:
                        inc *= -1
                        ij = i
                    if i<Npix:
                        ij += inc
            tmp = random.randint(0,pixHist_R[ij],pixHist_M[i])
            duncert_id[idpix_M==i] = R_ids[idpix_R==ij][tmp]

        e['ef_parallax'] = tmp_epara[duncert_id]
        if include_correlations:
            f_para = 0.671 # equal for dwarfs and giants!
            e['ef_parallax'] *= f_para
        e['parallax'] = UDF(1./d.Survey['/rad'],e['ef_parallax']/d.Survey['/rad'])

        if include_correlations:
            pdlogg = 0.272
            e['parallax'] *= 1. + pdlogg*dlogg
    else:
        e['parallax'] = 1./d.Survey['/rad']
    e['distance'] = 1./e['parallax']
    if distance_error:
        e['distance'][e['ef_parallax']<=0.] == nan
    
    return e

def SchegelExtinctionCorrection(exBV):
    ''' According to Sharma, Bland-Hawthorn et. al (2013)'''
    return (0.2*(1. - tanh((exBV-0.15)/0.3)) + 0.6)*exBV



if __name__ == '__main__':
    fold = ''
#    sim,rel = '../MockRAVE/AquariusC_switched_prunedSph_0.05',False
#    sim,rel = '/z/til/MockSurveyProject/Aquarius2Galaxia/MockRAVE/AquariusC_switched',False
#    sim,rel = '/z/til/MockSurveyProject/Aquarius2Galaxia/MockRAVE/originalGalaxia_newThickDisk',False
#    sim,rel = '../TestSurveyRoutine/TestDiskGalaxy',False
#    sim,rel = '../TestSurveyRoutine/TestDiskGalaxy_LR',False
#    sim,rel = '../TestSurveyRoutine/TestDiskGalaxy_HR',False
#    sim,rel = '../TestSurveyRoutine/TestDiskGalaxy_z0.5',False

#    sim,rel = '/z/til/testsNEMO/galaxy_default/NEMO-galaxy_default',False
#    sim,rel = '/z/til/MockSurveyProject/A_Good_Nbody_MW_model/oldAttemps/GoodMWmodel0',False
#    sim,rel = '/z/til/MockSurveyProject/A_Good_Nbody_MW_model/GoodMWmodel1',False
#    sim,rel = '/z/til/MockSurveyProject/A_Good_Nbody_MW_model/GoodMWmodel2',False
#    sim,rel = '/z/til/MockSurveyProject/A_Good_Nbody_MW_model/JoBovy-6Disks/JoBovy-6Disks',False
#    sim,rel = '/z/til/MockSurveyProject/A_Good_Nbody_MW_model/Attempt_with_MakeDiskGalaxy/AttemptMakeDiskGalaxy6/AttemptMakeDiskGalaxy6',False

    # Original Galaxia model
    #sim,fname,rel = '/z/til/MockSurveyProject/MockRAVESurvey/originalGalaxiaModel','/originalGALAXIA.ebf',False

    # Chemo-dynamical model (Minchev+2013)
    #sim,fname,rel = '/z/til/ChemoDynModel/GalaxiaOutput/imf_Scalo86_IsochroneCorrection_mass5/ChemoDynModel_M5_smoothing_from_random_azimuth','.ebf',False
    #sim,fname,rel = '/z/til/ChemoDynModel/GalaxiaOutput/imf_Scalo86_IsochroneCorrection_mass10','/ChemoDynModel_M5_smoothing_from_random_azimuth.ebf',False
    #sim,fname,rel = '/z/til/ChemoDynModel/GalaxiaOutput/imf_Scalo86_mass5/ChemoDynModel_M5_smoothing_from_random_azimuth','.ebf',False
    #sim,fname,rel = '/z/til/ChemoDynModel/GalaxiaOutput/imf_Chabrier_mass5','/ChemoDynModel_M5_smoothing_from_random_azimuth.ebf',False
    #sim,fname,rel = '/z/til/ChemoDynModel/GalaxiaOutput/SmoothingTests','/ChDM_M5_fakeSmoothing.ebf',False
    #sim,fname,rel = '/z/til/ChemoDynModel/GalaxiaOutput/SmoothingTests_d3n64','/ChDM_M5_fakeSmoothing64.ebf',False
    #sim,fname,rel = '/z/til/ChemoDynModel/GalaxiaOutput/imf_Scalo86_PARSEC-Isochrones','/ChemoDynModel_M5_smoothing_from_random_azimuth.ebf',False
    #sim,fname,fold,rel = '/z/til/ChemoDynModel/GalaxiaOutput/stacked2/Scalo86_IsoCorr','/ChDM_M10_stacked2.ebf','/6D',False
    sim,fname,fold,rel = '/z/til/ChemoDynModel/GalaxiaOutput/stacked2/Scalo86_IsoCorr','/ChDM_M10_stacked2_3D.ebf','/3D',False
    #sim,fname,fold,rel = '/z/til/ChemoDynModel/GalaxiaOutput/stacked2/Scalo86_IsoCorr','/ChDM_M10_stacked2_6D.ebf','/4D',False
    #sim,fname,fold,rel = '/z/til/ChemoDynModel/GalaxiaOutput/stacked2/Scalo86','/ChDM_M10_stacked2_3D.ebf','/3D',False

    # Aquarius Simulations
    #sim,fname,rel = '/z/til/MockSurveyProject/MockRAVESurvey/AquilaSimulations/GalaxiaOutput/AquariusC/','AquariusC.ebf',False
    #sim,fname,fold,rel = '/z/til/MockSurveyProject/MockRAVESurvey/AquilaSimulations/GalaxiaOutput/stacked2/AquariusG/','AquariusG_stacked2.ebf','6D/',False
    #sim,fname,fold,rel = '/z/til/MockSurveyProject/MockRAVESurvey/AquilaSimulations/GalaxiaOutput/stacked2/AquariusG/','AquariusG_stacked2_3D.ebf','3D/',False
    #sim,fname,fold,rel = '/z/til/MockSurveyProject/MockRAVESurvey/AquilaSimulations/GalaxiaOutput/stacked2/AquariusG_f4/','AquariusG_stacked2_3D.ebf','3D/',False
    #sim,fname,fold,rel = '/z/til/MockSurveyProject/MockRAVESurvey/AquilaSimulations/GalaxiaOutput/stacked2/noMetalCorr/AquariusG_f4/','AquariusG_stacked2_3D.ebf','3D/',False
    
    metal_correction = True
    if True:
        d = MockRaveSurvey(sim+fname,sim+fold,rel,metal_correction,VERBOSE=False)
    else:
        d = load_Survey(sim)
