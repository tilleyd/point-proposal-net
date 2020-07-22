#!/usr/bin/python3

IMAGE_SIZE = 1024
NUM_SOURCES = 200
NUM_IMAGES = 1
RA_MIN, RA_MAX = (-50, 50)
DEC_MIN, DEC_MAX = (-30, 50)

OFFSET_RA = 1.1 # need to fine tune to fit sources into image
OFFSET_DEC = 0.7

#
# measure bg rms
#

from astropy.io import fits
import numpy as np

try:
    # if there is already a background image, use it
    bg = np.squeeze(fits.getdata('output/0-nosource-image.fits'))
    sigma = np.std(bg)
except Exception:
    print('Warning: no background reference found, using default sigma')
    sigma = 0.000033921602153

#
# create sky models
#

# create an equal number of point sources with flux from 0.33sigma to 10sigma
# with bins of size 0.33
bins = np.arange(1.0/3.0, 10.1, 1.0/3.0)
sources_per_bin = NUM_SOURCES // len(bins)
total_sources = sources_per_bin * len(bins)
if NUM_SOURCES != total_sources:
    print('Warning: the number of sources is not divisible by the bins')
    print('         images will have %d/%d sources' %(total_sources, NUM_SOURCES))

#
# create stimela recipe
#

import stimela

INPUT = 'input'
OUTPUT = 'output'
MSDIR = 'msdir'
MS = 'meerkat.ms'

recipe = stimela.Recipe(name='Make noise image populated with sources',
                        ms_dir=MSDIR,
                        JOB_TYPE='udocker')


for img in range(0, NUM_IMAGES):

	# determine field centre
	centre_ra = np.random.uniform(RA_MIN, RA_MAX)
	centre_dec = np.random.uniform(DEC_MIN, DEC_MAX)

	# create sky model
    with open('input/%d-skymodel.txt' %(img), 'w') as f:
        f.write('#format: ra_d dec_d i\n')
        for flux in bins:
            for i in range(0, sources_per_bin):
                ra = np.random.uniform(centre_ra-OFFSET_RA, centre_ra+OFFSET_RA)
                dec = np.random.uniform(centre_dec-OFFSET_DEC, centre_dec+OFFSET_DEC)
                f.write(' %.2f %.2f %.15f\n' %(ra, dec, flux * sigma))
        f.write('~\n')

# create empty measurement set (only once per set of images)
recipe.add('cab/simms',
           'simms',
           {
               "msname"    : MS,
               "telescope" : "meerkat",
               "synthesis" : 1, # exposure time, 1 hour
               "dtime"     : 60*5, # integration time, 5 minutes
               "freq0"     : '1400MHz', # starting frequency
               "dfreq"     : '2.0MHz', # channel width
               "nchan"     : 10, # number of channels
               "direction" : 'J2000,%.1fdeg,%.1fdeg' %(centre_ra, centre_dec), # telescope pointing target
               "feed"      : 'perfect R L',
               "pol"       : 'RR RL LR LL',
           },
           input=INPUT,
           output=OUTPUT,
           label='simms:: Create empty MS')

for img in range(0, NUM_IMAGES):

    # simulate noise into measurement set
    recipe.add('cab/simulator',
               'simsky',
               {
                   "msname"    : MS,
                   "addnoise"  : True,
                   "sefd"      : 450,
                   "threads"   : 16,
                   "column"    : 'DATA',
               },
               input=INPUT,
               output=OUTPUT,
               label='simsky:: Simulate sky')

    # image the noise
    recipe.add('cab/wsclean',
               'image',
               {
                   "msname"    : MS,
                   "prefix"    : "%d-nosource" %(img), # image prefix
                   "column"    : "DATA",
                   "weight"    : "briggs 1.5",
                   "cellsize"  : 5, # pixel size in arcsec
                   "npix"      : IMAGE_SIZE,
                   "trim"      : 1000,
                   "niter"     : 10000,
               },
               input=INPUT,
               output=OUTPUT,
               label='image:: Image data')

    # add point sources
    recipe.add('cab/tigger_restore',
               'add_sources',
               {
                   "input-image" : '%d-nosource-image.fits:output' %(img), # the noise image
                   "input-skymodel" : '%d-skymodel.txt' %(img), # the input skymodel
                   "output-image" : '%d-sky.fits' %(img), # output image name
                   "force" : True,
               },
               input=INPUT,
               output=OUTPUT,
               label='add_sources%d' %(img))

recipe.run()
