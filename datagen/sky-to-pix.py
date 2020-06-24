#!/usr/bin/python3

NUM_IMAGES = 120

import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy import units
from astropy.coordinates import SkyCoord

for i in range(0, NUM_IMAGES):

    inputmap = 'output/%d-sky.fits' %(i)
    skymap = 'input/%d-skymodel.txt' %(i)
    pixmap = 'output/%d-pix.txt' %(i)

    # datamap = np.squeeze(fits.getdata(inputmap)) # units: Jy/beam1

    header = fits.getheader(inputmap)
    map_wcs = wcs.WCS(header)

    with open(skymap, 'r') as sky:
        dat = sky.readlines()
        ra_dec = []
        x_y_Jy = []
        for l in dat:
            if l[0] == ' ':
                items = l.split(' ')
                ra = float(items[1])
                dec = float(items[2])
                jy = float(items[3])

                # create skycoord object and convert to pixel coordinates
                coord = SkyCoord(ra, dec, unit='deg')
                x, y = wcs.utils.skycoord_to_pixel(coord, map_wcs)
                x_y_Jy.append((x, y, jy))

    with open(pixmap, 'w') as pix:
        for s in x_y_Jy:
            pix.write('%.2f,%.2f,%.15f\n' %s)

    print('%s written' %(pixmap))
