import astropy
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, ICRS, AltAz
from astropy.time import Time

CSBF_LOCATION = EarthLocation(lat=31.779690 * u.deg, lon=-95.716 * u.deg)


def ra_dec_to_alt_az(ra, dec, time, location=CSBF_LOCATION):
    radec_corner = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    altaz_corner = radec_corner.transform_to(AltAz(obstime=time, location=location))
    return altaz_corner
