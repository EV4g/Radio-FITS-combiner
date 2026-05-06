from astropy.io import fits

INPUT_FITS = "image.fits"

BMAJ_ARCSEC = 6.0   # major axis  [arcsec]
BMIN_ARCSEC = 6.0   # minor axis  [arcsec]
BPA_DEG     = 0.0   # position angle [deg]

with fits.open(INPUT_FITS, mode="update") as hdul:
    hdr = hdul[0].header

    # if anything is already there, print it
    for key in ("BMAJ", "BMIN", "BPA"):
        if key in hdr: print(f"  overwriting  {key} = {hdr[key]}")
        else:          print(f"  adding       {key}  (was absent)")

    hdr["BMAJ"] = (BMAJ_ARCSEC / 3600.0, "[deg] Restoring beam major axis")
    hdr["BMIN"] = (BMIN_ARCSEC / 3600.0, "[deg] Restoring beam minor axis")
    hdr["BPA"]  = (BPA_DEG,              "[deg] Restoring beam position angle")

    hdul.flush()

print(f"\nDone.  Written to {INPUT_FITS}.")
print(f"  BMAJ = {BMAJ_ARCSEC:.2f}\"  BMIN = {BMIN_ARCSEC:.2f}\"  BPA = {BPA_DEG:.1f}°")
