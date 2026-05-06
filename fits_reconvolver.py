import glob
import os
import sys

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.convolution import convolve_fft
from radio_beam import Beam

INPUT_PATTERN = "*.fits"
OUTPUT_DIR    = "."

# Manual beam config, for when header is not written
CURRENT_BEAM = Beam(major=6.0 * u.arcsec,
                    minor=6.0 * u.arcsec,
                    pa=0.0    * u.deg)

TARGET_BEAM  = Beam(major=53.0 * u.arcsec,
                    minor=38.0 * u.arcsec,
                    pa=56.0   * u.deg)

def beam_area_pix2(beam, pixscale):
    """Beam solid angle in pixel²  =  (π / 4ln2) * BMAJ_pix * BMIN_pix"""
    return (np.pi / (4 * np.log(2))
            * (beam.major.to(u.arcsec) / pixscale).decompose().value
            * (beam.minor.to(u.arcsec) / pixscale).decompose().value)

def reconvolve(input_fits, output_fits):
    """Load *input_fits*, reconvolve to TARGET_BEAM, write *output_fits*."""

    # Load fits file
    with fits.open(input_fits, memmap=True) as hdul:
        header = hdul[0].header.copy()
        data   = hdul[0].data.astype(np.float64)
    
    # If data format is [stokes, freq, x, y], then squeeze to [x, y]
    data2d = np.squeeze(data)
    print(f"  Working array shape : {data2d.shape}")
    
    pixscale = abs(header['CDELT1']) * 3600 * u.arcsec
    print(f"  Pixel scale         : {pixscale:.4f}")

    # Use beam from header when available; fall back to hardcoded CURRENT_BEAM
    if all(k in header for k in ('BMAJ', 'BMIN', 'BPA')):
        current_beam = Beam(
            major=header['BMAJ'] * u.deg,
            minor=header['BMIN'] * u.deg,
            pa=header['BPA']    * u.deg,
        )
        print("  Current beam        : read from header")
    else:
        current_beam = CURRENT_BEAM
        print("  Current beam        : WARNING — not in header, using hardcoded value")

    kernel_beam = TARGET_BEAM.deconvolve(current_beam)
    print(f"  Current beam        : {current_beam}")
    print(f"  Target beam         : {TARGET_BEAM}")
    print(f"  Convolution kernel  : {kernel_beam}")
    
    kernel = kernel_beam.as_kernel(pixscale)
    print(f"  Kernel array size   : {kernel.shape}")

    # Rescale flux
    old_area = beam_area_pix2(current_beam, pixscale)
    new_area = beam_area_pix2(TARGET_BEAM,  pixscale)
    print(f"\n  Old beam area : {old_area:.2f} pix²")
    print(f"  New beam area : {new_area:.2f} pix²")
    
    data_jy_pix = data2d / old_area
    
    # FFT
    print(f"\n  Convolving... (this may take a few minutes for {data2d.shape[0]}² pixels)")
    data_conv = convolve_fft(
        data_jy_pix,
        kernel,
        normalize_kernel=True,
        nan_treatment='fill',
        fill_value=0.0,
        allow_huge=True,
        fft_pad=False,
    )
    
    data_out = (data_conv * new_area).reshape(data.shape).astype(np.float32)

    # Update header
    header['BMAJ'] = TARGET_BEAM.major.to(u.deg).value
    header['BMIN'] = TARGET_BEAM.minor.to(u.deg).value
    header['BPA']  = TARGET_BEAM.pa.to(u.deg).value
    header['HISTORY'] = (
        f"Reconvolved to "
        f"{TARGET_BEAM.major.to(u.arcsec):.0f}\" x "
        f"{TARGET_BEAM.minor.to(u.arcsec):.0f}\" "
        f"PA={TARGET_BEAM.pa.to(u.deg):.0f} "
    )

    # Write output
    fits.writeto(output_fits, data_out, header, overwrite=True)
    print(f"\n  Written: {output_fits}")
    print(f"    BMAJ={header['BMAJ']*3600:.1f}\"  BMIN={header['BMIN']*3600:.1f}\"  BPA={header['BPA']:.1f}°")

matched = sorted(glob.glob(INPUT_PATTERN))

if not matched:
    print(f"ERROR: No files found:\n  {INPUT_PATTERN}", file=sys.stderr)
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, input_path in enumerate(matched, start=1):
    output_name = (
        f"{input_path}_reconvolved_"
        f"{int(TARGET_BEAM.major.to(u.arcsec).value)}x"
        f"{int(TARGET_BEAM.minor.to(u.arcsec).value)}_"
        f"pa{int(TARGET_BEAM.pa.to(u.deg).value)}.fits"
    )
    output_path = os.path.join(OUTPUT_DIR, output_name)

    print(f"\n=== Processing ({i}/{len(matched)}): {input_path} ===")
    reconvolve(input_path, output_path)
