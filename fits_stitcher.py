import glob
import os
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy import units as u
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd

# Inputs / outputs
INPUT_PATTERN = "*.fits"
OUTPUT_FITS   = "reconvolved.fits"
SCRATCH_DIR   = "."

# Per-pixel Gaussian primary-beam weight (degrees FWHM).
# LOFAR HBA core: ~3-4 deg.  LBA is larger.
BEAM_FWHM_DEG = 3.5
WEIGHT_FLOOR  = 0.05

# Cutout region of interest. Applied after mosaicking.
MAKE_CUTOUT     = True
CUTOUT_CENTER   = (312.75, 30.67)   # (RA, Dec)
CUTOUT_SIZE_DEG = (5.0, 5.0)        # (height, width) in deg

# One FITS per input pointing showing just that pointing's contribution.
SAVE_PER_POINTING_CUTOUTS = True

# Crop each input to the cutout's bounding box + this much padding.
# Pad must be larger than ~2x the restored beam in pixels.
# At 6"/pix and 53" restored beam, 150 px is comfortable.
PRECROP_TO_CUTOUT = True
PRECROP_PAD_PIX   = 150

# Number of threads for reproject's parallel interpolation.
N_WORKERS = 16

# Recast all inputs to float32. Set to None to ignore.
FORCE_DTYPE = np.float32

# .fits file HDU
HDU_INDEX = 0


def parse_cutout_center():
    """Return CUTOUT_CENTER as a SkyCoord."""
    if isinstance(CUTOUT_CENTER, str):
        return SkyCoord(CUTOUT_CENTER, unit=(u.hourangle, u.deg))
    return SkyCoord(CUTOUT_CENTER[0] * u.deg, CUTOUT_CENTER[1] * u.deg, frame="icrs")


def derive_name(path, all_paths):
    """Short unique identifier for `path`, used for printouts and cache files.

    Prefers the parent dir name when those are distinct across `all_paths`
    (the P312+31/ subfolder layout). Falls back to the filename stem when
    parent dirs aren't unique (single dir with `*.fits`)."""
    if len(all_paths) > 1:
        parents = [os.path.basename(os.path.dirname(os.path.abspath(p))) for p in all_paths]
        if all(parents) and len(set(parents)) == len(parents):
            return parents[all_paths.index(path)]
    return os.path.splitext(os.path.basename(path))[0]


def squeeze_to_2d(data, wcs):
    """Drop dummy STOKES/FREQ axes"""
    while data.ndim > 2:
        if data.shape[0] != 1:
            raise ValueError(
                f"Real data cube of shape {data.shape}; "
                f"expected leading axes to be size 1.")
        data = data[0]
    return data, wcs.celestial


def beam_center_from_header(header, wcs, shape):
    """Pointing center from the header. Falls back to image center."""
    for ra_key, dec_key in [("CRVAL1", "CRVAL2"),
                            ("OBSRA",  "OBSDEC"),
                            ("RA",     "DEC")]:
        if ra_key in header and dec_key in header:
            try:
                return SkyCoord(header[ra_key] * u.deg, header[dec_key] * u.deg, frame="icrs")
            except Exception:
                pass
    return wcs.pixel_to_world(shape[1] / 2, shape[0] / 2)


def fill_gaussian_weight(out, wcs, beam_center, fwhm_deg):
    """Fill `out` with per-pixel Gaussian weight about beam_center."""
    ny, nx = out.shape
    sigma = fwhm_deg / (2 * np.sqrt(2 * np.log(2)))
    ra0  = np.deg2rad(beam_center.ra.deg)
    dec0 = np.deg2rad(beam_center.dec.deg)

    chunk = 4096
    x = np.arange(nx, dtype=np.float64)
    for y0 in range(0, ny, chunk):
        y1 = min(y0 + chunk, ny)
        ys = np.arange(y0, y1, dtype=np.float64)
        xx, yy = np.meshgrid(x, ys)
        ra, dec = wcs.all_pix2world(xx, yy, 0)
        ra  = np.deg2rad(ra,  out=ra)
        dec = np.deg2rad(dec, out=dec)
        a = (np.sin(0.5 * (dec - dec0))**2 + np.cos(dec0) * np.cos(dec) * np.sin(0.5 * (ra - ra0))**2)
        sep_deg = np.rad2deg(2 * np.arcsin(np.sqrt(np.minimum(a, 1.0))))
        out[y0:y1] = np.exp(-0.5 * (sep_deg / sigma)**2).astype(out.dtype, copy=False)


def precrop_bounds(full_shape, full_wcs, pad):
    """Pixel bbox in `full_wcs` covering the cutout region. None if no overlap."""
    ny, nx = full_shape
    center = parse_cutout_center()
    half_h = CUTOUT_SIZE_DEG[0] / 2
    half_w = CUTOUT_SIZE_DEG[1] / 2
    ra_scale = 1.0 / max(np.cos(np.deg2rad(center.dec.deg)), 0.05)

    ts = np.linspace(-1, 1, 200)
    ra_pts = np.concatenate([
        center.ra.deg + ts * half_w * ra_scale,
        center.ra.deg + ts * half_w * ra_scale,
        np.full(200, center.ra.deg + half_w * ra_scale),
        np.full(200, center.ra.deg - half_w * ra_scale),
    ])
    dec_pts = np.concatenate([
        np.full(200, center.dec.deg + half_h),
        np.full(200, center.dec.deg - half_h),
        center.dec.deg + ts * half_h,
        center.dec.deg + ts * half_h,
    ])

    xs, ys = full_wcs.world_to_pixel_values(ra_pts, dec_pts)
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    if not xs.size or not ys.size:
        return None

    x0 = max(int(np.floor(xs.min())) - pad, 0)
    y0 = max(int(np.floor(ys.min())) - pad, 0)
    x1 = min(int(np.ceil(xs.max()))  + pad, nx)
    y1 = min(int(np.ceil(ys.max()))  + pad, ny)
    if x1 <= x0 or y1 <= y0:
        return None

    new_wcs = full_wcs.deepcopy()
    new_wcs.wcs.crpix[0] -= x0
    new_wcs.wcs.crpix[1] -= y0
    return slice(y0, y1), slice(x0, x1), new_wcs


def write_npy_atomic(path, data, dtype):
    """Stream-copy `data` to a .npy memmap at `path` via .tmp + atomic rename.
    A crashed run thus leaves either a complete file or nothing at all."""
    tmp = path + ".tmp"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=data.shape)
    chunk = 2048
    for y0 in range(0, data.shape[0], chunk):
        y1 = min(y0 + chunk, data.shape[0])
        out[y0:y1] = data[y0:y1].astype(dtype, copy=False)
    out.flush()
    del out
    os.replace(tmp, path)


def load_input(input_fits, name, scratch_dir):
    """Open one input. Optionally precrop. Cache cropped data + weight map
    to scratch as .npy memmaps. Returns dict, or None if no overlap.

    Cache files are keyed by name + crop bounds + beam FWHM.
    Manually delete files in SCRATCH_DIR if you change FORCE_DTYPE."""
    print(f"  opening {name}")

    hdul = fits.open(input_fits, memmap=True)
    hdu  = hdul[HDU_INDEX]
    data, wcs = squeeze_to_2d(hdu.data, WCS(hdu.header))
    full_shape = data.shape

    # Precrop to cutout bbox
    crop_tag = ""
    if MAKE_CUTOUT and PRECROP_TO_CUTOUT:
        bounds = precrop_bounds(full_shape, wcs, PRECROP_PAD_PIX)
        if bounds is None:
            print("    no overlap with cutout; skipping")
            hdul.close()
            return None
        sy, sx, wcs = bounds
        crop_h = sy.stop - sy.start
        crop_w = sx.stop - sx.start
        print(f"    crop : {full_shape} -> ({crop_h}, {crop_w})")
        crop_tag = f"_crop{sy.start}-{sy.stop}_{sx.start}-{sx.stop}"
        data = data[sy, sx]

    # Beam center while header is still loaded
    beam_center = beam_center_from_header(hdu.header, wcs, data.shape)
    bmaj = hdu.header.get("BMAJ")  # degrees
    bmin = hdu.header.get("BMIN")  # degrees
    bpa  = hdu.header.get("BPA")   # degrees

    # Cache cropped, native-endian data
    target_dtype = np.dtype(FORCE_DTYPE).newbyteorder("=")
    cast_path = os.path.join(scratch_dir, f"{name}{crop_tag}_data.npy")
    if os.path.exists(cast_path):
        print("    data : cached")
    else:
        print(f"    data : writing  shape={data.shape}  dtype={target_dtype}")
        write_npy_atomic(cast_path, data, target_dtype)
    hdul.close()
    data = np.load(cast_path, mmap_mode="r")

    # Cache weight map
    wpath = os.path.join(
        scratch_dir,
        f"{name}{crop_tag}_weight_fwhm{BEAM_FWHM_DEG}.npy")
    if os.path.exists(wpath):
        print("    weight: cached")
    else:
        print("    weight: computing")
        tmp = wpath + ".tmp"
        w = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=data.shape)
        fill_gaussian_weight(w, wcs, beam_center, BEAM_FWHM_DEG)
        if WEIGHT_FLOOR > 0:
            for y0 in range(0, data.shape[0], 4096):
                y1 = min(y0 + 4096, data.shape[0])
                row = w[y0:y1]
                row[row < WEIGHT_FLOOR] = 0.0
        w.flush()
        del w
        os.replace(tmp, wpath)
    weights = np.load(wpath, mmap_mode="r")

    return {
        "name": name, "data": data, "wcs": wcs,
        "weights": weights, "beam_center": beam_center,
        "bmaj": bmaj, "bmin": bmin, "bpa": bpa,
    }


def stitch(matched, output_fits, scratch_dir):
    """Reproject + coadd `matched` into one mosaic. Optional cutouts and
    per-pointing diagnostic outputs are written next to it."""
    
    # Load inputs (precrop + cache)
    print("\n=== Loading inputs ===")
    entries = []
    for path in matched:
        name = derive_name(path, matched)
        e = load_input(path, name, scratch_dir)
        if e is not None:
            entries.append(e)

    if not entries:
        print("ERROR: no inputs overlap the cutout region.", file=sys.stderr)
        sys.exit(1)

    # Collect restoring beam values and verify all inputs agree
    beam_entries = [(e["bmaj"], e["bmin"], e["bpa"]) for e in entries if e["bmaj"] is not None and e["bmin"] is not None]
    if beam_entries:
        beam_out: tuple[float, float, float] | None = beam_entries[0]
        if len(set(beam_entries)) > 1:
            print("  WARNING: input files have inconsistent beam keywords — using first entry's values")
            for e, bv in zip(entries, beam_entries):
                print(f"    {e['name']}: BMAJ={bv[0]*3600:.2f}\"  BMIN={bv[1]*3600:.2f}\"  BPA={bv[2]:.1f}°")
    else:
        beam_out = None
        print("  WARNING: no BMAJ/BMIN/BPA found in any input — beam keywords will not be written")

    # Build output WCS from the (possibly cropped) inputs
    print(f"\n=== Reproject + coadd ({len(entries)} inputs) ===")
    out_wcs, shape_out = find_optimal_celestial_wcs(
        [(e["data"].shape, e["wcs"]) for e in entries])
    print(f"  output shape : {shape_out}")

    mosaic, footprint = reproject_and_coadd(
        [(e["data"], e["wcs"]) for e in entries],
        output_projection=out_wcs,
        shape_out=shape_out,
        input_weights=[e["weights"] for e in entries],
        reproject_function=reproject_interp,
        match_background=False,
        combine_function="mean",
        parallel=N_WORKERS,
    )

    # Cutout
    cutout_data = cutout_wcs = cutout_fp = None
    if MAKE_CUTOUT:
        print("\n=== Applying cutout ===")
        center = parse_cutout_center()
        size = (CUTOUT_SIZE_DEG[0] * u.deg, CUTOUT_SIZE_DEG[1] * u.deg)
        cut = Cutout2D(mosaic,     position=center, size=size, wcs=out_wcs, mode="trim")
        fcut = Cutout2D(footprint, position=center, size=size, wcs=out_wcs, mode="trim")
        cutout_data, cutout_wcs, cutout_fp = cut.data, cut.wcs, fcut.data
        print(f"  cutout shape : {cutout_data.shape}")

    # Per-pointing cutouts
    per_dir = None
    if MAKE_CUTOUT and SAVE_PER_POINTING_CUTOUTS:
        print("\n=== Per-pointing cutouts ===")
        per_dir = output_fits.replace(".fits", "_per_pointing")
        os.makedirs(per_dir, exist_ok=True)
        for e in entries:
            print(f"  {e['name']}")
            arr, fp = reproject_interp(
                (e["data"], e["wcs"]),
                output_projection=cutout_wcs,
                shape_out=cutout_data.shape,
                parallel=N_WORKERS,
                return_footprint=True,
            )
            # Compute weights analytically on the cutout grid: faster and
            # more accurate than reprojecting the input weight map.
            w = np.empty(cutout_data.shape, dtype=np.float32)
            fill_gaussian_weight(w, cutout_wcs, e["beam_center"], BEAM_FWHM_DEG)
            if WEIGHT_FLOOR > 0:
                w[w < WEIGHT_FLOOR] = 0.0

            arr = arr.astype(np.float32, copy=False)
            no_data = (fp == 0) | ~np.isfinite(arr)
            weighted = arr * w
            weighted[no_data] = np.nan

            hdr = cutout_wcs.to_header()
            hdr["POINTING"] = (e["name"], "Pointing name")
            hdr["BEAMFWHM"] = (BEAM_FWHM_DEG, "Gaussian weight FWHM, deg")
            if beam_out is not None:
                bmaj, bmin, bpa = beam_out
                hdr["BMAJ"] = (bmaj, "[deg] Restoring beam major axis")
                hdr["BMIN"] = (bmin, "[deg] Restoring beam minor axis")
                hdr["BPA"]  = (bpa,  "[deg] Restoring beam position angle")
            fits.writeto(os.path.join(per_dir, f"{e['name']}.fits"), weighted, hdr, overwrite=True)

    # Free memmap handles before final writes
    for e in entries: del e["weights"], e["data"]
    del entries

    # Write main mosaic + summed-weights image
    print("\n=== Writing output ===")
    if MAKE_CUTOUT:
        final_data, final_wcs, final_fp = cutout_data, cutout_wcs, cutout_fp
    else:
        final_data, final_wcs, final_fp = mosaic, out_wcs, footprint

    hdr = final_wcs.to_header()
    hdr["BEAMFWHM"] = (BEAM_FWHM_DEG, "Gaussian weight FWHM, deg")
    hdr["NINPUT"]   = (len(matched), "Number of input files co-added")
    if beam_out is not None:
        bmaj, bmin, bpa = beam_out
        hdr["BMAJ"] = (bmaj, "[deg] Restoring beam major axis")
        hdr["BMIN"] = (bmin, "[deg] Restoring beam minor axis")
        hdr["BPA"]  = (bpa,  "[deg] Restoring beam position angle")
    fits.writeto(output_fits, final_data, hdr, overwrite=True)

    weights_path = output_fits.replace(".fits", "_weights.fits")
    fits.writeto(weights_path, final_fp, final_wcs.to_header(), overwrite=True)

    print(f"  mosaic   : {output_fits}")
    print(f"  weights  : {weights_path}")
    if beam_out is not None:
        bmaj, bmin, bpa = beam_out
        print(f"  beam     : BMAJ={bmaj*3600:.2f}\"  BMIN={bmin*3600:.2f}\"  BPA={bpa:.1f}°")
    if per_dir is not None:
        print(f"  per-ptg  : {per_dir}/")


matched = sorted(glob.glob(INPUT_PATTERN))

if not matched:
    print(f"ERROR: No files found:\n  {INPUT_PATTERN}", file=sys.stderr)
    sys.exit(1)

os.makedirs(SCRATCH_DIR, exist_ok=True)

print(f"Found {len(matched)} files matching {INPUT_PATTERN}:")
for p in matched:
    print(f"  {derive_name(p, matched)}")

stitch(matched, OUTPUT_FITS, SCRATCH_DIR)

print("\nDone.")
