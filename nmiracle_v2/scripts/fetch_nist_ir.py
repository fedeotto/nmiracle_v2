import argparse
import requests
import pandas as pd
import numpy as np
import h5py
import json
import time
from pathlib import Path
from scipy.interpolate import interp1d
from tqdm import tqdm

#Constants matching your simulated IR grid
WN_MIN     = 400
WN_MAX     = 4000
RESOLUTION = 2
N_POINTS   = int((WN_MAX - WN_MIN) / RESOLUTION)   # = 1800
WN_GRID    = np.linspace(WN_MIN, WN_MAX, N_POINTS, endpoint=False)

RETRY_WAIT    = 2
MAX_RETRIES   = 3
REQUEST_DELAY = 1.0

#Download JCAMP-DX from NIST
def fetch_jcamp(nist_id: str) -> str | None:
    url = (
        f"https://webbook.nist.gov/cgi/cbook.cgi"
        f"?JCAMP={nist_id}&Index=0&Type=IR"
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                text = resp.text.strip()
                if "##JCAMP-DX" in text or "##XYDATA" in text or "##XYPOINTS" in text:
                    return text
                else:
                    print(f"  ✗ No JCAMP content for {nist_id} (got HTML)")
                    return None
            else:
                print(f"  HTTP {resp.status_code} for {nist_id} (attempt {attempt})")
        except Exception as e:
            print(f"  Request error for {nist_id}: {e} (attempt {attempt})")
        time.sleep(RETRY_WAIT * attempt)
    return None


#Parse JCAMP-DX
def parse_jcamp(jcamp_text: str) -> dict | None:
    x_factor  = 1.0
    y_factor  = 1.0
    deltax    = None
    data_type = None
    yunits    = "TRANSMITTANCE"
    xunits    = "1/CM"              # ← track X units
    x_vals: list[float] = []
    y_vals: list[float] = []
    reading   = False

    for line in jcamp_text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("##XFACTOR="):
            x_factor = float(line.split("=", 1)[1].strip())
        elif line.startswith("##YFACTOR="):
            y_factor = float(line.split("=", 1)[1].strip())
        elif line.startswith("##YUNITS="):
            yunits = line.split("=", 1)[1].strip().upper()
        elif line.startswith("##XUNITS="):
            xunits = line.split("=", 1)[1].strip().upper()   # ← read xunits
        elif line.startswith("##DELTAX="):
            deltax = float(line.split("=", 1)[1].strip())
        elif line.startswith("##XYDATA="):
            data_type = "xydata"
            reading   = True
        elif line.startswith("##XYPOINTS="):
            data_type = "xypoints"
            reading   = True
        elif line.startswith("##END="):
            reading = False
        elif reading:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                if data_type == "xydata":
                    x0 = float(parts[0]) * x_factor
                    for j, y in enumerate(parts[1:]):
                        x_vals.append(x0 + j * (deltax or 0.0))
                        y_vals.append(float(y) * y_factor)
                else:
                    x_vals.append(float(parts[0]) * x_factor)
                    y_vals.append(float(parts[1]) * y_factor)
            except ValueError:
                continue

    if len(x_vals) < 10:
        return None

    # Guard against length mismatch (can occur with malformed JCAMP blocks)
    min_len = min(len(x_vals), len(y_vals))
    x_vals  = x_vals[:min_len]
    y_vals  = y_vals[:min_len]

    wn = np.array(x_vals, dtype=np.float32)

    # ── Convert µm → cm⁻¹ if needed
    if "MICROMETERS" in xunits or "MICRON" in xunits:
        wn = 10000.0 / wn          # inversion also reverses order, that's fine
        tqdm.write(f"    ↳ Converted µm → cm⁻¹  (now {wn.min():.0f}–{wn.max():.0f} cm⁻¹)")
    elif "NANOMETERS" in xunits:
        wn = 1e7 / wn
        tqdm.write(f"    ↳ Converted nm → cm⁻¹  (now {wn.min():.0f}–{wn.max():.0f} cm⁻¹)")
    # "1/CM" or "CM-1" → already correct

    return {
        "wavenumber": wn,
        "intensity":  np.array(y_vals, dtype=np.float32),
        "yunits":     yunits,
        "xunits":     xunits,      # keep for debugging
    }


#Interpolate to the fixed 1800-point grid
def interpolate_to_grid(wavenumbers: np.ndarray,
                         intensities: np.ndarray,
                         yunits: str = "TRANSMITTANCE") -> np.ndarray | None:
    """
    Interpolate spectrum onto the fixed 1800-point grid [400, 4000) @ 2 cm⁻¹.
    Converts transmittance → absorbance to match simulated spectra.
    Normalises to [0, 1]. Returns float32 (1800,) or None.
    """
    # Sort low→high (NIST often stores high→low)
    order = np.argsort(wavenumbers)
    wn    = wavenumbers[order]
    ins   = intensities[order]

    # Check any overlap at all with target range (loose check)
    if wn.max() < WN_MIN or wn.min() > WN_MAX:
        tqdm.write(f"    ✗ No overlap: spectrum range {wn.min():.1f}–{wn.max():.1f} cm⁻¹")
        return None

    # Convert transmittance → absorbance BEFORE interpolating (on the native grid)
    # so that linear interpolation operates in absorbance space (sharper peaks).
    # Fill value of 0.0 = no absorption, which is the correct baseline in absorbance.
    if "TRANSMITTANCE" in yunits:
        ins = np.clip(ins, 1e-6, 1.0)   # guard against log(0)
        ins = -np.log10(ins)
    # If already ABSORBANCE just use as-is

    # Interpolate onto fixed grid — fill outside measured range with 0 (no absorption)
    f   = interp1d(wn, ins, kind="linear", bounds_error=False, fill_value=0.0)
    out = f(WN_GRID).astype(np.float32)

    # Normalise to [0, 1]
    max_val = np.max(np.abs(out))
    if max_val > 0:
        out /= max_val

    return out


#Main pipeline
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("nist_ir_manifest.csv"),
                    help="reference list from scripts/build_nist_manifest.py; its row order "
                         "is the h5 row order that split_indices.p indexes into")
    ap.add_argument("--out_dir", type=Path, default=Path("data/nist_h5"))
    args = ap.parse_args()

    OUT_DIR = args.out_dir
    OUT_H5, OUT_SMILES, OUT_INDEX = (OUT_DIR / "nist_spectra.h5",
                                     OUT_DIR / "nist_smiles.npy",
                                     OUT_DIR / "nist_ir_index.json")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest).sort_values("h5_idx")
    print(f"Loaded {len(df)} compounds from {args.manifest}")
    print(f"Target grid: {N_POINTS} points, {WN_MIN}–{WN_MAX} cm⁻¹ @ {RESOLUTION} cm⁻¹")
    assert N_POINTS == 1800

    index, smiles_out, spectra, failed = [], [], [], []
    splits_out = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching spectra"):
        nist_id = str(row["nist_id"])
        smiles  = str(row["smiles"])
        time.sleep(REQUEST_DELAY)

        #Download
        jcamp_text = fetch_jcamp(nist_id)
        if jcamp_text is None:
            failed.append({"id": nist_id, "reason": "no_jcamp"}); continue

        #Parse
        parsed = parse_jcamp(jcamp_text)
        if parsed is None:
            tqdm.write(f"  ✗ Parse failed for {nist_id}")
            failed.append({"id": nist_id, "reason": "parse_failed"}); continue

        #Interpolate → 1800-point grid (pass yunits through!)
        try:
            spectrum = interpolate_to_grid(
                parsed["wavenumber"],
                parsed["intensity"],
                parsed["yunits"],
            )
        except Exception as e:
            tqdm.write(f"  ✗ Interpolation error for {nist_id}: {e}")
            failed.append({"id": nist_id, "reason": f"interp_error: {e}"}); continue
        if spectrum is None:
            failed.append({"id": nist_id, "reason": "no_overlap"}); continue

        idx = len(spectra)
        spectra.append(spectrum)
        smiles_out.append(smiles)
        # Split assignment travels with the row, not with a position in the manifest: that
        # is what lets a missing record simply drop its molecule instead of shifting every
        # later one into the wrong split.
        splits_out.setdefault(str(row["split"]), []).append(idx)
        index.append({
            "h5_idx":    idx,
            "nist_id":   nist_id,
            "smiles":    smiles,
            "split":     str(row["split"]),
            "formula":   str(row.get("formula", "")),
            "inchi_key": str(row.get("inchi_key", "")),
            "yunits":    parsed["yunits"],
        })
        tqdm.write(f"  ✓ {nist_id}  ({len(parsed['wavenumber'])} raw pts → 1800)  [{parsed['yunits']}]")

    if not spectra:
        print("No spectra collected."); return

    if len(spectra) != len(df):
        missing = ", ".join(f["id"] for f in failed[:5]) + (", …" if len(failed) > 5 else "")
        print(f"\n⚠️  {len(df) - len(spectra)} of {len(df)} manifest records were not "
              f"rebuilt ({missing}).\n    The h5 and split_indices.p written below are "
              f"consistent with each other, but this dataset is NOT identical to the one "
              f"the published results were computed on.")

    #Save HDF5
    spectra_array = np.stack(spectra, axis=0)   # (N, 1800)
    print(f"\nSaving {spectra_array.shape} → {OUT_H5}")
    with h5py.File(OUT_H5, "w") as h5f:
        h5f.create_dataset("spectra", data=spectra_array,
                           dtype=np.float32, compression="gzip", compression_opts=4)
        h5f.attrs.update({"wn_min": WN_MIN, "wn_max": WN_MAX,
                           "resolution": RESOLUTION, "n_points": N_POINTS,
                           "normalised": True, "units": "absorbance_normalised"})

    #Save SMILES
    np.save(OUT_SMILES, np.array(smiles_out, dtype=object))
    print(f"Saved {len(smiles_out)} SMILES → {OUT_SMILES}")

    OUT_SPLITS = OUT_DIR / "split_indices.p"
    keep = {k: np.array(v, dtype=int) for k, v in splits_out.items() if k != "excluded"}
    np.save(OUT_SPLITS, keep, allow_pickle=True)          # NistIRDataset loads it with np.load(...).item()
    print("Splits → " + ", ".join(f"{k}={len(v)}" for k, v in sorted(keep.items()))
          + f"  ({len(splits_out.get('excluded', []))} excluded) → {OUT_SPLITS}")

    #Save index JSON ────────────────────────────────────────────────────
    with open(OUT_INDEX, "w") as f:
        json.dump({"n_ok": len(index), "n_failed": len(failed),
                   "compounds": index, "failed": failed}, f, indent=2)
    print(f"Index: {len(index)} ok, {len(failed)} failed → {OUT_INDEX}")


if __name__ == "__main__":
    main()