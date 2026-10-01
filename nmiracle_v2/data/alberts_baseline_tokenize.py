"""Spectral-to-text tokenization, ported line-for-line from
alberts_baseline/benchmark/generate_input.py -- the actual preprocessing
code vendored in this repo (the rxn4chemistry/multimodal-spectroscopic-dataset
benchmark script, which MM-Spectrum's README says it keeps as-is): an
open-vocabulary text serialization (literal rounded-decimal strings as
tokens), not a fixed bin-to-id table.

`process_hnmr`/`process_msms` need peak-level annotations (multiplicity,
J-couplings, proton counts / MS-MS peak lists) that only exist in the raw
Alberts parquet dataset used by alberts_baseline, not in nmiracle_v3's
spectra.h5 (continuous traces only). scripts/build_annotated_peaks_companion.py
joins that raw annotation data onto nmiracle_v3's index space (by InChIKey) so
`build_source_string` can use real h_nmr_peaks/c_nmr_peaks when available --
see AlbertsDataset's `annotated_peaks_path` option. `process_msms` is still
unused: MS/MS isn't part of nmiracle_v3's modalities at all.
"""
import re
from typing import Dict, List, Union

import numpy as np
from scipy.interpolate import interp1d


def tokenize_formula(formula: str) -> str:
    return ' '.join(re.findall(r"[A-Z][a-z]?|\d+|.", formula)) + ' '


def process_hnmr(multiplets: List[Dict[str, Union[str, float, int]]]) -> str:
    multiplet_str = "1HNMR "
    for peak in multiplets:
        range_max = float(peak["rangeMax"])
        range_min = float(peak["rangeMin"])

        formatted_peak = "{:.2f} {:.2f} ".format(range_max, range_min)
        formatted_peak += "{} {}H ".format(peak["category"], peak["nH"])

        js = str(peak["j_values"])
        if js != "None":
            split_js = list(filter(None, js.split("_")))
            processed_js = ["{:.2f}".format(float(j)) for j in split_js]
            formatted_peak += "J " + " ".join(processed_js)

        multiplet_str += formatted_peak.strip() + " | "

    return multiplet_str[:-2]


def process_cnmr(carbon_nmr: List[Dict[str, Union[str, float, int]]]) -> str:
    nmr_string = "13CNMR "
    for peak in carbon_nmr:
        nmr_string += str(round(float(peak["delta (ppm)"]), 1)) + " "
    return nmr_string


def process_ir(ir: np.ndarray, interpolation_points: int = 400) -> str:
    original_x = np.linspace(400, 4000, len(ir))
    interpolation_x = np.linspace(400, 4000, interpolation_points)

    interp = interp1d(original_x, ir)
    interp_ir = interp(interpolation_x)

    # Normalise
    interp_ir = interp_ir + abs(min(interp_ir))
    interp_ir = (interp_ir / max(interp_ir)) * 100
    interp_ir = np.round(interp_ir, decimals=0).astype(int).astype(str)
    return 'IR ' + ' '.join(interp_ir) + ' '


def process_msms(msms: List[List[float]]) -> str:
    msms_string = ''
    for peak in msms:
        msms_string += "{:.1f} {:.1f} ".format(round(peak[0], 1), round(peak[1], 1))
    return msms_string


def build_source_string(
    formula: str = None,
    h_nmr_peaks: List[Dict[str, Union[str, float, int]]] = None,
    c_nmr_peaks: List[Dict[str, Union[str, float, int]]] = None,
    ir_spectrum: np.ndarray = None,
) -> str:
    """Mirrors generate_input.py:tokenise_data's per-sample concatenation
    order (formula, 1H NMR, 13C NMR, IR, MS/MS), restricted to the modalities
    nmiracle_v3 can supply: formula, 1H NMR peaks (when real annotations are
    joined in via annotated_peaks.h5), 13C NMR peaks, IR trace.
    """
    parts = []
    if formula is not None:
        parts.append(tokenize_formula(formula))
    if h_nmr_peaks is not None:
        parts.append(process_hnmr(h_nmr_peaks))
    if c_nmr_peaks is not None:
        parts.append(process_cnmr(c_nmr_peaks))
    if ir_spectrum is not None:
        parts.append(process_ir(ir_spectrum))

    return ''.join(parts).strip()
