# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from math import gcd
from itertools import permutations
import re
import numpy as np

# ---------------- USER SETTINGS ----------------
ROOT_DIR = Path(r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\New\CeldaUnitaria")
DEST_DIR = Path(r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\New\Supercell\SurfaceEnergy\111_1x1")

HKL_INPUT          = (3, 1, 1)     # base Miller index; scans its {hkl} family if you want
SCAN_FAMILY        = True          # set True to scan permutations of HKL_INPUT
SURFACE_LABEL      = "311"          # used in output names <prefix>_<SURFACE_LABEL>_T#

N_SUBLAYERS_TARGET = 6
VACUUM_ANGSTROM    = 10.0
FREEZE_N_BOTTOM    = 2
ORIGIN_SAMPLES     = 10             # set >1 to sample origins along normal; dedup will collapse repeats
AREA_DECIMALS      = 3              # quantize area to merge numeric dupes

# QE input defaults
CALCULATION_QE  = "relax"
OUTDIR_QE       = "."
PSEUDO_DIR      = "/tmpu/isholek_g/isholek/MIGUEL/QE/PP"
PSEUDO_MAP      = {"Ni": "Ni.upf", "Pd": "Pd.upf"}
INPUT_DFT_QE    = "PBE"
ECUTWFC_QE      = 60.0
ECUTRHO_QE      = 480.0
DEGAUSS_QE      = 0.0146997171
OCCUPATIONS_QE  = "smearing"
SMEARING_QE     = "mp"
VDW_CORR_QE     = "DFT-D3"
DFTD3_VERSION_QE= 6
NSPIN_QE        = 2
NOSYM_QE        = True
NOINV_QE        = True
ASSUME_ISO_QE   = "2D"
CONV_THR_ELEC   = "6.4d-09"
FORC_CONV_THR   = "7.78d-4"
ION_DYN_QE      = "bfgs"
START_MAG_MAP   = {"Ni": 0.65, "Pd": 0.00}

# LSF
LSF_QUEUE = "q_residual"
LSF_NPROC = 16
LSF_EMAIL = "ccuan@pceim.unam.mx"

# k-mesh settings
BULK_KPOINTS_FALLBACK = (4, 4, 4)   # used if bulk .out has no K_POINTS automatic block
K_CLAMP               = (1, 64)     # min/max per direction after scaling

# Convergence markers
END_OK_MARKERS = ("JOB DONE", "JOB DONE.")

# ---------------- ASE ----------------
from ase import Atoms
from ase.io import write as ase_write
from ase.build import surface
from ase.constraints import FixAtoms

# ---------------- HELPERS ----------------
BOHR_TO_ANG = 0.529177210903
_STOP_TOKENS = ("CELL_PARAMETERS","ATOMIC_POSITIONS","K_POINTS","ATOMIC_SPECIES","&","Begin","End")

def reduce_hkl(hkl: Tuple[int,int,int]) -> Tuple[int,int,int]:
    h,k,l = map(int, hkl)
    g = gcd(abs(h), gcd(abs(k), abs(l))) or 1
    h,k,l = h//g, k//g, l//g
    for x in (h,k,l):
        if x < 0: return (-h,-k,-l)
        if x > 0: break
    return (h,k,l)

def hkl_family_variants(hkl: Tuple[int,int,int]) -> List[Tuple[int,int,int]]:
    if not SCAN_FAMILY:
        return [reduce_hkl(hkl)]
    base = tuple(int(x) for x in hkl)
    out, seen = [], set()
    for p in set(permutations(base, 3)):
        rp = reduce_hkl(p)
        if rp not in seen and rp != (0,0,0):
            seen.add(rp); out.append(rp)
    return out

def infer_prefix(stem: str) -> str:
    m = re.match(r'^((?:[A-Z][a-z]?\d*){1,10})', stem)
    if m: return m.group(1)
    m2 = re.match(r'^(.*?)(?:_|$)', stem)
    return m2.group(1) if m2 else stem

# -------- robust QE .out -> Atoms (alat/crystal safe) --------
def _parse_cell_block(lines: List[str], i: int):
    hdr = lines[i].strip().lower()
    if "{angstrom}" in hdr: scale = 1.0
    elif "{bohr}" in hdr:  scale = BOHR_TO_ANG
    else:
        m = re.search(r"\(alat\s*=\s*([0-9.+-Ee]+)\)", lines[i])
        scale = float(m.group(1))*BOHR_TO_ANG if m else 1.0
    v1 = np.fromstring(lines[i+1], sep=" ")
    v2 = np.fromstring(lines[i+2], sep=" ")
    v3 = np.fromstring(lines[i+3], sep=" ")
    return np.vstack([v1,v2,v3]) * scale, i+4

def _parse_positions_block(lines: List[str], i: int, cell: np.ndarray):
    hdr = lines[i].strip().lower()
    mode = "crystal" if "crystal" in hdr else ("bohr" if "bohr" in hdr else "angstrom")
    syms, coords, j = [], [], i+1
    while j < len(lines):
        s = lines[j].strip()
        if (not s) or any(s.startswith(t) for t in _STOP_TOKENS): break
        t = s.split()
        if len(t) < 4: break
        syms.append(t[0]); coords.append([float(t[1]), float(t[2]), float(t[3])])
        j += 1
    X = np.array(coords, float)
    if mode == "crystal": X = X @ cell
    elif mode == "bohr":  X = X * BOHR_TO_ANG
    return syms, X, j

def robust_read_qe_out(path: Path) -> Atoms:
    txt = path.read_text(errors="ignore")
    if not any(m in txt for m in END_OK_MARKERS):
        raise ValueError("Output not converged")
    lines = txt.splitlines()
    cell_is = [i for i,s in enumerate(lines) if s.strip().upper().startswith("CELL_PARAMETERS")]
    pos_is  = [i for i,s in enumerate(lines) if s.strip().upper().startswith("ATOMIC_POSITIONS")]
    if not cell_is or not pos_is:
        raise ValueError("Missing CELL_PARAMETERS/ATOMIC_POSITIONS")
    last_pos = pos_is[-1]
    cell_i = ([i for i in cell_is if i < last_pos] or cell_is)[-1]
    cell, _ = _parse_cell_block(lines, cell_i)
    syms, X, _ = _parse_positions_block(lines, last_pos, cell)
    at = Atoms(symbols=syms, positions=X, cell=cell, pbc=True)
    at.wrap()
    return at

def parse_bulk_kpoints_from_out(path: Path) -> Optional[Tuple[int,int,int]]:
    try: txt = path.read_text(errors="ignore")
    except Exception: return None
    lines = txt.splitlines()
    for i,s in enumerate(lines):
        if s.strip().upper().startswith("K_POINTS") and "automatic" in s.lower():
            j = i+1
            while j < len(lines) and not lines[j].strip(): j += 1
            if j < len(lines):
                t = lines[j].split()
                if len(t) >= 3:
                    try: return (int(t[0]), int(t[1]), int(t[2]))
                    except: pass
    return None

# ---------------- geometry ----------------
def n_hat_from_cell(cell: np.ndarray):
    c = cell[2]; n = np.linalg.norm(c)
    return np.array([0,0,1.0]) if n < 1e-12 else c/n

def project_z(at: Atoms):
    nh = n_hat_from_cell(at.get_cell())
    return at.get_positions() @ nh, nh

def group_sublayers(z: np.ndarray):
    z = np.asarray(z, float)
    order = np.argsort(z); zs = z[order]
    if len(zs) <= 1: return [order.tolist()], np.array([zs.mean() if len(zs) else 0.0])
    dz  = np.diff(zs)
    dzp = dz[dz>1e-8]
    med = np.median(dzp) if dzp.size else (zs.ptp()/max(2,len(zs)-1))
    tol = max(1e-3, 0.30*med)
    planes = [[order[0]]]
    for i in range(1, len(zs)):
        if (zs[i]-zs[i-1]) <= tol: planes[-1].append(order[i])
        else: planes.append([order[i]])
    centers = np.array([z[p].mean() for p in planes], float)
    return planes, centers

def center_with_vacuum(at: Atoms, vacuum: float):
    z, nh = project_z(at)
    thick = float(z.max() - z.min())
    target_c = thick + 2.0*vacuum
    cell = at.get_cell(); cell[2,:] = nh * target_c
    at.set_cell(cell, scale_atoms=False)
    shift = 0.5*target_c - 0.5*(z.max()+z.min())
    at.translate(nh * shift)
    at.set_pbc((True,True,False))
    return at

def freeze_bottom_sublayers(at: Atoms, planes: List[List[int]], n_freeze: int):
    idx = [i for k in range(min(n_freeze, len(planes))) for i in planes[k]]
    if idx:
        at.set_constraint(FixAtoms(indices=sorted(set(idx))))

def top_layer_species_category(at: Atoms) -> Tuple[Tuple[str,...], int]:
    z, _ = project_z(at)
    planes, _ = group_sublayers(z)
    top = planes[-1]
    return tuple(sorted({at[i].symbol for i in top})), len(top)

def enumerate_terminations_simple(bulk: Atoms, hkl: Tuple[int,int,int],
                                  n_sublayers: int, vacuum: float,
                                  freeze_bottom: int) -> List[Atoms]:
    parent_layers = max(int(n_sublayers)+6, int(n_sublayers)+2)
    parent = surface(bulk, hkl, layers=parent_layers, vacuum=0.0, periodic=True)
    z, _ = project_z(parent)
    planes_all, _ = group_sublayers(z)
    slabs = []
    for start in range(0, len(planes_all)-n_sublayers+1):
        idx = [i for pl in planes_all[start:start+n_sublayers] for i in pl]
        s = parent[idx]
        s = center_with_vacuum(s, vacuum)
        z2,_ = project_z(s)
        planes_s,_ = group_sublayers(z2)
        freeze_bottom_sublayers(s, planes_s, freeze_bottom)
        slabs.append(s)
    return slabs

# ---------------- k-mesh (DENSITY from bulk) ----------------
def recip(cell: np.ndarray) -> np.ndarray:
    A = np.array(cell, float).T
    B = 2*np.pi * np.linalg.inv(A)
    return B.T

def kmesh_from_bulk_density(bulk_cell, bulk_k, slab_cell, clamp=(1, 64)):
    """
    Density-match in-plane k-mesh:
      k_i(slab) ≈ k_i(bulk) * (L_i_bulk / L_i_slab),  i = 1,2  (sorted longest→shortest)
      kz(slab) = 1
    Works robustly for cubic bulks and arbitrary (hkl) slabs (skewed included).
    """
    bulk_cell = np.array(bulk_cell, float)
    slab_cell = np.array(slab_cell, float)

    # Bulk: take the three lattice-vector lengths and pick the two longest
    Lb = np.array([np.linalg.norm(bulk_cell[0]),
                   np.linalg.norm(bulk_cell[1]),
                   np.linalg.norm(bulk_cell[2])], float)
    Lb_sorted = np.sort(Lb)[::-1][:2]  # longest two

    # Slab: take in-plane vectors (rows 0 and 1) and sort their lengths (longest first)
    Ls = np.array([np.linalg.norm(slab_cell[0]),
                   np.linalg.norm(slab_cell[1])], float)
    # Guard degenerate cells
    Ls = np.maximum(Ls, 1e-12)
    Ls_sorted = np.sort(Ls)[::-1]

    # Use the same order (longest→shortest) for a stable mapping
    kx_b, ky_b, _kz_b = bulk_k
    scale1 = Lb_sorted[0] / Ls_sorted[0]
    scale2 = Lb_sorted[1] / Ls_sorted[1]

    kx = int(np.clip(np.rint(kx_b * scale1), clamp[0], clamp[1]))
    ky = int(np.clip(np.rint(ky_b * scale2), clamp[0], clamp[1]))
    kz = 1
    return (kx, ky, kz)



# ---------------- writers ----------------
def species_order(at: Atoms) -> List[str]:
    seen=set(); seq=[]
    for a in at:
        if a.symbol not in seen:
            seen.add(a.symbol); seq.append(a.symbol)
    return seq

def frozen_indices(at: Atoms) -> set:
    cons = getattr(at, "constraints", None)
    if not cons: return set()
    if not isinstance(cons,(list,tuple)): cons=[cons]
    idx=set()
    for c in cons:
        if hasattr(c, "get_indices"): idx.update(c.get_indices())
    return idx

def write_pw_in(slab: Atoms, fn: Path, kpts: Tuple[int,int,int], title: Optional[str]=None):
    symbols = [a.symbol for a in slab]
    species = species_order(slab)
    frozen  = frozen_indices(slab)
    L=[]; A=L.append
    A("&CONTROL")
    A(f"  title           = '{title or fn.stem}'")
    A(f"  calculation     = '{CALCULATION_QE}'")
    A(f"  pseudo_dir      = '{PSEUDO_DIR}'")
    A(f"  prefix          = '{fn.stem}'")
    A(f"  outdir          = '{OUTDIR_QE}'")
    A("  restart_mode    = 'from_scratch'")
    A("  tstress         = .true.")
    A("  tprnfor         = .true.")
    etot_thr_ry = len(slab)*7.3498645e-8
    A(f"  etot_conv_thr   = {str(f'{etot_thr_ry:.7e}').replace('e','d')}")
    A(f"  forc_conv_thr   = {FORC_CONV_THR}")
    A("/\n")
    A("&SYSTEM")
    A("    ibrav                     = 0")
    A(f"    nat                       = {len(slab)}")
    A(f"    ntyp                      = {len(species)}")
    A(f"    nspin                     = {NSPIN_QE}")
    A(f"    nosym                     = {'.TRUE.' if NOSYM_QE else '.FALSE.'}")
    A(f"    noinv                     = {'.TRUE.' if NOINV_QE else '.FALSE.'}")
    A("    constrained_magnetization = 'none'")
    A(f"    occupations               = '{OCCUPATIONS_QE}'")
    A(f"    smearing                  = '{SMEARING_QE}'")
    for i,s in enumerate(species,1):
        val = START_MAG_MAP.get(s, 1.0)
        A(f"    starting_magnetization({i}) = {val if isinstance(val,(int,float)) else 1}")
    A(f"    input_dft                 = '{INPUT_DFT_QE}'")
    A(f"    assume_isolated           = '{ASSUME_ISO_QE}'")
    A(f"    vdw_corr                  = '{VDW_CORR_QE}'")
    A(f"    dftd3_version             = {DFTD3_VERSION_QE}")
    A(f"    degauss                   = {DEGAUSS_QE}")
    A(f"    ecutrho                   = {ECUTRHO_QE}")
    A(f"    ecutwfc                   = {ECUTWFC_QE}")
    A("/\n")
    A("&ELECTRONS")
    A(f"    conv_thr         = {CONV_THR_ELEC}")
    A("    diagonalization  = 'david'")
    A("    electron_maxstep = 120")
    A("    mixing_beta      = 0.2")
    A("    mixing_mode      = 'local-TF'")
    A("    mixing_ndim      = 12")
    A("/\n")
    A("&IONS")
    A(f"    ion_dynamics     = '{ION_DYN_QE}'")
    A("/\n")
    A("CELL_PARAMETERS {angstrom}")
    for v in slab.get_cell():
        A(f"  {v[0]:.11f} {v[1]:.11f} {v[2]:.11f}")
    A("")
    A("ATOMIC_SPECIES")
    for s in species:
        mass = slab[symbols.index(s)].mass
        pseudo = PSEUDO_MAP.get(s, f"{s}.upf")
        A(f"  {s} {mass:.4f} {pseudo}")
    A("")
    A("ATOMIC_POSITIONS {angstrom}")
    for i,a in enumerate(slab):
        flag = "0 0 0" if i in frozen else "1 1 1"
        A(f"  {a.symbol} {a.x:.11f} {a.y:.11f} {a.z:.11f} {flag}")
    A("")
    A("K_POINTS automatic")
    A(f"  {kpts[0]} {kpts[1]} {kpts[2]} 0 0 0\n")
    fn.write_text("\n".join(L), encoding="utf-8")

def write_lsf(lsf_path: Path, inp_name: str, jobname: str):
    lsf = f"""#!/bin/bash
#BSUB -J {jobname}
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q {LSF_QUEUE}
#BSUB -n {LSF_NPROC}
#BSUB -R "span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -u {LSF_EMAIL}
#

module load tbb/2021.6.0
module load compiler-rt/2022.1.0
module load intel/2022.1.0
module load mpi/intel-2021.6.0
module load mkl/2022.1.0
module load quantum/7.3

mpirun -np {LSF_NPROC} pw.x -inp {inp_name} > {inp_name.replace('.in','.out')}
"""
    lsf_path.write_text(lsf, encoding="utf-8")

# --- k-mesh helpers (drop in above main) ---------------------------------

def _recip(cell: np.ndarray) -> np.ndarray:
    """Rows = reciprocal vectors b1,b2,b3 for ASE row-wise cell."""
    A = np.array(cell, float)          # rows: a1,a2,a3
    B = 2.0*np.pi * np.linalg.inv(A).T # rows: b1,b2,b3
    return B

def _bulk_target_k_spacing(bulk_cell: np.ndarray, bulk_k: tuple[int,int,int]) -> float:
    """
    Orientation-agnostic target Δk from bulk:
      Δk_i = |b_i(bulk)| / k_i(bulk),  i=1..3 ;  return mean(Δk_i).
    For cubic a and 4×4×4 this is (2π/a)/4.
    """
    B = _recip(bulk_cell)
    kx, ky, kz = bulk_k
    denom = np.array([max(kx,1), max(ky,1), max(kz,1)], float)
    dk = np.linalg.norm(B, axis=1) / denom
    return float(dk.mean())

def kmesh_from_bulk_density_using_surface_ref(
    bulk: "Atoms",
    hkl: tuple[int,int,int],
    bulk_k: tuple[int,int,int],
    slab_cell: np.ndarray,
    clamp: tuple[int,int]=(1,64),
    equalize_tol: float = 1e-6,
) -> tuple[int,int,int]:
    """
    Match slab in-plane Δk to the bulk’s target Δk (orientation independent).
    kx ≈ |b1(slab)| / Δk*, ky ≈ |b2(slab)| / Δk*, kz = 1.
    If in-plane reciprocal lengths are (near) equal (hex/rect), enforce kx=ky.
    """
    dk_target = _bulk_target_k_spacing(bulk.get_cell(), bulk_k)
    B_slab = _recip(slab_cell)

    ks1 = float(np.linalg.norm(B_slab[0]))  # |b1| in-plane
    ks2 = float(np.linalg.norm(B_slab[1]))  # |b2| in-plane

    kx = int(np.clip(np.rint(ks1 / max(dk_target, 1e-12)), clamp[0], clamp[1]))
    ky = int(np.clip(np.rint(ks2 / max(dk_target, 1e-12)), clamp[0], clamp[1]))

    # keep symmetry if in-plane metric is effectively isotropic
    if abs(ks1 - ks2) / max(ks1, ks2, 1e-12) < equalize_tol:
        kx = ky = max(kx, ky)

    return (kx, ky, 1)
# -------------------------------------------------------------------------


# ---------------- MAIN ----------------
def main():
    (DEST_DIR/"geom").mkdir(parents=True, exist_ok=True)
    (DEST_DIR/"relax").mkdir(parents=True, exist_ok=True)

    outs = sorted(ROOT_DIR.rglob("*.[oO][uU][tT]"))
    if not outs:
        print("[!] No QE .out files under ROOT_DIR."); return

    # Per-(system, family) map: (species_set, area_key) -> best slab
    families: Dict[Tuple[str,str], Dict[Tuple[Tuple[str,...], float], Tuple[int,float,Atoms,Tuple[int,int,int]]]] = {}

    hkls = hkl_family_variants(HKL_INPUT)
    print(f"[info] HKL variants: {hkls}")

    for f in outs:
        try: txt = f.read_text(errors="ignore")
        except Exception: continue
        if not any(m in txt for m in END_OK_MARKERS):
            print(f"[skip] {f.name} (not converged)"); continue

        try:
            bulk = robust_read_qe_out(f)
        except Exception as e:
            print(f"[skip] {f.name}: parse error: {e}"); continue

        k_bulk = parse_bulk_kpoints_from_out(f) or BULK_KPOINTS_FALLBACK
        prefix = infer_prefix(f.stem)
        fam_key = (prefix, SURFACE_LABEL)
        if fam_key not in families:
            families[fam_key] = {}

        for hkl in hkls:
            slabs = enumerate_terminations_simple(bulk, hkl, N_SUBLAYERS_TARGET, VACUUM_ANGSTROM, FREEZE_N_BOTTOM)

            # (optional) origin sampling along normal
            if ORIGIN_SAMPLES > 1:
                tmp = surface(bulk, hkl, layers=3, vacuum=0.0, periodic=True)
                nhat = tmp.get_cell()[2] / np.linalg.norm(tmp.get_cell()[2])
                z = tmp.get_positions() @ nhat
                dz = np.diff(np.sort(z)); dzp = dz[dz>1e-3]
                d_hkl = float(np.median(dzp)) if dzp.size else float(np.ptp(z)/max(2,len(z)-1))
                for s_idx in range(1, ORIGIN_SAMPLES):
                    frac = s_idx / ORIGIN_SAMPLES
                    bulk_shifted = bulk.copy(); bulk_shifted.translate(nhat * (frac * d_hkl)); bulk_shifted.wrap()
                    slabs += enumerate_terminations_simple(bulk_shifted, hkl, N_SUBLAYERS_TARGET, VACUUM_ANGSTROM, FREEZE_N_BOTTOM)

            fmap = families[fam_key]
            for s in slabs:
                try:
                    species_key, _n_top = top_layer_species_category(s)
                except Exception as ve:
                    print(f"[skip] {prefix} {hkl}: {ve}")
                    continue

                a_vec, b_vec, _ = s.get_cell()
                area_xy = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
                area_key = round(area_xy, AREA_DECIMALS)
                nat = len(s)
                # old
# kpts = kmesh_from_bulk_density(bulk.get_cell(), k_bulk, s.get_cell(), clamp=K_CLAMP)

# new
                kpts = kmesh_from_bulk_density_using_surface_ref(
                    bulk, hkl, k_bulk, s.get_cell(), clamp=K_CLAMP
                )


                key = (species_key, area_key)
                prev = fmap.get(key)
                if (prev is None) or (nat < prev[0]):
                    fmap[key] = (nat, area_xy, s, kpts)

    # Write one per (species_set, area) with T numbering per (system,family)
    for fam_key, fmap in families.items():
        prefix, family = fam_key
        if not fmap: continue
        # order: single-species alphabetical, then mixed, each by area
        singles, mixed = [], []
        for (species_key, area_key), val in fmap.items():
            rec = (species_key, area_key, *val)  # (species_key, area_key, nat, area_xy, slab, kpts)
            (singles if len(species_key)==1 else mixed).append(rec)
        singles.sort(key=lambda r: (r[0][0], r[1]))
        mixed.sort(key=lambda r: ("+".join(r[0]), r[1]))
        ordered = singles + mixed

        tnum = 0
        for species_key, area_key, nat, area_xy, slab, kpts in ordered:
            tnum += 1
            base = f"{prefix}_{family}_T{tnum}"
            ase_write(DEST_DIR/"geom"/f"{base}.cif", slab, format="cif", wrap=True)
            ase_write(DEST_DIR/"geom"/f"{base}.xyz", slab)
            title = f"{base} (top={' + '.join(species_key)}, area={area_key} Å^2)"
            inpath = DEST_DIR/"relax"/f"{base}_relax.in"
            write_pw_in(slab, inpath, kpts, title=title)
            write_lsf(DEST_DIR/"relax"/f"{base}_relax.lsf", inpath.name, jobname=f"relax-{base}")
            print(f"[ok] {base}: top={'+'.join(species_key)}; area={area_xy:.3f} Å^2; nat={nat}; kpts={kpts}")

    print("[done] One slab per (top-chemistry, area) per (system, family). kz=1; kx,ky density-matched to bulk.")

if __name__ == "__main__":
    main()
