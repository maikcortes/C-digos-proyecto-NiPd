# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import date
import re
import numpy as np

from ase.io import read, write
from ase import Atoms
from ase.constraints import FixAtoms

# ───────────── USER SETTINGS ───────────────────────────────────────────────
ROOT_DIR         = Path(r"Path")

# Save here (two folders will be created: structures/ and inputs/)
OUTPUT_DIR       = Path(r"Path")
PRESERVE_SUBDIRS = True               # keep relative subpath under ROOT_DIR

STRUCT_SUBDIR    = "structures"       # .xyz, .cif
INPUT_SUBDIR     = "relax"           # .in,  .lsf

# Supercell & freezing
REPEAT           = (3, 3, 1)          # (nx, ny, nz). Use nz=1 for slabs.
FROZEN_PLANES    = 2                  # bottom N atomic planes of 1×1 slab to freeze (carried to supercell)

# QE inputs
KPOINTS          = (1, 1, 1)
ECUTWFC, ECUTRHO = 60, 480
DEGAUSS          = 0.0146997171
PSEUDO_DIR       = "Path"
PSEUDO_MAP       = {"Ni": "Ni.upf", "Pd": "Pd.upf"}
START_MAGN       = {"Ni": 0.75, "Pd": 0.1}    # starting magnetizations

# LSF job defaults
QUEUE_NAME       = "q_"
QUEUE_CORES      = cores
OMP_THREADS      = 1                           # 1 = pure MPI; 2 = hybrid (ranks = CORES//2)
RESOURCE_STRING  = 'span[ptile=]'             # or 'span[ptile=32]' for 32-core nodes

# ───────────── Internals ───────────────────────────────────────────────────
END_OK_RE        = re.compile(r"JOB\s*DONE", re.I)
ENERGY_RE_STRICT = re.compile(r"!\s*total energy\s*=\s*([\-0-9.]+)\s*Ry", re.I)
ENERGY_RE_FALLBK = re.compile(r"\btotal energy\s*=\s*([\-0-9.]+)\s*Ry", re.I)

# remove any “T#” token when grouping variants (e.g. *_T1_*, -T2-, " T3 ")
T_SUFFIX_RE      = re.compile(r"(?i)(?:^|[ _-])T\d+(?=$|[ _-])")

# ─────────── helpers: energy parsing & grouping ────────────────────────────
def extract_final_energy_ry(text: str):
    m = ENERGY_RE_STRICT.findall(text)
    if m:
        try:
            return float(m[-1])
        except ValueError:
            pass
    m2 = ENERGY_RE_FALLBK.findall(text)
    if m2:
        try:
            return float(m2[-1])
        except ValueError:
            pass
    return None

def system_key(out_path: Path):
    """Group by (parent, base) while stripping any T# token from the stem."""
    parent = str(out_path.parent.resolve())
    base   = T_SUFFIX_RE.sub("", out_path.stem).strip("_- ")
    return (parent, base)

# ─────────── helpers: geometry / planes / replication ─────────────────────
def _n_hat_from_cell(cell):
    c = np.asarray(cell, float)[2]
    n = np.linalg.norm(c)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return c / n

def group_planes_with_tol(zvals, tol_override=None):
    """
    Cluster indices into atomic planes by ~equal z (along ĉ).
    Returns (planes [bottom→top], centers[], tol_used).
    """
    z = np.asarray(zvals, float)
    order = np.argsort(z)
    z_sorted = z[order]
    if len(z_sorted) <= 1:
        return [order.tolist()], np.array([float(z.mean())]), (1e-4 if tol_override is None else float(tol_override))

    dif = np.diff(z_sorted)
    if tol_override is None:
        nz = dif[dif > 1e-6]
        tol = max(1e-4, 0.30 * np.percentile(nz, 95)) if len(nz) else 1e-4
    else:
        tol = float(tol_override)

    planes = [[order[0]]]
    for k in range(1, len(z_sorted)):
        if (z_sorted[k] - z_sorted[k-1]) <= tol:
            planes[-1].append(order[k])
        else:
            planes.append([order[k]])
    centers = np.array([z[p].mean() for p in planes], float)
    return planes, centers, tol

def supercell_expand_crop_with_map(slab1: Atoms, nx: int, ny: int):
    """
    Edge-safe supercell:
      1) Build a target cell with a1'=a1*nx, a2'=a2*ny, c'=c (ĉ unchanged).
         a1,a2 are projected to be orthogonal to ĉ to preserve planarity.
      2) Generate images over i=0..nx, j=0..ny in fractional coords and
         keep only those strictly inside f_x,f_y < 1 (crop the last border).
      3) Return the supercell and a map of each supercell atom → source index in 1×1.

    This avoids “missing” border atoms due to periodic wrap.
    """
    A = np.asarray(slab1.get_cell(), float)
    c = A[2]; cn = np.linalg.norm(c) or 1.0
    n_hat = c / cn

    # Project a1,a2 to be orthogonal to ĉ (keeps atomic planes flat in z)
    a1 = A[0] - np.dot(A[0], n_hat) * n_hat
    a2 = A[1] - np.dot(A[1], n_hat) * n_hat
    if np.linalg.norm(a1) < 1e-10 or np.linalg.norm(a2) < 1e-10:
        a1, a2 = A[0], A[1]  # fallback

    A_tgt = np.array([a1 * nx, a2 * ny, n_hat * cn], float)

    sfrac = slab1.get_scaled_positions(wrap=True)
    sfrac = np.minimum(sfrac, 1.0 - 1e-12)
    symbols = np.array(slab1.get_chemical_symbols())

    pos_list, sym_list, src_idx = [], [], []
    for i in range(nx + 1):
        for j in range(ny + 1):
            f = sfrac.copy()
            f[:, 0] = (f[:, 0] + i) / nx
            f[:, 1] = (f[:, 1] + j) / ny
            # strict crop: drop the last border (avoids duplicates and preserves counts)
            mask = (f[:, 0] < 1.0 - 1e-12) & (f[:, 1] < 1.0 - 1e-12)
            if not np.any(mask):
                continue
            r = f[mask] @ A_tgt
            pos_list.append(r)
            sym_list.append(symbols[mask])
            src_idx.append(np.nonzero(mask)[0])

    R = np.vstack(pos_list)
    S = np.concatenate(sym_list).tolist()
    src_map = np.concatenate(src_idx)  # len == len(R)

    scell = Atoms(symbols=S, positions=R, cell=A_tgt, pbc=(True, True, slab1.pbc[2]))

    # clip fractional back inside [0,1)
    frac = scell.get_scaled_positions(wrap=True)
    scell.set_scaled_positions(np.minimum(frac, 1.0 - 1e-12))
    return scell, src_map

def freeze_from_base_planes(slab1: Atoms, scell: Atoms, src_map, n_freeze: int):
    """
    Detect bottom n_freeze atomic planes on the 1×1 slab (ONLY),
    then freeze in the supercell exactly those atoms replicated from the base.
    """
    if n_freeze <= 0:
        return []

    z1 = (slab1.get_positions() @ _n_hat_from_cell(slab1.get_cell()))
    planes1, _, _ = group_planes_with_tol(z1)
    base_freeze = set(i for k in range(min(n_freeze, len(planes1))) for i in planes1[k])

    freeze_idx = [k for k, src in enumerate(src_map) if src in base_freeze]
    if freeze_idx:
        scell.set_constraint(FixAtoms(indices=sorted(set(freeze_idx))))
    return sorted(set(freeze_idx))

def format_plane_counts(counts):
    return "[" + ", ".join(str(x) for x in counts) + "]"

# ─────────── QE input / LSF writers ───────────────────────────────────────
def write_pw_in(slab: Atoms, fn: Path, frozen_idx, kpts):
    symbols = [a.symbol for a in slab]
    species = sorted(set(symbols), key=symbols.index)
    stem = fn.stem
    L, A = [], lambda s: L.append(s)

    # &CONTROL
    A("&CONTROL")
    A(f"  title          = '{stem}',")
    A("  calculation     = 'relax',")
    A("  tstress         = .true.,")
    A("  tprnfor         = .true.,")
    A(f"  pseudo_dir     = '{PSEUDO_DIR}',")
    A(f"  prefix         = '{stem}',")
    A("  outdir          = '.',")
    A("  restart_mode    = 'from_scratch',")
    etot_thr_ry = len(slab) * 7.3498645e-7
    A(f"  etot_conv_thr   = {format(etot_thr_ry, '.6e').replace('e','d')}")
    A("  forc_conv_thr   = 7.8d-4")
    A("/\n")

    # &SYSTEM
    A("&SYSTEM")
    A("  ibrav        = 0,")
    A(f"  nat        = {len(slab)},")
    A(f"  ntyp       = {len(species)},")
    A("  nspin        = 2,")
    for i, s in enumerate(species, 1):
        A(f"  starting_magnetization({i}) =     {START_MAGN.get(s, 0.0):.4f},")
    A(f"  ecutwfc      = {ECUTWFC},")
    A(f"  ecutrho      = {ECUTRHO},")
    A("  occupations  = 'smearing',")
    A("  smearing     = 'mp',")
    A("  nosym        = .true.,")
    A("  input_dft    = 'PBE',")
    A("  assume_isolated = '2D',")
    A("  noinv        = .true.,")
    A(f"  degauss      = {DEGAUSS},")
    A("  vdw_corr     = 'DFT-D3',")
    A("  dftd3_version = 6,")
    A("/\n")

    # &ELECTRONS
    A("&ELECTRONS")
    etot_thr_ry_conv = len(slab) * 7.3498645e-7 * 1e-3
    A(f"  conv_thr         = {format(etot_thr_ry_conv, '.6e').replace('e','d')}")
    A('  diagonalization  = "david"')
    A("  electron_maxstep = 512")
    A("  mixing_beta      = 0.2")
    A('  mixing_mode      = "local-TF"')
    A("  mixing_ndim      = 12")
    A("/\n")

    # &IONS (ionic stopping criteria belong here)
    A("&IONS")
    A("  ion_dynamics  = 'bfgs'")
    A("/\n")

    # ATOMIC_SPECIES
    A("ATOMIC_SPECIES")
    for s in species:
        mass = slab[symbols.index(s)].mass
        pseudo = PSEUDO_MAP.get(s, f"{s}.upf")
        A(f"  {s:2s} {mass:.4f} {pseudo}")
    A("")

    # ATOMIC_POSITIONS
    frozen_set = set(frozen_idx)
    A("ATOMIC_POSITIONS {angstrom}")
    for a in slab:
        flag = "0 0 0" if a.index in frozen_set else "1 1 1"
        A(f"  {a.symbol:2s} {a.x: .10f} {a.y: .10f} {a.z: .10f} {flag}")
    A("")

    # CELL / K_POINTS
    A("CELL_PARAMETERS {angstrom}")
    for v in slab.get_cell():
        A(f"  {v[0]: .10f} {v[1]: .10f} {v[2]: .10f}")
    A("")
    A("K_POINTS {Gamma}")

    fn.write_text("\n".join(L), encoding="utf-8")


def write_lsf(lsf_path: Path, inp_name: str):
    job = inp_name.rsplit(".in", 1)[0]
    ranks = max(1, QUEUE_CORES // max(1, OMP_THREADS))
    lsf_path.write_text(f"""#!/bin/bash
#BSUB -J {job}
#BSUB -q {QUEUE_NAME}
#BSUB -n {QUEUE_CORES}
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -R "{RESOURCE_STRING}"
#BSUB -B
#BSUB -N
#BSUB -u ccuan@pceim.unam.mx
#

module load tbb/2021.6.0
module load compiler-rt/2022.1.0
module load intel/2022.1.0
module load mpi/intel-2021.6.0
module load mkl/2022.1.0
module load quantum/7.3

mpirun -np {ranks} pw.x -in {inp_name} > {job}.out 2>&1
""", encoding="utf-8")

# ─────────── per-file processing ───────────────────────────────────────────
def process_out(out_path: Path):
    txt = out_path.read_text(errors="ignore")
    if not END_OK_RE.search(txt):
        print(f"   · no JOB DONE, skipping: {out_path.name}")
        return

    slab1 = read(out_path, format="espresso-out", index=-1)  # 1×1 relaxed slab
    if REPEAT[2] != 1:
        raise ValueError("This script is for in-plane supercells only (nz must be 1).")
    nx, ny, nz = REPEAT

    # sanity on base planes (for printing only)
    z1 = (slab1.get_positions() @ _n_hat_from_cell(slab1.get_cell()))
    planes1, _, tol1 = group_planes_with_tol(z1)
    base_counts = [len(p) for p in planes1]
    nat_1x1 = len(slab1)

    # build supercell with source map (edge-safe)
    scell, src_map = supercell_expand_crop_with_map(slab1, nx, ny)

    # carry-over freezing from base planes
    frozen_idx = freeze_from_base_planes(slab1, scell, src_map, FROZEN_PLANES)

    # sanity: plane counts in supercell using same tol (slightly relaxed)
    zS = (scell.get_positions() @ _n_hat_from_cell(scell.get_cell()))
    planesS, _, _ = group_planes_with_tol(zS, tol_override=tol1 * 1.05)
    sup_counts = [len(p) for p in planesS]

    expected_atoms = nat_1x1 * nx * ny
    ok_atoms = (len(scell) == expected_atoms)
    ok_planes = (len(base_counts) == len(sup_counts) and
                 all(a * nx * ny == b for a, b in zip(base_counts, sup_counts)))

    # destinations
    if OUTPUT_DIR:
        struct_root = OUTPUT_DIR / STRUCT_SUBDIR
        input_root  = OUTPUT_DIR / INPUT_SUBDIR
        if PRESERVE_SUBDIRS:
            rel_parent = out_path.parent.relative_to(ROOT_DIR)
            struct_base = struct_root / rel_parent
            input_base  = input_root  / rel_parent
        else:
            struct_base = struct_root
            input_base  = input_root
    else:
        struct_base = out_path.parent / STRUCT_SUBDIR
        input_base  = out_path.parent / INPUT_SUBDIR

    struct_base.mkdir(parents=True, exist_ok=True)
    input_base.mkdir(parents=True, exist_ok=True)

    stem = f"{out_path.stem}_{nx}x{ny}x{nz}"
    xyz = struct_base / f"{stem}.xyz"
    cif = struct_base / f"{stem}.cif"
    inp = input_base  / f"{stem}.in"
    lsf = input_base  / f"{stem}.lsf"

    write(xyz, scell)
    write(cif, scell, format="cif", wrap=False)  # keep exact positions, no wrap in CIF
    write_pw_in(scell, inp, frozen_idx, KPOINTS)
    write_lsf(lsf, inp.name)

    shown_struct = struct_base if OUTPUT_DIR is None else struct_base.relative_to(OUTPUT_DIR)
    shown_input  = input_base  if OUTPUT_DIR is None else input_base.relative_to(OUTPUT_DIR)
    print(f"   √ {shown_struct}  |  {shown_input}  (atoms: {len(scell)}, frozen: {len(frozen_idx)})")

    tag1 = "OK" if ok_planes else "[! plane mismatch]"
    tag2 = "OK" if ok_atoms  else f"[! expected {expected_atoms}]"
    print(f"     Sanity planes (↓): 1×1={format_plane_counts(base_counts)}  "
          f"vs  {nx}×{ny}×{nz}={format_plane_counts(sup_counts)} {tag1}")
    print(f"     Atom count: {len(scell)} (expected {expected_atoms}) {tag2}")

# ─────────── main: pick winners & process ──────────────────────────────────
def main():
    outs = list(ROOT_DIR.rglob("*.out"))
    print(f"Scanning {len(outs)} *.out in {ROOT_DIR}")

    winners = {}  # key -> (path, energy_ry)
    skipped = 0

    for f in outs:
        txt = f.read_text(errors="ignore")
        if not END_OK_RE.search(txt):
            skipped += 1
            continue
        e_ry = extract_final_energy_ry(txt)
        if e_ry is None:
            skipped += 1
            continue

        key = system_key(f)
        if key not in winners or e_ry < winners[key][1]:  # more negative is better
            winners[key] = (f, e_ry)

    print(f"Found {len(winners)} systems with a winner (skipped: {skipped}).")
    for (parent, base), (fbest, e) in winners.items():
        rel = fbest.relative_to(ROOT_DIR)
        print(f" → [{base}] winner: {rel.name}  E_final = {e:.6f} Ry")

    for (_, _), (fbest, _) in winners.items():
        print("Processing:", fbest.relative_to(ROOT_DIR))
        process_out(fbest)

    print(f"\nDone ({date.today()})")

if __name__ == "__main__":
    main()

