# -*- coding: utf-8 -*-
"""
qe2cif_accurate_recursive.py ─ Convierte .out de QE (convergidos) a CIF SIN repetir la celda.
Ajusta posiciones fraccionales con un epsilon para que VESTA muestre correctamente
átomos en los bordes bajo PBC (sin superceldas en el archivo).

Uso:
1) Ajusta ROOT_DIR (entrada) y OUT_DIR (salida).
2) Ejecuta en Spyder / Terminal.
   - Terminal: python qe2cif_accurate_recursive.py "carpeta_in" "carpeta_out"

Dependencias:
    conda install -c conda-forge ase
"""

from pathlib import Path
import sys
from typing import Optional, Iterable
from ase.io import read, write
from ase import Atoms
import numpy as np

# ============ CONFIGURACIÓN ============

# Carpeta raíz con .out de QE
ROOT_DIR = r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Resultados\Nanoparticles\QE\HER_Molecules"
# Carpeta raíz donde guardar .cif (se conserva la estructura de subcarpetas)
OUT_DIR  = r"G:\My Drive\Work\UNAM\Doctorado\Proyecto\Images\HER_Molecules"

# Marcadores de convergencia en el .out de QE
END_OK   = "JOB DONE."
IONIC_OK = ("End final coordinates",)

# Límite opcional de tamaño (None = sin límite)
MAX_OUT_BYTES: Optional[int] = None

# Ajuste para VESTA (no cambia el contenido físico; solo evita frac exactamente 0 o 1)
ADJUST_FOR_VESTA = True
EPS_WRAP = 1e-9           # usado al envolver
EPS_FACE = 1e-6           # empuje mínimo lejos de 0.0 y 1.0 en coordenadas fraccionales

# ============ UTILIDADES ============

def is_converged(path: Path) -> bool:
    """¿El .out terminó OK y tiene coordenadas finales?"""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return (END_OK in text) and any(key in text for key in IONIC_OK)
    except Exception as e:
        print(f"[⚠] No se pudo checar convergencia en {path.name}: {e}")
        return False

def has_positions_qe(path: Path) -> bool:
    """Chequeo rápido de que existan posiciones en el out."""
    keys = ("ATOMIC_POSITIONS", "ATOMIC_POSITIONS (angstrom)", "Begin final coordinates")
    try:
        with open(path, "r", errors="ignore") as f:
            for i, line in enumerate(f):
                if any(k in line for k in keys):
                    return True
                if i > 200000:  # corte temprano por rendimiento
                    break
    except Exception:
        pass
    return False

def read_qe_last_atoms(out_file: Path) -> Atoms:
    """Lee el último frame con posiciones de un .out de QE, con fallback robusto."""
    if MAX_OUT_BYTES is not None and out_file.stat().st_size > MAX_OUT_BYTES:
        raise RuntimeError(f"Archivo muy grande (> {MAX_OUT_BYTES} bytes).")

    if not has_positions_qe(out_file):
        raise RuntimeError("OUT sin geometría (no se encontró ATOMIC_POSITIONS / final coordinates).")

    # Intento rápido: último frame
    try:
        atoms = read(str(out_file), format="espresso-out", index=-1)
        if isinstance(atoms, list):
            atoms = atoms[-1]
        if not isinstance(atoms, Atoms) or len(atoms) == 0:
            raise RuntimeError("Lectura vacía con index=-1.")
        return atoms
    except Exception as e_fast:
        # Fallback: leer todos y tomar el último válido
        try:
            imgs = list(read(str(out_file), format="espresso-out", index=":"))
            imgs = [im for im in imgs if isinstance(im, Atoms) and len(im) > 0]
            if not imgs:
                raise RuntimeError("No se encontraron imágenes válidas en el OUT.")
            return imgs[-1]
        except Exception as e_all:
            raise RuntimeError(f"Fallo leyendo OUT (index=-1 y ':'): {e_fast} | {e_all}")

def adjust_fractional_for_vesta(atoms: Atoms, eps_wrap: float = EPS_WRAP, eps_face: float = EPS_FACE) -> Atoms:
    """
    Devuelve una copia con:
    - wrap(eps) para llevar todo a [0,1) con margen.
    - 'empuje' de las fracciones demasiado cercanas a 0 o 1 (evita que VESTA oculte átomos en la cara).
    No crea superceldas; no duplica átomos. Cambia posiciones en ~1e-6 fraccional (negligible para visualización).
    """
    a = atoms.copy()
    # 1) Envolver cerca de [0,1)
    a.wrap(eps=eps_wrap)

    # 2) Empuje leve de bordes en coordenadas fraccionales
    s = a.get_scaled_positions(wrap=True)  # en [0,1)
    # clamp a (eps_face, 1-eps_face) donde sea necesario
    s = np.where(s < eps_face, eps_face, s)
    s = np.where(s > 1.0 - eps_face, 1.0 - eps_face, s)
    a.set_scaled_positions(s)
    return a

def write_cif_safe(atoms: Atoms, cif_path: Path, adjust_for_vesta: bool = True) -> None:
    """Escribe CIF (P1). Si adjust_for_vesta=True, aplica el “empuje” para visibilidad en VESTA."""
    cif_path.parent.mkdir(parents=True, exist_ok=True)
    a = adjust_fractional_for_vesta(atoms) if adjust_for_vesta else atoms
    write(str(cif_path), a, format="cif")

def convert_to_cif(out_file: Path, root_dir: Path, out_dir: Path) -> None:
    """Convierte un .out de QE a CIF en OUT_DIR, preservando la estructura de subcarpetas."""
    try:
        atoms = read_qe_last_atoms(out_file)

        rel_path   = out_file.relative_to(root_dir)         # p.ej. a\b\c.out
        target_dir = Path(out_dir) / rel_path.parent        # OUT_DIR\a\b
        target_dir.mkdir(parents=True, exist_ok=True)
        cif_path   = target_dir / (out_file.stem + ".cif")  # OUT_DIR\a\b\c.cif

        write_cif_safe(atoms, cif_path, adjust_for_vesta=ADJUST_FOR_VESTA)
        print(f"[✔] {cif_path.relative_to(out_dir)}")

    except Exception as err:
        print(f"[⚠] {out_file.relative_to(root_dir)}: error leyendo/escribiendo ({err})")

# ============ MAIN ============

def main(root: Optional[str] = None, out: Optional[str] = None) -> None:
    root_dir = Path(root or ROOT_DIR).expanduser().resolve()
    out_dir  = Path(out  or OUT_DIR ).expanduser().resolve()

    if not root_dir.exists():
        sys.exit(f"[✘] Carpeta no encontrada: {root_dir}")

    print(f"[ℹ] Escaneando: {root_dir}")
    print(f"[ℹ] Guardando en: {out_dir}")

    outs: Iterable[Path] = root_dir.rglob("*.out")
    any_found = False
    for out_file in outs:
        any_found = True
        if is_converged(out_file):
            convert_to_cif(out_file, root_dir, out_dir)
        else:
            print(f"[✘] {out_file.relative_to(root_dir)}  (no convergió o sin coordenadas finales)")

    if not any_found:
        print("[⚠] No se encontraron archivos .out en el directorio.")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("[ℹ] Sin argumentos. Usando ROOT_DIR y OUT_DIR por defecto.")
        main()
