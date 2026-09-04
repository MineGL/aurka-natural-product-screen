#!/usr/bin/env python3
"""Draw 2D depictions of the five compounds clearing both AURKA filters.

Stereochemistry is drawn explicitly (survey precedent: IDO1 Fig 14 -- weakly
supported at one paper, but correct for natural products, where stereocentres
are the point). Black-on-white skeletal, no annotation baked into the image;
identifiers, role tags and scores are composed around the glyphs afterwards so
that the text obeys the figure's own font ladder.

Runs on the workstation with the pipeline's own RDKit (2023.09.6) so the
depiction comes from the same toolkit version that produced the screen.
"""
import json
import os
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

OUT = os.path.expanduser(
    "RESCREEN_ROOT/figures"
)


def draw(smiles, name, width=900, height=700):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("unparseable SMILES for %s" % name)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.bondLineWidth = 3
    opts.fixedFontSize = 34
    opts.clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    path = os.path.join(OUT, "mol_%s.png" % name)
    with open(path, "wb") as fh:
        fh.write(drawer.GetDrawingText())
    # editable .mol for the user
    molblock_path = os.path.join(OUT, "mol_%s.mol" % name)
    Chem.MolToMolFile(mol, molblock_path)
    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                             useLegacyImplementation=False))
    return {
        "png": path,
        "png_bytes": os.path.getsize(path),
        "mol": molblock_path,
        "n_atoms": mol.GetNumHeavyAtoms(),
        "n_stereocentres": n_stereo,
        "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        "canonical_smiles": Chem.MolToSmiles(mol),
    }


def main():
    spec = json.load(open(sys.argv[1]))
    os.makedirs(OUT, exist_ok=True)
    report = {}
    for row in spec:
        info = draw(row["smiles"], row["id"].replace(".", "_"))
        report[row["id"]] = info
        print("%-14s atoms=%-3d stereocentres=%-2d %-14s png=%d" % (
            row["id"], info["n_atoms"], info["n_stereocentres"],
            info["formula"], info["png_bytes"]))
    with open(os.path.join(OUT, "mol_draw_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
