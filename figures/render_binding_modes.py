#!/usr/bin/env python3
"""Build AURKA receptor-ligand complexes and render binding-mode panels.

Conventions follow the literature survey (AURKA_figure_conventions.md 2.7):
grey cartoon receptor; ligand as thick sticks in one saturated colour; contact
residues as green sticks labelled three-letter-number; hinge hydrogen bonds as
dashed lines with distances printed; Glu211 and Ala213 labelled in EVERY panel
(precedent: heliyon Fig 5, LRRK2 Fig 3B). All panels share one orientation so
the three poses are directly comparable.

Placeholders are @@-delimited so that no substituted value can collide with a
literal token in the template -- the previous version replaced the bare string
"LIG" and wrote a filesystem path into the residue-name field.

PyMOL 3.1.0 from the mmp13_docking conda env.
"""
import json
import os
import subprocess

PYMOL = "pymol"
HOME = os.path.expanduser("~")
RESCREEN = os.path.join(HOME, "AURKA_manuscript/rescreen_xgb_gat_20260903")
DOCKIN = os.path.join(HOME, "AURKA_manuscript/docking_coconut_v4_gnina_cuda/inputs")
FIGDIR = os.path.join(RESCREEN, "figures")
RECEPTOR = os.path.join(DOCKIN, "receptor_prepared.pdb")

# name, ligand source, saturated ligand colour (one colour per entity, fixed)
JOBS = [
    ("HIT_CNP0248709_1", os.path.join(RESCREEN, "md/inputs/HIT_CNP0248709_1_pose1.sdf"), "tv_orange"),
    ("HIT_CNP0403858_1", os.path.join(RESCREEN, "md/inputs/HIT_CNP0403858_1_pose1.sdf"), "marine"),
    # yellow, not grey: the cognate ligand was previously indistinguishable
    # from the grey receptor cartoon (heliyon Fig 4 uses yellow for the
    # crystallographic pose).
    ("REF_SKE_cognate", os.path.join(DOCKIN, "crystal_ligand_SKE.sdf"), "yellow"),
]

APPEARANCE = """
bg_color white
set ray_opaque_background, 0
# The receptor is a backdrop, not the subject: a strongly transparent cartoon
# plus outline rendering keeps the ligand separable from the protein.
set cartoon_transparency, 0.70
set stick_radius, 0.13
set ray_trace_mode, 1
set ray_trace_color, black
set ray_trace_gain, 0.12
set label_size, 20
set label_color, black
set label_font_id, 7
set label_outline_color, white
set dash_width, 3.0
set dash_gap, 0.35
set dash_color, black
set label_distance_digits, 2
set antialias, 2
set specular, 0.12
set ray_shadows, 0
"""

BUILD = """
load @@REC@@, rec
load @@LIGSRC@@, ligand
alter ligand, resn="LIG"
alter ligand, chain="B"
alter ligand, resi="1"
rebuild
sort
create cplx, rec or ligand
save @@OUTPDB@@, cplx
python
from pymol import cmd
print("BUILD_RESN", sorted(set(a.resn for a in cmd.get_model("cplx and not polymer").atom)))
print("BUILD_NLIG", cmd.count_atoms("cplx and resn LIG"))
python end
"""

PANEL = """
disable all
enable @@NAME@@
hide everything, @@NAME@@
show cartoon, @@NAME@@ and polymer
color grey90, @@NAME@@ and polymer
set cartoon_side_chain_helper, on, @@NAME@@
show sticks, @@NAME@@ and resn LIG
show spheres, @@NAME@@ and resn LIG
color @@COLOR@@, @@NAME@@ and resn LIG and elem C
util.cnc("@@NAME@@ and resn LIG and not elem C")
# Ligand drawn as ball-and-stick at roughly twice the receptor stick radius so
# it reads as the subject of the panel rather than as part of the pocket.
set stick_radius, 0.26, @@NAME@@ and resn LIG
set sphere_scale, 0.24, @@NAME@@ and resn LIG
hide everything, @@NAME@@ and resn LIG and hydro
# Residues are drawn and labelled ONLY where PLIP reported an interaction for
# this ligand, plus the two hinge residues. Labelling the whole 4.5 A shell
# produced ~20 colliding labels and hid the pose.
select ctc, @@NAME@@ and polymer and resi @@LABELRESI@@
show sticks, ctc and (sidechain or name CA)
color palegreen, ctc and elem C
# Labels are restricted further, to the polar partners (hydrogen bond, salt
# bridge) plus the two hinge residues. Labelling every PLIP contact still
# produced colliding text ("LTHR2917" where Leu139 and Thr217 overlapped);
# hydrophobic contacts remain visible as green sticks but carry no label.
select ctc_lab, @@NAME@@ and polymer and resi @@POLARRESI@@
label ctc_lab and name CA, "%s%s" % (resn, resi)
# Heavy atoms only. The receptor carries explicit hydrogens, so an unrestricted
# polar-contact measurement printed ligand-heavy-to-receptor-hydrogen distances
# (e.g. 1.73 A) that would misread as unphysically short heavy-atom contacts.
distance hb, (@@NAME@@ and resn LIG and not hydro), (@@NAME@@ and polymer and resi 211+213 and not hydro), 3.6, mode=2
color black, hb
# PyMOL's polar-contact mode selects its own atom pairs and printed
# hydrogen-inclusive distances (1.73 A) that would misread as heavy-atom
# contacts. The dashes are kept; exact heavy-atom distances go in the caption.
hide labels, hb
set_view (@@VIEW@@)
png @@OUTPNG@@, width=1600, height=1350, dpi=300, ray=1
python
from pymol import cmd
print("N_LIG", cmd.count_atoms("@@NAME@@ and resn LIG"))
print("N_HBDASH", cmd.count_atoms("hb"))
print("CTC", sorted(set(a.resn + a.resi for a in cmd.get_model("ctc and name CA").atom)))
python end
delete ctc
delete ctc_lab
delete hb
"""


def fill(template, **kw):
    out = template
    for key, val in kw.items():
        out = out.replace("@@" + key + "@@", str(val))
    assert "@@" not in out, "unsubstituted placeholder in template"
    return out


def run(script_text, tag):
    path = os.path.join(FIGDIR, "_" + tag + ".pml")
    with open(path, "w") as fh:
        fh.write(script_text)
    proc = subprocess.run([PYMOL, "-cq", path], capture_output=True, text=True)
    return proc.stdout + proc.stderr


def grab(out, key):
    for line in out.splitlines():
        if line.startswith(key + " "):
            return line[len(key) + 1:].strip()
    return None


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    report = {"complexes": {}, "panels": {}}

    for name, ligsrc, _ in JOBS:
        outpdb = os.path.join(FIGDIR, name + "_complex.pdb")
        out = run(fill(BUILD, REC=RECEPTOR, LIGSRC=ligsrc, OUTPDB=outpdb), "build_" + name)
        resn = grab(out, "BUILD_RESN")
        nlig = grab(out, "BUILD_NLIG")
        assert resn and "LIG" in resn, "resn not set for %s: %s" % (name, resn)
        assert nlig and int(nlig) > 0, "no ligand atoms for %s" % name
        report["complexes"][name] = {"resn": resn, "n_ligand_atoms": int(nlig)}
        print("built %-20s resn=%-12s ligand_atoms=%s" % (name, resn, nlig))

    loads = "\n".join(
        "load %s/%s_complex.pdb, %s" % (FIGDIR, n, n) for n, _, _ in JOBS
    )
    view_script = "\n".join([
        loads, APPEARANCE,
        "hide everything",
        "show sticks, resn LIG",
        "orient (resn LIG) or (polymer and resi 211+213)",
        "zoom (resn LIG) or (polymer and resi 211+213), 4.0",
        "turn x, -10",
        "python",
        "from pymol import cmd",
        'print("VIEW", ",".join("%.6f" % x for x in cmd.get_view()))',
        'print("VIEW_NLIG", cmd.count_atoms("resn LIG"))',
        "python end",
    ])
    out = run(view_script, "view")
    view = grab(out, "VIEW")
    nlig_all = grab(out, "VIEW_NLIG")
    assert view, "no view captured:\n" + out[-800:]
    assert nlig_all and int(nlig_all) == 79, "expected 79 ligand atoms across 3 complexes, got %s" % nlig_all
    report["view"] = view
    print("shared view over %s ligand atoms" % nlig_all)

    polarmap = {'HIT_CNP0248709_1': '143+211+213', 'HIT_CNP0403858_1': '211+213+274', 'REF_SKE_cognate': '211+213'}
    resimap = {'HIT_CNP0248709_1': '139+143+147+160+211+213+217+263', 'HIT_CNP0403858_1': '139+211+213+217+263+274', 'REF_SKE_cognate': '211+213'}
    for name, _, colour in JOBS:
        outpng = os.path.join(FIGDIR, "panel_3d_" + name + ".png")
        if os.path.exists(outpng):
            os.remove(outpng)
        panel_text = "\n".join([
            loads,
            APPEARANCE,
            fill(PANEL, NAME=name, COLOR=colour, VIEW=view, OUTPNG=outpng, LABELRESI=resimap[name], POLARRESI=polarmap[name]),
        ])
        out = run(panel_text, "panel_" + name)
        info = {
            "n_ligand_atoms": grab(out, "N_LIG"),
            "n_hbond_dash_atoms": grab(out, "N_HBDASH"),
            "contact_residues": grab(out, "CTC"),
            "png_bytes": os.path.getsize(outpng) if os.path.exists(outpng) else 0,
        }
        assert info["png_bytes"] > 0, "no png for %s" % name
        assert int(info["n_ligand_atoms"]) > 0, "panel %s has no ligand" % name
        report["panels"][name] = info
        print("panel %-20s lig=%-4s hb_atoms=%-4s png=%d" % (
            name, info["n_ligand_atoms"], info["n_hbond_dash_atoms"], info["png_bytes"]))
        print("    contacts:", (info["contact_residues"] or "")[:160])

    locator = "\n".join([
        loads, APPEARANCE,
        "disable all",
        "enable %s" % JOBS[0][0],
        "hide everything, %s" % JOBS[0][0],
        "show cartoon, %s and polymer" % JOBS[0][0],
        "color grey70, %s and polymer" % JOBS[0][0],
        "show spheres, %s and resn LIG" % JOBS[0][0],
        "color tv_orange, %s and resn LIG" % JOBS[0][0],
        "show sticks, %s and polymer and resi 211+213" % JOBS[0][0],
        "color palegreen, %s and polymer and resi 211+213 and elem C" % JOBS[0][0],
        "set sphere_scale, 0.4",
        "orient %s and polymer" % JOBS[0][0],
        "zoom %s and polymer, 3" % JOBS[0][0],
        "png %s/panel_3d_locator.png, width=1300, height=1300, dpi=300, ray=1" % FIGDIR,
    ])
    run(locator, "locator")
    p = os.path.join(FIGDIR, "panel_3d_locator.png")
    report["locator_png_bytes"] = os.path.getsize(p) if os.path.exists(p) else 0
    print("locator png=%d" % report["locator_png_bytes"])

    # PyMOL session with all three complexes, for the user to re-pose
    session = "\n".join([
        loads, APPEARANCE,
        "hide everything",
        "show cartoon, polymer",
        "color grey80, polymer",
        "show sticks, resn LIG",
        "set_view (%s)" % view,
        "save %s/AURKA_binding_modes.pse" % FIGDIR,
    ])
    run(session, "session")
    s = os.path.join(FIGDIR, "AURKA_binding_modes.pse")
    report["session_bytes"] = os.path.getsize(s) if os.path.exists(s) else 0
    print("session pse=%d" % report["session_bytes"])

    with open(os.path.join(FIGDIR, "render_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
