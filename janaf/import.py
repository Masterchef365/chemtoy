#!/usr/bin/env python3
import sys
import janaf
import polars as pl
import json
import re
from janaf_parse import analyze_formula, ELEMENT_MASSES

#ELEMENTS = "H|He|Li|Be|B|C|N|O|F|Ne|Na|Mg|Al|Si|P|S|Cl|Ar|K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|Fr|Ra|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn|Nh|Fl|Mc|Lv|Ts|Og"
ELEMENTS = "H"
regex=f"(({ELEMENTS})+[0-9]*)+[+-]?$"
filter = pl.col("formula").str.contains(f"^{regex}")
filter &= pl.col("phase").str.contains('^g|ref')

filtered = janaf.db.filter(filter)

results = []

for name, index, formula, nice_name, phase in filtered.iter_rows():
    table = janaf.Table(index).df

    at_stp = table.filter(table['T(K)'] == 298.15)
    delta_g = at_stp['delta-f G'][0]

    mass, charge, composition = analyze_formula(formula, ELEMENT_MASSES)

    #print(f"\"{name}\", \"{formula}\", \"{delta_g}\"")
    results.append({
        "name": name, 
        #"nice_name": nice_name, 
        "formula": formula, 
        "delta_g": delta_g, 
        "charge": charge,
        "mass": mass,
        "composition": composition,
    })

elements = [{"symbol": sym, "mass": mass} for sym, mass in ELEMENT_MASSES.items()]
output = {
        "elements": elements,
    "compounds": results,
}

with open('../src/compounds.json', 'w') as f:
    json.dump(output, f, indent=True)
#results.sort(key=lambda f: f[-1])

#for name, formula, delta_g in results:
    #print(f"\"{name}\", \"{formula}\", \"{delta_g}\"")


