#!/usr/bin/env python3
import sys
import janaf
import polars as pl
import json
ELEMENTS = "H|He|Li|Be|B|C|N|O|F|NeNa|Mg|Al|Si|P|S|Cl|ArK|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|KrRb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|XeCs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|LuHf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|RnFr|Ra|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|LrRf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn|Nh|Fl|Mc|Lv|Ts|Og"
#regex=f"(({ELEMENTS})+[0-9]*)+[+-]?$"
#filter = pl.col("formula").str.contains(f"^{regex}")
#filter &= pl.col("phase").str.contains('g')
filter = pl.col("phase").str.contains('g')
filtered = janaf.db.filter(filter)

results = []

for name, index, formula, _, phase in filtered.iter_rows():
    table = janaf.Table(index).df
    at_stp = table.filter(table['T(K)'] == 298.15)
    delta_g = at_stp['delta-f G'][0]

    print(f"\"{name}\", \"{formula}\", \"{delta_g}\"")
    results.append({"name": name, "formula": formula, "delta_g": delta_g})

with open('results.json', 'w') as f:
    json.dump(results, f)
#results.sort(key=lambda f: f[-1])

#for name, formula, delta_g in results:
    #print(f"\"{name}\", \"{formula}\", \"{delta_g}\"")


