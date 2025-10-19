#!/usr/bin/env python3
import sys
import janaf
import polars as pl
import json
import re

def get_charge(formula: str) -> int:
    """
    Given a JANAF-style chemical formula (e.g. 'Al', 'Al+', 'Al-2', 'AlCl2-', 'AlCl3+3'),
    return the integer charge.
    """
    formula = formula.strip()
    
    # Match a charge pattern at the end: '+' or '-' optionally followed by digits
    match = re.search(r'([+-])(\d*)$', formula)
    
    if not match:
        return 0  # Neutral compound
    
    sign, num = match.groups()
    magnitude = int(num) if num else 1  # default to ±1 if no number given
    return magnitude if sign == '+' else -magnitude

ELEMENTS = "H|He|Li|Be|B|C|N|O|F|NeNa|Mg|Al|Si|P|S|Cl|ArK|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|KrRb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|XeCs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|LuHf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|RnFr|Ra|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|LrRf|Db|Sg|Bh|Hs|Mt|Ds|Rg|Cn|Nh|Fl|Mc|Lv|Ts|Og"
#ELEMENTS = "H"
regex=f"(({ELEMENTS})+[0-9]*)+[+-]?$"
filter = pl.col("formula").str.contains(f"^{regex}")
filter &= pl.col("phase").str.contains('^g|ref')

filtered = janaf.db.filter(filter)

results = []

for name, index, formula, _nice_name, phase in filtered.iter_rows():
    table = janaf.Table(index).df

    at_stp = table.filter(table['T(K)'] == 298.15)
    delta_g = at_stp['delta-f G'][0]
    charge = get_charge(formula)

    #print(f"\"{name}\", \"{formula}\", \"{delta_g}\"")
    results.append({
        "name": name, 
        "formula": formula, 
        "delta_g": delta_g, 
        "charge": charge
    })

with open('../src/compounds.json', 'w') as f:
    json.dump(results, f, indent=True)
#results.sort(key=lambda f: f[-1])

#for name, formula, delta_g in results:
    #print(f"\"{name}\", \"{formula}\", \"{delta_g}\"")


