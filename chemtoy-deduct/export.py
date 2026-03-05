from rmgpy.chemkin import load_chemkin_file, load_transport_file
import json

T = 298.0 # Kelvin

species, reactions = load_chemkin_file(
    "/mnt/chemkin/chem.inp",
    "/mnt/chemkin/species_dictionary.txt"
)

# Load transport
#transport = load_transport_file(transport_params_path, species)
transport_params_path = "/mnt/chemkin/tran.dat"

import numpy as np
transport = np.loadtxt(transport_params_path, skiprows=1, dtype=str, usecols=range(9))
transport_lookup = {}
key_names = transport[0][1:]

for row in transport[1:]:
    transport_lookup[row[0]] = {key_name: row[idx] for idx, key_name in enumerate(key_names)}

species_json = []
for s in species:
    species_json.append({
        "smiles": s.smiles, 
        "label": s.label, 
        "charge": s.get_net_charge(), 
        "mass_kg": s.molecular_weight.value_si, 
        "inchi": s.inchi,
        "transport": transport_lookup[s.to_chemkin()],
    })

reactions_json = []
for rxn in reactions:
    reactants = [sp.molecule[0].to_smiles() for sp in rxn.reactants]
    products  = [sp.molecule[0].to_smiles() for sp in rxn.products]

    kinetics = rxn.kinetics

    if hasattr(kinetics, 'A'):
        A = kinetics.A.value_si
        n = kinetics.n.value_si
        Ea = kinetics.Ea.value_si

        reactions_json.append({
            "reactants": reactants,
            "products": products,
            "delta_g": rxn.get_free_energy_of_reaction(T),
            "A": A,
            "n": n,
            "Ea": Ea
        })

print(json.dumps({
    "reactions": reactions_json,
    "species": species_json,
}))
