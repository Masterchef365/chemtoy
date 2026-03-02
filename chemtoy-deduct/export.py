from rmgpy.chemkin import load_chemkin_file
import json

T = 298.0 # Kelvin

species, reactions = load_chemkin_file(
    "/mnt/chemkin/chem.inp",
    "/mnt/chemkin/species_dictionary.txt"
)

species_json = []
for s in species:
    species_json.append({
        "smiles": s.smiles, 
        "label": s.label, 
        "charge": s.get_net_charge(), 
        "mass_kg": s.molecular_weight.value_si, 
        "inchi": s.inchi
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
