from rmgpy.chemkin import load_chemkin_file
import json

species, reactions = load_chemkin_file(
    "/mnt/chemkin/chem.inp",
    "/mnt/chemkin/species_dictionary.txt"
)

species_json = []
for s in species:
    #print(s.label, s.molecular_weight)
    species_json.append((s.label, s.molecular_weight.value_si))

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
            "A": A,
            "n": n,
            "Ea": Ea
        })

print(json.dumps({
    "reactions": reactions_json,
    "species": species_json,
}))
