use std::collections::{HashMap, HashSet};

use crate::{ActivationEnergy, Compound, CompoundId, Laws};

#[derive(Clone, Debug)]
pub struct Decomposition {
    pub products: Vec<CompoundId>,
    pub activation_energy: ActivationEnergy,
}

#[derive(Clone, Debug)]
pub enum SynthesisOutputs {
    Single(CompoundId),
    Exchange(CompoundId, CompoundId),
}

#[derive(Clone, Debug)]
pub struct Synthesis {
    pub products: SynthesisOutputs,
    pub activation_energy: ActivationEnergy,
}

#[derive(Clone, Debug)]
pub struct Derivations {
    /// For each compound, which other sets of compounds could be formed?
    pub decompositions: HashMap<CompoundId, Vec<Decomposition>>,
    /// Reverse of decompositions, but for combinations of only two compounds.
    /// If the compounds are (A, B), then the ID of A must be less than or equal to the ID of B. This makes it
    /// so that there are no redundant indices.
    pub synthesis: HashMap<(CompoundId, CompoundId), Synthesis>,
    pub compound_lookup: HashMap<CompoundId, Compound>,
}

impl Derivations {
    pub fn from_laws(laws: &Laws) -> Self {
        let mut synthesis = HashMap::new();
        let mut decompositions: HashMap<CompoundId, Vec<Decomposition>> = Default::default();

        for rxn in &laws.reactions {
            if rxn.reactants.len() == 2 && rxn.products.len() <= 2 {
                let a = rxn.reactants[0].clone();
                let b = rxn.reactants[1].clone();
                let products = match rxn.products.len() {
                    1 => SynthesisOutputs::Single(rxn.products[0].clone()),
                    _ => SynthesisOutputs::Exchange(rxn.products[0].clone(), rxn.products[1].clone()),
                };

                synthesis.insert(
                    (a, b),
                    Synthesis {
                        products,
                        activation_energy: rxn.energy,
                    },
                );
            }

            if rxn.reactants.len() == 1 {
                decompositions
                    .entry(rxn.reactants[0].clone())
                    .or_default()
                    .push(Decomposition {
                        products: rxn.products.clone(),
                        activation_energy: rxn.energy,
                    });
            }
        }

        let mut compound_lookup = HashMap::new();
        for s in &laws.species {
            compound_lookup.insert(s.smiles.clone(), s.clone());
        }

        Self {
            decompositions,
            synthesis,
            compound_lookup,
        }
    }
}

impl SynthesisOutputs {
    pub fn iter(&self) -> impl Iterator<Item = &CompoundId> + '_ {
        match self {
            Self::Single(id) => [Some(id), None],
            Self::Exchange(a, b) => [Some(a), Some(b)],
        }.into_iter().flatten()
    }
}
