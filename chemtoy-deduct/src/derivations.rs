use std::collections::{HashMap, HashSet};

use interned_string::IString;

use crate::{ActivationEnergy, Laws, CompoundId};

#[derive(Clone, Debug)]
pub struct Decomposition {
    pub products: Vec<CompoundId>,
    pub activation_energy: ActivationEnergy,
}

#[derive(Clone, Debug)]
pub struct Synthesis {
    pub product: CompoundId,
    pub activation_energy: ActivationEnergy,
}

#[derive(Clone, Debug)]
pub struct Derivations {
    /// For each compound, which other sets of compounds could be formed?
    pub decompositions: HashMap<CompoundId, Decomposition>,
    /// Reverse of decompositions, but for combinations of only two compounds.
    /// If the compounds are (A, B), then the ID of A must be less than or equal to the ID of B. This makes it
    /// so that there are no redundant indices.
    pub synthesis: HashMap<(CompoundId, CompoundId), Synthesis>,
}

impl Derivations {
    pub fn from_laws(laws: &Laws) -> Self {
        let mut synthesis = HashMap::new();
        let mut decompositions = HashMap::new();
        for rxn in &laws.reactions {
            if rxn.reactants.len() == 2 && rxn.products.len() == 1 {
                let a = rxn.reactants[0].clone();
                let b = rxn.reactants[1].clone();
                let product = rxn.products[0].clone();
                synthesis.insert((a, b), Synthesis {
                    product,
                    activation_energy: rxn.energy,
                });
            }

            if rxn.reactants.len() == 1 {
                decompositions.insert(rxn.reactants[0].clone(), Decomposition { products: rxn.products.clone(), activation_energy: rxn.energy });
            }
        }



        Self {
            decompositions,
            synthesis,
        }
    }
}
