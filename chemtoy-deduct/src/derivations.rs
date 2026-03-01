use std::collections::{HashMap, HashSet};

use interned_string::IString;

use crate::{Laws};

pub type CompoundId = IString;
pub type ProductSet = Vec<IString>;

impl Derivations {
    pub fn from_laws(laws: &Laws) -> Self {
    }
}

#[derive(Clone, Debug)]
pub struct Derivations {
    /// For each compound, which other sets of compounds could be formed?
    pub decompositions: HashMap<CompoundId, ProductSet>,
    /// Reverse of decompositions, but for combinations of only two compounds.
    /// If the compounds are (A, B), then the ID of A must be less than or equal to the ID of B. This makes it
    /// so that there are no redundant indices.
    pub synthesis: Synthesis,
}

/// Reverse of decompositions, but for combinations of only two compounds.
/// If the compounds are (A, B), then the ID of A must be less than or equal to the ID of B. This makes it
/// so that there are no redundant indices.
#[derive(Clone, Debug)]
pub struct Synthesis(pub HashMap<(CompoundId, CompoundId), CompoundId>);
