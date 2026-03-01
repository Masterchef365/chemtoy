pub use import::*;
pub use derivations::*;
use interned_string::IString;

mod import;
//mod laws;
mod derivations;

#[derive(Clone, Debug)]
pub struct ChemicalWorld {
    pub laws: Laws,
    pub deriv: Derivations,
}

/// SMILES
pub type CompoundId = IString;

impl ChemicalWorld {
    pub fn from_laws(laws: Laws) -> Self {
        Self {
            deriv: Derivations::from_laws(&laws),
            laws,
        }
    }
}

pub fn load_builtin() -> ChemicalWorld {
    ChemicalWorld::from_laws(Laws::built_in())
}
