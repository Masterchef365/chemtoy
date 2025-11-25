use crate::{derivations::Derivations, import::ImportFile};

pub use laws::*;

mod import;
mod laws;
mod derivations;

const COMPOUNDS_JSON: &str = include_str!("compounds.json");

#[derive(Clone, Debug)]
pub struct ChemicalWorld {
    pub laws: Laws,
    pub deriv: Derivations,
}

impl ChemicalWorld {
    pub fn from_laws(laws: Laws) -> Self {
        Self {
            deriv: Derivations::from_laws(&laws),
            laws,
        }
    }
}

pub fn load_builtin() -> ChemicalWorld {
    println!("Loading database...");
    let import: ImportFile = serde_json::de::from_str(COMPOUNDS_JSON).unwrap();
    println!("Deriving laws...");
    ChemicalWorld::from_laws(import.convert())
}
