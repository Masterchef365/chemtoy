use crate::import::ImportFile;

pub use laws::*;

mod import;
mod laws;

const COMPOUNDS_JSON: &str = include_str!("compounds.json");

pub fn load_builtin() -> ChemicalWorld {
    println!("Loading database...");
    let import: ImportFile = serde_json::de::from_str(COMPOUNDS_JSON).unwrap();
    println!("Deriving laws...");
    ChemicalWorld::from_laws(import.convert())
}
