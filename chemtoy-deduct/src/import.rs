use std::collections::{BTreeMap, HashMap};

use interned_string::IString;
use serde::{Deserialize, Serialize};

use crate::laws::{Compound, Element, ElementId, Elements, Formula, Laws};

#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub struct Laws {
    pub reactions: Vec<ImportReaction>,
    pub species: Vec<ImportCompound>,
}

#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub struct ImportCompound {
    pub smiles: IString,
    pub label: IString,
    pub mass_amu: f32,
    pub inchi: IString,
}

#[derive(Debug)]
#[derive(Serialize, Deserialize, Clone)]
pub struct ImportReaction {
    #[serde(rename = "A")]
    pub a: f32,
    #[serde(rename = "n")]
    pub n: f32,
    #[serde(rename = "Ea")]
    pub e_a: f32,
    // TODO: Use interned strings(!)
    pub reactants: Vec<IString>,
    pub products: Vec<IString>,
}

impl Laws {
    pub fn built_in() -> Self {
        serde_json::from_slice(include_bytes!("chem.json")).unwrap()
    }
}

#[test]
fn test_parse_built_in() {
    Laws::built_in();
}
