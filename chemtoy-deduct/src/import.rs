use interned_string::IString;
use serde::{Deserialize, Serialize};

use crate::CompoundId;

#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub struct Laws {
    pub reactions: Vec<Reaction>,
    pub species: Vec<Compound>,
}

#[derive(Debug, Clone)]
#[derive(Serialize, Deserialize)]
pub struct Compound {
    pub smiles: CompoundId,
    pub label: IString,
    pub mass_kg: f64,
    pub inchi: IString,
    pub charge: f64,
    #[serde(rename = "transport")]
    pub transport: Transport,
}

#[derive(Debug)]
#[derive(Serialize, Deserialize, Clone)]
pub struct Reaction {
    #[serde(flatten)]
    pub energy: ActivationEnergy,
    // TODO: Use interned strings(!)
    pub reactants: Vec<IString>,
    pub products: Vec<IString>,
}

#[derive(Debug)]
#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct ActivationEnergy {
    #[serde(rename = "A")]
    pub a: f64,
    #[serde(rename = "n")]
    pub n: f64,
    #[serde(rename = "Ea")]
    pub e_a: f64,
    #[serde(rename = "delta_g")]
    pub delta_g: f64,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct Transport {
    #[serde(rename = "LJ-diam")]
    pub diameter_angstroms: f64,
}

impl Laws {
    pub fn built_in() -> Self {
        serde_json::from_slice(include_bytes!("chem.json")).unwrap()
    }
}

pub const METERS_PER_ANGSTROM: f64 = 1e-10;
impl Transport {
    pub fn radius_meters(&self) -> f64 {
        self.diameter_angstroms * METERS_PER_ANGSTROM / 2.0
    }
}

#[test]
fn test_parse_built_in() {
    Laws::built_in();
}

impl std::fmt::Display for ActivationEnergy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} e^({}/KT)", self.a, self.e_a)
    }
}

impl ActivationEnergy {
    pub fn rate(&self, temperature_kelvin: f64) -> f64 {
        self.a * (self.e_a / temperature_kelvin / 1.381e-23)
    }
}
