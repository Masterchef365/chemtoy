use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::laws::{Compound, Element, ElementId, Elements, Formula, Laws};

#[derive(Serialize, Deserialize)]
pub struct ImportFile {
    elements: Vec<ImportElement>,
    compounds: Vec<ImportCompound>,
}

#[derive(Serialize, Deserialize)]
struct ImportCompound {
    name: String,
    formula: String,
    delta_g: f32,
    charge: i32,
    mass: f32,
    composition: HashMap<String, usize>,
}

#[derive(Serialize, Deserialize, Clone)]
struct ImportElement {
    symbol: String,
    mass: f32,
}

fn cvt_formula(elements: &[ImportElement], composition: &HashMap<String, usize>) -> Formula {
    let mut map = BTreeMap::new();
    for (k, v) in composition {
        let id = elements.iter().position(|elem| &elem.symbol == k).unwrap();
        let id = ElementId(id);
        map.insert(id, *v);
    }
    Formula(map)
}

impl ImportFile {
    pub fn convert(&self) -> Laws {
        Laws {
            elements: crate::laws::Elements(
                self.elements
                    .iter()
                    .cloned()
                    .map(|elem| elem.into())
                    .collect(),
            ),
            compounds: crate::laws::Compounds(
                self.compounds
                    .iter()
                    .map(|cmpd| Compound {
                        name: cmpd.name.clone(),
                        mass: cmpd.mass,
                        charge: cmpd.charge,
                        formula: cvt_formula(&self.elements, &cmpd.composition),
                        std_free_energy: cmpd.delta_g,
                    })
                    .collect(),
            ),
        }
    }
}

impl Into<Element> for ImportElement {
    fn into(self) -> Element {
        Element {
            symbol: self.symbol,
            mass: self.mass,
        }
    }
}
