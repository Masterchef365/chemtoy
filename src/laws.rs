use std::collections::HashMap;

type Formula = HashMap<ElementId, usize>;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ElementId(usize);

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CompoundId(usize);

pub struct Laws {
    pub elements: Elements,
    pub compounds: Compounds,
}

pub struct Element {
    pub symbol: String,
    pub mass: f32,
}

pub struct Compound {
    pub name: String,
    pub formula: Formula,
    pub charge: i32,
    pub std_free_energy: f32,
    pub mass: f32,
}

pub struct Derivations {
    /// For each compound, which other sets of compounds could be formed?
    pub decompositions: HashMap<CompoundId, ProductSet>,
    /// Reverse of decompositions, but for combinations of only two compounds.
    /// If the compounds are (A, B), then the ID of A must be less than or equal to the ID of B. This makes it
    /// so that there are no redundant indices.
    pub reactions: HashMap<(CompoundId, CompoundId), ProductSet>,
}

/// Product set. Sorted by total_std_free_energy.
pub struct ProductSet(Vec<Products>);

pub struct Products {
    /// How many of each compound (238099, 2) -> 2 H2O
    pub compounds: HashMap<CompoundId, usize>,
    pub total_std_free_energy: f32,
}

pub struct ChemicalWorld {
    pub laws: Laws,
    pub deriv: Derivations,
}

#[derive(Default)]
pub struct Compounds(pub Vec<Compound>);

#[derive(Default)]
pub struct Elements(pub Vec<Element>);

impl Element {
    pub fn new(symbol: &str, mass: f32) -> Self {
        Self {
            symbol: symbol.to_string(),
            mass,
        }
    }
}

impl Compound {
    pub fn new(
        name: &str,
        charge: i32,
        std_free_energy: f32,
        formula: &[(ElementId, usize)],
        elements: &Elements,
    ) -> Self {
        let formula: Formula = formula.iter().copied().collect();

        Self {
            name: name.to_string(),
            charge,
            std_free_energy,
            mass: calculate_formula_mass(&formula, elements),
            formula,
        }
    }
}

impl ChemicalWorld {
    pub fn from_laws(laws: Laws) -> Self {
        Self {
            deriv: Derivations::from_laws(&laws),
            laws,
        }
    }
}

impl Derivations {
    pub fn from_laws(laws: &Laws) -> Self {
        Self {
            //reactions: todo!(),
            //decompositions: todo!(),
            reactions: HashMap::new(),
            decompositions: HashMap::new(),
        }
    }
}

impl Elements {
    pub fn lookup(&self, symbol: &str) -> ElementId {
        self.0
            .iter()
            .position(|p| p.symbol == symbol)
            .map(ElementId)
            .expect("Failed to find element")
    }

    pub fn push(&mut self, element: Element) -> ElementId {
        let idx = ElementId(self.0.len());
        self.0.push(element);
        idx
    }
}

impl std::ops::Index<ElementId> for Elements {
    type Output = Element;
    fn index(&self, ElementId(idx): ElementId) -> &Self::Output {
        &self.0[idx]
    }
}

impl std::ops::Index<CompoundId> for Compounds {
    type Output = Compound;
    fn index(&self, CompoundId(idx): CompoundId) -> &Self::Output {
        &self.0[idx]
    }
}

fn calculate_formula_mass(formula: &Formula, elements: &Elements) -> f32 {
    formula
        .iter()
        .map(|(element, n)| *n as f32 * elements[*element].mass)
        .sum()
}

impl Compounds {
    pub fn new(compounds: Vec<Compound>) -> Self {
        Self(compounds)
    }

    pub fn enumerate(&self) -> impl Iterator<Item = (CompoundId, &Compound)> + '_ {
        self.0
            .iter()
            .enumerate()
            .map(|(idx, comp)| (CompoundId(idx), comp))
    }
}
