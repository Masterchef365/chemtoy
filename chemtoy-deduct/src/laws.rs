use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug)]
pub struct Formula(pub BTreeMap<ElementId, usize>);

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ElementId(pub usize);

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct CompoundId(pub usize);

#[derive(Clone, Debug)]
pub struct Laws {
    pub elements: Elements,
    pub compounds: Compounds,
}

#[derive(Clone, Debug)]
pub struct Element {
    pub symbol: String,
    pub mass: f32,
}

#[derive(Clone, Debug)]
pub struct Compound {
    pub name: String,
    pub formula: Formula,
    pub charge: i32,
    pub std_free_energy: f32,
    pub mass: f32,
}

/// Product set. Sorted by total_std_free_energy.
#[derive(Default, Clone, Debug)]
pub struct ProductSet {
    pub products: Vec<Products>,
    pub n: usize,
}

#[derive(Default, Clone, Debug)]
pub struct Products {
    /// How many of each compound (238099, 2) -> 2 H2O
    pub compounds: BTreeMap<CompoundId, usize>,
    pub total_std_free_energy: f32,
}
#[derive(Default, Clone, Debug)]
pub struct Compounds(pub Vec<Compound>);

#[derive(Default, Clone, Debug)]
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
        let formula = Formula(formula.iter().copied().collect());

        Self {
            name: name.to_string(),
            charge,
            std_free_energy,
            mass: formula.mass(&elements),
            formula,
        }
    }

    pub fn display(&self, elements: &Elements) -> String {
        let mut s = self.formula.display(elements);
        print_superscript_number(&mut s, self.charge);
        s
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

impl Formula {
    pub fn mass(&self, elements: &Elements) -> f32 {
        self.0
            .iter()
            .map(|(element, n)| *n as f32 * elements[*element].mass)
            .sum()
    }

    pub fn display(&self, elements: &Elements) -> String {
        let mut s = String::new();
        for (id, n) in self.0.iter() {
            s.push_str(&elements[*id].symbol);
            print_subscript_number(&mut s, *n as i32);
        }
        s
    }
}

fn print_superscript_number(s: &mut String, mut number: i32) {
    const LUT: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    if number == 0 {
        return;
    }

    if number < 0 {
        number *= -1;
        s.push('-');
    } else {
        s.push('+');
    }

    if number == 1 {
        return;
    }

    let number: usize = number as _;
    for i in (0..number.ilog10() + 1).rev() {
        let v = number / 10_usize.pow(i);
        s.push(LUT[v as usize % 10]);
    }
}

fn print_subscript_number(s: &mut String, mut number: i32) {
    const LUT: [char; 10] = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
    if number < 0 {
        number *= -1;
        s.push('-');
    }

    if number == 1 {
        return;
    }

    let number: usize = number as _;
    for i in (0..number.max(1).ilog10() + 1).rev() {
        let v = number / 10_usize.pow(i);
        s.push(LUT[v as usize % 10]);
    }
}

impl Products {
    pub fn from_compound_ids(ids: &[CompoundId], laws: &Laws) -> Self {
        let mut inst = Self::default();
        for id in ids {
            let compound = &laws.compounds[*id];
            inst.total_std_free_energy += compound.std_free_energy;
            *inst.compounds.entry(*id).or_default() += 1;
        }
        inst
    }

    pub fn count(&self) -> usize {
        self.compounds.iter().map(|(_, n)| *n).sum()
    }
}

impl PartialEq for Products {
    fn eq(&self, other: &Self) -> bool {
        self.compounds == other.compounds
    }
}

impl Eq for Products {}

impl PartialOrd for Products {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.total_std_free_energy
            .partial_cmp(&other.total_std_free_energy)
    }
}

impl Ord for Products {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(&other).unwrap()
    }
}

impl ProductSet {
    pub fn nearest_energy(&self, energy: f32) -> Option<usize> {
        let mut output = None;
        // TODO: Binary search (maybe not even worth it?)
        for (idx, set) in self.products.iter().enumerate() {
            if energy > set.total_std_free_energy {
                output = Some(idx);
            }
        }
        output
    }

    pub fn sort(&mut self) {
        self.products.sort_by(|a, b| {
            a.total_std_free_energy
                .partial_cmp(&b.total_std_free_energy)
                .unwrap()
        });
    }
}
