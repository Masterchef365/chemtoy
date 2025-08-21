use std::collections::{BTreeMap, BTreeSet, HashMap};

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

#[derive(Clone, Debug)]
pub struct Derivations {
    /// For each compound, which other sets of compounds could be formed?
    pub decompositions: Vec<(CompoundId, ProductSet)>,
    /// Reverse of decompositions, but for combinations of only two compounds.
    /// If the compounds are (A, B), then the ID of A must be less than or equal to the ID of B. This makes it
    /// so that there are no redundant indices.
    pub reactions: HashMap<(CompoundId, CompoundId), ProductSet>,
}

/// Product set. Sorted by total_std_free_energy.
#[derive(Default, Clone, Debug)]
pub struct ProductSet(pub Vec<Products>);

#[derive(Default, Clone, Debug)]
pub struct Products {
    /// How many of each compound (238099, 2) -> 2 H2O
    pub compounds: BTreeMap<CompoundId, usize>,
    pub total_std_free_energy: f32,
}

#[derive(Clone, Debug)]
pub struct ChemicalWorld {
    pub laws: Laws,
    pub deriv: Derivations,
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
            decompositions: compute_decompositions(laws),
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

fn compute_decompositions(laws: &Laws) -> Vec<(CompoundId, ProductSet)> {
    laws.compounds
        .enumerate()
        .map(|(compound_id, _)| {
            (
                compound_id,
                compute_decompositions_for_compound(laws, compound_id),
            )
        })
        .collect()
}

fn compute_decompositions_for_compound(laws: &Laws, compound_id: CompoundId) -> ProductSet {
    let compound = &laws.compounds[compound_id];

    // All compounds which only contain elements from our compound, and fewer or equal
    // amounts of each (can't have negative amounts in a formula! ... or can you??)
    let relevant_compounds: Vec<(CompoundId, Compound)> = laws
        .compounds
        .enumerate()
        .filter(|(other_id, _)| other_id != &compound_id)
        .filter_map(|(other_id, other_comp)| {
            other_comp
                .formula
                .0
                .iter()
                .all(|(element, n)| Some(n) <= compound.formula.0.get(&element))
                .then(|| (other_id, other_comp.clone()))
        })
        .collect();

    let mut output = ProductSet::default();
    let start = std::time::Instant::now();
    find_decompositions_rec(
        laws,
        compound,
        &relevant_compounds,
        &mut output,
        &mut vec![],
    );

    output.0.sort_by(|a, b| a.total_std_free_energy.partial_cmp(&b.total_std_free_energy).unwrap());

    dbg!(
        laws.compounds[compound_id].display(&laws.elements),
        start.elapsed().as_secs_f32()
    );
    output
}

// TODO: Slow lol
fn find_decompositions_rec(
    laws: &Laws,
    compound: &Compound,
    relevant_compounds: &[(CompoundId, Compound)],
    output: &mut ProductSet,
    stack: &mut Vec<CompoundId>,
) {
    if !check_stack_continue(laws, compound.formula.clone(), stack) {
        if check_stack(laws, compound.formula.clone(), compound.charge, stack) {
            output.0.push(Products::from_compound_ids(&stack, laws));
        }
        return;
    }

    for (relevant_compound_id, _) in relevant_compounds {
        // We only want to visit each possible number of each compounds once
        if Some(relevant_compound_id) >= stack.last() {
            stack.push(*relevant_compound_id);
            find_decompositions_rec(laws, compound, relevant_compounds, output, stack);
            stack.pop();
        }
    }
}

/// Check that "lhs" can decompose to "stack"
fn check_stack(laws: &Laws, mut formula: Formula, mut charge: i32, stack: &[CompoundId]) -> bool {
    for compound_id in stack {
        let compound = &laws.compounds[*compound_id];
        charge -= compound.charge;

        for (element, n) in &mut formula.0 {
            let required = compound.formula.0.get(element).copied().unwrap_or(0);
            if let Some(new_n) = n.checked_sub(required) {
                *n = new_n;
            } else {
                return false;
            }
        }
    }

    formula.0.values().all(|n| *n == 0) && charge == 0
}

/// Check that "lhs" can decompose to "stack"
fn check_stack_continue(laws: &Laws, mut formula: Formula, stack: &[CompoundId]) -> bool {
    let mut charge = 0;
    for compound_id in stack {
        let compound = &laws.compounds[*compound_id];
        charge -= compound.charge;

        for (element, n) in &mut formula.0 {
            let required = compound.formula.0.get(element).copied().unwrap_or(0);
            if let Some(new_n) = n.checked_sub(required) {
                *n = new_n;
            } else {
                return false;
            }
        }
    }

    // TODO: These are slow! Precompute...
    let max_n: usize = formula.0.values().sum();
    let max_charge_mag = laws
        .compounds
        .0
        .iter()
        .map(|c| c.charge.abs())
        .max()
        .unwrap_or(0);
    let has_reasonable_charge = charge.abs() <= max_n as i32 * max_charge_mag;

    formula.0.values().any(|n| *n > 0) && has_reasonable_charge
}

impl Products {
    fn from_compound_ids(ids: &[CompoundId], laws: &Laws) -> Self {
        let mut inst = Self::default();
        for id in ids {
            let compound = &laws.compounds[*id];
            inst.total_std_free_energy += compound.std_free_energy;
            *inst.compounds.entry(*id).or_default() += 1;
        }
        inst
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


use std::cmp::Ordering;

impl ProductSet {
    pub fn nearest_energy(&self, energy: f32) -> Option<usize> {
        // WARNING: chatgpt
        if self.0.is_empty() {
            return None;
        }

        match self.0.binary_search_by(|p| {
            p.total_std_free_energy
                .partial_cmp(&energy)
                .unwrap_or(Ordering::Equal) // handles NaN gracefully
        }) {
            Ok(idx) => Some(idx), // exact match
            Err(idx) => {
                if idx == 0 {
                    Some(0) // energy is below all elements
                } else if idx >= self.0.len() {
                    Some(self.0.len() - 1) // energy is above all elements
                } else {
                    // Between idx-1 and idx -> choose closer
                    let prev = &self.0[idx - 1].total_std_free_energy;
                    let next = &self.0[idx].total_std_free_energy;

                    if (energy - prev).abs() <= (next - energy).abs() {
                        Some(idx - 1)
                    } else {
                        Some(idx)
                    }
                }
            }
        }
    }

    pub fn max_energy(&self) -> f32 {
        self.0.last().map(|v| v.total_std_free_energy).unwrap_or(0.0)
    }

    pub fn min_energy(&self) -> f32 {
        self.0.last().map(|v| v.total_std_free_energy).unwrap_or(0.0)
    }
}

