use std::collections::{HashMap, HashSet};

use crate::{Compound, CompoundId, Formula, Laws, ProductSet, Products};

impl Derivations {
    pub fn from_laws(laws: &Laws) -> Self {
        println!("Computing decompositions...");
        let decompositions = compute_decompositions(laws);
        println!("Computing syntheses...");
        let synthesis = compute_synthesis(&decompositions);

        let synthesized: HashSet<CompoundId> = synthesis.iter().map(|(_k, v)| *v).collect();
        for (id, compound) in laws.compounds.enumerate() {
            if !synthesized.contains(&id) {
                eprintln!("NOT FOUND: {}", compound.name);
            }
        }

        Self {
            decompositions,
            synthesis,
        }
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

fn compute_decompositions(laws: &Laws) -> HashMap<CompoundId, ProductSet> {
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

    // All other compounds which only contain elements from our compound, and fewer or equal
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
    find_decompositions_rec(
        laws,
        compound,
        &relevant_compounds,
        &mut output,
        &mut vec![],
    );

    output.sort();

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
            output.products.push(Products::from_compound_ids(&stack, laws));
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

/// Reverse of decompositions, but for combinations of only two compounds.
/// If the compounds are (A, B), then the ID of A must be less than or equal to the ID of B. This makes it
/// so that there are no redundant indices.
#[derive(Clone, Debug)]
pub struct Synthesis(HashMap<(CompoundId, CompoundId), CompoundId>);

fn compute_synthesis(
    decompositions: &HashMap<CompoundId, ProductSet>,
) -> Synthesis {
    let mut output: HashMap<(CompoundId, CompoundId), CompoundId> = HashMap::new();
    for (product_id, reactions) in decompositions {
        for reaction in &reactions.products {
            if reaction.count() != 2 {
                continue;
            }

            let mut keys = [CompoundId(0); 2];
            let mut i = 0;
            for (compound_id, n) in reaction.compounds.iter() {
                for _ in 0..*n {
                    keys[i] = *compound_id;
                    i += 1;
                }
            }
            keys.sort_by_key(|CompoundId(i)| *i);

            let [a, b] = keys;
            output.insert((a, b), *product_id);
        }
    }

    Synthesis(output)
}

impl Synthesis {
    pub fn lookup(&self, mut a: CompoundId, mut b: CompoundId) -> Option<CompoundId> {
        if a.0 > b.0 {
            std::mem::swap(&mut a.0, &mut b.0);
        }

        self.0.get(&(a, b)).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&(CompoundId, CompoundId), &CompoundId)> + '_ {
        self.0.iter()
    }
}

pub struct Reaction {
    pub reactants: (CompoundId, CompoundId),
    pub products: CompoundId,
}
