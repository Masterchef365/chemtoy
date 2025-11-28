use chemtoy_deduct::{ChemicalWorld, Compound, CompoundId};
use egui::Ui;

#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum Page {
    #[default]
    Compounds,
    Reactions,
    Decompositions,
}

pub fn update_chembook(ctx: &egui::Context, chem: &ChemicalWorld, selected_cmpd: &mut CompoundId) {
    let mut page = ctx.memory_mut(|mem| *mem.data.get_temp_mut_or_default::<Page>("pages".into()));

    egui::SidePanel::left("pages")
        .resizable(true)
        .show(ctx, |ui| {
            ui.selectable_value(&mut page, Page::Compounds, "Compounds");
            ui.selectable_value(&mut page, Page::Reactions, "Reactions");
            ui.selectable_value(&mut page, Page::Decompositions, "Decompositions");
        });

    ctx.memory_mut(|mem| *mem.data.get_temp_mut_or_default::<Page>("pages".into()) = page);
    egui::SidePanel::right("info")
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::both().show(ui, |ui| {
                let cmpd = &chem.laws.compounds[*selected_cmpd];
                ui.heading(&cmpd.name);
                ui.strong("Info");
                egui::Grid::new("cmpd_info").striped(true).show(ui, |ui| {
                    let Compound {
                        name,
                        formula,
                        charge,
                        std_free_energy,
                        mass,
                    } = cmpd;
                    ui.strong("Formula: ");
                    ui.label(formula.display(&chem.laws.elements));
                    ui.end_row();

                    ui.strong("Charge: ");
                    ui.label(charge.to_string());
                    ui.end_row();

                    ui.strong("Free energy: ");
                    ui.label(std_free_energy.to_string());
                    ui.end_row();

                    ui.strong("Mass: ");
                    ui.label(mass.to_string());
                    ui.end_row();
                });
                ui.separator();

                ui.strong("Formation reactions");
                egui::Grid::new("cmpd_formation")
                    .striped(true)
                    .show(ui, |ui| {
                        for ((a, b), res) in chem.deriv.synthesis.0.iter() {
                            if *res == *selected_cmpd {
                                ui.horizontal(|ui| {
                                    selectable_cmpd(ui, chem, *a, selected_cmpd);
                                    ui.label("+");
                                    selectable_cmpd(ui, chem, *b, selected_cmpd);
                                });
                                ui.horizontal(|ui| {
                                    ui.label("->");
                                    selectable_cmpd(ui, chem, *res, selected_cmpd);
                                });
                                ui.end_row();
                            }
                        }
                    });
                ui.separator();

                ui.strong("Reagents");
                egui::Grid::new("cmpd_reagents")
                    .striped(true)
                    .show(ui, |ui| {
                        for ((a, b), res) in chem.deriv.synthesis.0.iter() {
                            if *a == *selected_cmpd || *b == *selected_cmpd {
                                ui.horizontal(|ui| {
                                    selectable_cmpd(ui, chem, *a, selected_cmpd);
                                    ui.label("+");
                                    selectable_cmpd(ui, chem, *b, selected_cmpd);
                                });
                                ui.horizontal(|ui| {
                                    ui.label("->");
                                    selectable_cmpd(ui, chem, *res, selected_cmpd);
                                });
                                ui.end_row();
                            }
                        }
                    });
                ui.separator();

                ui.strong("Decompositions");
                egui::Grid::new("cmpd_decomp").striped(true).show(ui, |ui| {
                    for productset in &chem.deriv.decompositions[&selected_cmpd].products {
                        ui.horizontal(|ui| {
                            ui.label("->");
                            for (i, (other_id, n)) in productset.compounds.iter().enumerate().rev()
                            {
                                ui.label(n.to_string());
                                selectable_cmpd(ui, chem, *other_id, selected_cmpd);
                                if i != 0 {
                                    ui.label(" + ");
                                }
                            }
                        });
                        ui.end_row();
                    }
                });
                ui.separator();
            });
        });

    egui::CentralPanel::default().show(ctx, |ui| {
        egui::ScrollArea::both()
            .id_salt("reactions")
            .show(ui, |ui| match page {
                Page::Compounds => show_compounds(ui, chem, selected_cmpd),
                Page::Reactions => show_reactions(ui, chem, selected_cmpd),
                Page::Decompositions => show_decompositions(ui, chem, selected_cmpd),
            });
    });
}

fn selectable_cmpd(
    ui: &mut Ui,
    chem: &ChemicalWorld,
    value: CompoundId,
    selected_cmpd: &mut CompoundId,
) -> egui::Response {
    ui.selectable_value(selected_cmpd, value, &chem.laws.compounds[value].name)
}

pub fn show_reactions(ui: &mut Ui, chem: &ChemicalWorld, selected_cmpd: &mut CompoundId) {
    ui.heading("Reactions");
    egui::Grid::new("reactions").striped(true).show(ui, |ui| {
        ui.strong("Reactants");
        ui.strong("");
        ui.strong("Products");
        ui.strong("Delta G");
        ui.end_row();

        for (&(compound_a, compound_b), &product) in chem.deriv.synthesis.iter() {
            let a = &chem.laws.compounds[compound_a];
            let b = &chem.laws.compounds[compound_b];
            let res = &chem.laws.compounds[product];
            let free_energy = a.std_free_energy + b.std_free_energy - res.std_free_energy;

            ui.horizontal(|ui| {
                selectable_cmpd(ui, chem, compound_a, selected_cmpd);
                ui.label("+");
                selectable_cmpd(ui, chem, compound_b, selected_cmpd);
            });
            ui.label("->");
            ui.label(&res.name);
            ui.label(format!("{}", free_energy));
            ui.end_row();
        }
    });
}

pub fn show_compounds(ui: &mut Ui, chem: &ChemicalWorld, selected_cmpd: &mut CompoundId) {
    ui.heading("Compounds");
    egui::Grid::new("compounds")
        .num_columns(5)
        .striped(true)
        .show(ui, |ui| {
            ui.strong("Index");
            ui.strong("Name");
            ui.strong("Charge");
            ui.strong("Std. gibbs free energy");
            ui.strong("Mass");
            ui.strong("Symbol");
            ui.end_row();

            for (id @ CompoundId(idx), compound) in chem.laws.compounds.enumerate() {
                ui.label(format!("{idx}"));
                //ui.label(&compound.name);
                selectable_cmpd(ui, chem, id, selected_cmpd);
                ui.label(format!("{}", &compound.charge));
                ui.label(format!("{} kJ/mol", &compound.std_free_energy));
                ui.label(format!("{} u", &compound.mass));
                ui.label(compound.display(&chem.laws.elements));
                ui.end_row();
            }
        });
}

pub fn show_decompositions(ui: &mut Ui, chem: &ChemicalWorld, selected_cmpd: &mut CompoundId) {
    ui.heading("Decompositions");
    for (compound_id, decompositions) in &chem.deriv.decompositions {
        let compound = &chem.laws.compounds[*compound_id];
        let header = format!("{} [{} kJ/mol]", compound.name, compound.std_free_energy);
        ui.collapsing(header, |ui| {
            egui::Grid::new("decomp").striped(true).show(ui, |ui| {
                ui.strong("Free energy");
                ui.strong("Products");
                ui.end_row();

                for products in decompositions.products.iter() {
                    ui.label(format!("{}", products.total_std_free_energy));
                    ui.horizontal(|ui| {
                        ui.label("->");
                        for (i, (other_id, n)) in products.compounds.iter().enumerate().rev() {
                            //let other_compound = &chem.laws.compounds[*other_id];
                            ui.label(n.to_string());
                            //ui.label(&other_compound.name);
                            selectable_cmpd(ui, chem, *other_id, selected_cmpd);
                            if i != 0 {
                                ui.label(" + ");
                            }
                        }
                    });
                    ui.end_row();
                }
            });
        });
    }
}
