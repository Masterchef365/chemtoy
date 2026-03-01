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
                let cmpd = &chem.deriv.compound_lookup[selected_cmpd];
                ui.heading(cmpd.label.as_ref());
                ui.strong("Info");
                egui::Grid::new("cmpd_info").striped(true).show(ui, |ui| {
                    let Compound { smiles, label, mass_amu, inchi, charge } = cmpd;
                    ui.strong("Label: ");
                    ui.label(label.as_ref());
                    ui.end_row();

                    ui.strong("SMILES: ");
                    ui.label(smiles.as_ref());
                    ui.end_row();

                    ui.strong("Charge: ");
                    ui.label(charge.to_string());
                    ui.end_row();

                    ui.strong("InChi: ");
                    ui.label(inchi.as_ref());
                    ui.end_row();

                    ui.strong("Mass (amu): ");
                    ui.label(mass_amu.to_string());
                    ui.end_row();
                });
                ui.separator();

                ui.strong("Formation reactions");
                egui::Grid::new("cmpd_formation")
                    .striped(true)
                    .show(ui, |ui| {
                        for ((a, b), res) in chem.deriv.synthesis.iter() {
                            if res.product == *selected_cmpd {
                                ui.horizontal(|ui| {
                                    selectable_cmpd(ui, chem, a.clone(), selected_cmpd);
                                    ui.label("+");
                                    selectable_cmpd(ui, chem, b.clone(), selected_cmpd);
                                });
                                ui.horizontal(|ui| {
                                    ui.label("->");
                                    selectable_cmpd(ui, chem, res.product.clone(), selected_cmpd);
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
                        for ((a, b), res) in chem.deriv.synthesis.iter() {
                            if a.clone() == *selected_cmpd || *b == *selected_cmpd {
                                ui.horizontal(|ui| {
                                    selectable_cmpd(ui, chem, a.clone(), selected_cmpd);
                                    ui.label("+");
                                    selectable_cmpd(ui, chem, b.clone(), selected_cmpd);
                                });
                                ui.horizontal(|ui| {
                                    ui.label("->");
                                    selectable_cmpd(ui, chem, res.product.clone(), selected_cmpd);
                                });
                                ui.end_row();
                            }
                        }
                    });
                ui.separator();

                ui.strong("Decompositions");
                egui::Grid::new("cmpd_decomp").striped(true).show(ui, |ui| {
                    let productset = &chem.deriv.decompositions[&selected_cmpd].products;
                        ui.horizontal(|ui| {
                            ui.label("->");
                            for (i, other_id) in productset.iter().enumerate().rev() {
                                selectable_cmpd(ui, chem, other_id.clone(), selected_cmpd);
                                if i != 0 {
                                    ui.label(" + ");
                                }
                            }
                        });
                        ui.end_row();
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

pub fn selectable_cmpd(
    ui: &mut Ui,
    chem: &ChemicalWorld,
    value: CompoundId,
    selected_cmpd: &mut CompoundId,
) -> egui::Response {
    let label = chem.deriv.compound_lookup[&value].label.as_ref();
    ui.selectable_value(selected_cmpd, value, label)
}

pub fn show_reactions(ui: &mut Ui, chem: &ChemicalWorld, selected_cmpd: &mut CompoundId) {
    ui.heading("Reactions");
    egui::Grid::new("reactions").striped(true).show(ui, |ui| {
        ui.strong("Reactants");
        ui.strong("");
        ui.strong("Products");
        ui.strong("Delta G");
        ui.end_row();

        for ((compound_a, compound_b), product) in chem.deriv.synthesis.iter() {
            let res = &chem.deriv.compound_lookup[&product.product];
            let activation_energy = product.activation_energy;

            ui.horizontal(|ui| {
                selectable_cmpd(ui, chem, compound_a.clone(), selected_cmpd);
                ui.label("+");
                selectable_cmpd(ui, chem, compound_b.clone(), selected_cmpd);
            });
            ui.label("->");
            ui.label(res.label.as_ref());
            ui.label(format!("{} kJ/mol", activation_energy));
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

            for (_idx, compound) in chem.laws.species.iter().enumerate() {
                selectable_cmpd(ui, chem, compound.smiles.clone(), selected_cmpd);
                ui.label(format!("Mass: {}", &compound.charge));
                //ui.label(format!("{} kJ/mol", &compound.std_free_energy));
                ui.label(format!("{} amu", &compound.mass_amu));
                //ui.label(compound.display(&chem.laws.elements));
                ui.end_row();
            }
        });
}

pub fn show_decompositions(ui: &mut Ui, chem: &ChemicalWorld, selected_cmpd: &mut CompoundId) {
    ui.heading("Decompositions");
    for (compound_id, decomposition) in &chem.deriv.decompositions {
        let compound = &chem.deriv.compound_lookup[compound_id];
        let header = format!("{}", compound.label);
        ui.collapsing(header, |ui| {
            egui::Grid::new("decomp").striped(true).show(ui, |ui| {
                ui.strong("Activation energy");
                ui.strong("Products");
                ui.end_row();

                ui.horizontal(|ui| {
                    ui.label("->");
                    for (i, other_id) in decomposition.products.iter().enumerate().rev() {
                        selectable_cmpd(ui, chem, other_id.clone(), selected_cmpd);
                        if i != 0 {
                            ui.label(" + ");
                        }
                    }
                });
                ui.end_row();
            });
        });
    }
}
