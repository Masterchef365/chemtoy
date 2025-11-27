use chemtoy_deduct::ChemicalWorld;

pub fn update_chembook(ctx: &egui::Context, chem: &ChemicalWorld) {
    egui::SidePanel::right("reactions").resizable(true).show(ctx, |ui| {
        ui.heading("Reactions");
        egui::ScrollArea::both()
            .id_salt("reactions")
            .show(ui, |ui| {
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
                            ui.label(&a.name);
                            ui.label("+");
                            ui.label(&b.name);
                        });
                        ui.label("->");
                        ui.label(&res.name);
                        ui.label(format!("{}", free_energy));
                        ui.end_row();
                    }
                });
            });
    });


    egui::SidePanel::left("compounds").resizable(true).show(ctx, |ui| {
        ui.heading("Compounds");
        egui::ScrollArea::both()
            .id_salt("compounds")
            .show(ui, |ui| {
                egui::Grid::new("compounds")
                    .num_columns(5)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("Index");
                        ui.strong("Name");
                        ui.strong("Symbol");
                        ui.strong("Mass");
                        ui.strong("Charge");
                        ui.strong("Std. gibbs free energy");
                        ui.end_row();

                        for (idx, compound) in chem.laws.compounds.0.iter().enumerate() {
                            ui.label(format!("{idx}"));
                            ui.label(&compound.name);
                            ui.label(compound.display(&chem.laws.elements));
                            ui.label(format!("{} u", &compound.mass));
                            ui.label(format!("{}", &compound.charge));
                            ui.label(format!("{} kJ/mol", &compound.std_free_energy));
                            ui.end_row();
                        }
                    });
            });
    });

    egui::CentralPanel::default().show(ctx, |ui| {
        ui.heading("Decompositions");
        egui::ScrollArea::vertical()
            .id_salt("decomp")
            .show(ui, |ui| {
                for (compound_id, decompositions) in &chem.deriv.decompositions {
                    let compound = &chem.laws.compounds[*compound_id];
                    let header =
                        format!("{} [{} kJ/mol]", compound.name, compound.std_free_energy);
                    ui.collapsing(header, |ui| {
                        egui::Grid::new("decomp").striped(true).show(ui, |ui| {
                            ui.strong("Free energy");
                            ui.strong("Products");
                            ui.end_row();

                            for products in decompositions.products.iter() {
                                ui.label(format!("{}", products.total_std_free_energy));
                                ui.horizontal(|ui| {
                                    ui.label("->");
                                    for (i, (other_id, n)) in
                                        products.compounds.iter().enumerate().rev()
                                        {
                                            let other_compound =
                                                &chem.laws.compounds[*other_id];
                                            ui.label(n.to_string());
                                            ui.label(&other_compound.name);
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
            });
    });
}
