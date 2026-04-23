use chemtoy_deduct::{ChemicalWorld, Compound, CompoundId, Decomposition};
use egui::{RichText, Ui};
use egui_simpletabs::metric::to_metric_prefix;

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
                ui.heading(selected_cmpd.as_ref());
                component_ui(ui, cmpd);
                ui.separator();

                ui.heading("Formation reactions");
                egui::Grid::new("cmpd_formation")
                    .striped(true)
                    .show(ui, |ui| {
                        reaction_header(ui);
                        ui.end_row();

                        for ((a, b), res) in chem.deriv.synthesis.iter() {
                            if res.products.iter().find(|p| *p == selected_cmpd).is_some() {
                                draw_synthesis(ui, a, b, chem, selected_cmpd);
                            }
                        }

                        let sel = selected_cmpd.clone();
                        for (id, _decomps) in &chem.deriv.decompositions {
                            draw_decompositions(ui, id, chem, selected_cmpd, |decomp| {
                                decomp.products.iter().find(|c| *c == &sel).is_some()
                            });
                        }
                    });
                ui.separator();

                ui.heading("In Reagents");
                egui::Grid::new("cmpd_reagents")
                    .striped(true)
                    .show(ui, |ui| {
                        reaction_header(ui);
                        ui.end_row();

                        for (a, b) in chem.deriv.synthesis.keys() {
                            if a.clone() == *selected_cmpd || *b == *selected_cmpd {
                                draw_synthesis(ui, a, b, chem, selected_cmpd);
                            }
                        }
                    });
                ui.separator();

                ui.heading("Decompositions");
                show_decompositions_for_compound(ui, &selected_cmpd.clone(), chem, selected_cmpd);
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

use std::hash::Hasher;

pub fn compound_color(compound: &CompoundId) -> egui::Color32 {
    let mut hash = std::hash::DefaultHasher::new();
    hash.write(compound.as_bytes());
    let bytes = hash.finish().to_le_bytes();
    egui::Color32::from_rgb(bytes[0], bytes[1], bytes[2])
}

pub fn cmpd_label(chem: &ChemicalWorld, value: &CompoundId) -> RichText {
    let cmpd = &chem.deriv.compound_lookup[&value];
    let color = compound_color(&value);
    let mut label = egui::RichText::new(cmpd.label.as_ref()).background_color(color);

    if color.intensity() < 0.3 {
        label = label.color(egui::Color32::WHITE);
    } else {
        label = label.color(egui::Color32::BLACK);
    }

    label
}

pub fn selectable_cmpd(
    ui: &mut Ui,
    chem: &ChemicalWorld,
    value: CompoundId,
    selected_cmpd: &mut CompoundId,
) -> egui::Response {
    let label = cmpd_label(chem, &value);

    ui.selectable_value(selected_cmpd, value.clone(), label)
        .on_hover_ui(|ui| {
            let cmpd = &chem.deriv.compound_lookup[&value];
            component_ui(ui, cmpd);
        })
}

pub fn show_reactions(ui: &mut Ui, chem: &ChemicalWorld, selected_cmpd: &mut CompoundId) {
    ui.heading("Reactions");
    egui::Grid::new("reactions").striped(true).show(ui, |ui| {
        reaction_header(ui);
        ui.end_row();

        for (a, b) in chem.deriv.synthesis.keys() {
            draw_synthesis(ui, a, b, chem, selected_cmpd);
        }
    });
}

pub fn show_compounds(ui: &mut Ui, chem: &ChemicalWorld, selected_cmpd: &mut CompoundId) {
    ui.heading("Compounds");
    egui::Grid::new("compounds").striped(true).show(ui, |ui| {
        ui.strong("Name");
        ui.strong("Charge");
        ui.strong("Mass");
        ui.strong("Radius");
        ui.end_row();

        for (_idx, compound) in chem.laws.species.iter().enumerate() {
            selectable_cmpd(ui, chem, compound.smiles.clone(), selected_cmpd);
            ui.label(format!("{}", &compound.charge));
            ui.label(format!("{}", display_kilogram(compound.mass_kg)));
            ui.label(to_metric_prefix(compound.transport.radius_meters(), "m"));
            ui.end_row();
        }
    });
}

pub fn show_decompositions(ui: &mut Ui, chem: &ChemicalWorld, selected_cmpd: &mut CompoundId) {
    ui.heading("Decompositions");
    egui::Grid::new("decomp").striped(true).show(ui, |ui| {
        reaction_header(ui);
        ui.end_row();
        for (id, _decompositions) in &chem.deriv.decompositions {
            draw_decompositions(ui, id, chem, selected_cmpd, |_| true);
        }
    });
}

pub fn show_decompositions_for_compound(
    ui: &mut Ui,
    id: &CompoundId,
    chem: &ChemicalWorld,
    selected_cmpd: &mut CompoundId,
) {
    egui::Grid::new("decomp").striped(true).show(ui, |ui| {
        reaction_header(ui);
        ui.end_row();
        draw_decompositions(ui, id, chem, selected_cmpd, |_| true);
    });
}

fn draw_decompositions(
    ui: &mut Ui,
    id: &CompoundId,
    chem: &ChemicalWorld,
    selected_cmpd: &mut CompoundId,
    mut filter: impl FnMut(&Decomposition) -> bool,
) {
    let Some(decompositions) = chem.deriv.decompositions.get(id) else {
        return;
    };

    for decomp in decompositions {
        if !filter(decomp) {
            continue;
        }

        selectable_cmpd(ui, chem, id.clone(), selected_cmpd);
        ui.label("->");
        ui.horizontal(|ui| {
            for (i, other_id) in decomp.products.iter().enumerate().rev() {
                selectable_cmpd(ui, chem, other_id.clone(), selected_cmpd);
                if i != 0 {
                    ui.label(" + ");
                }
            }
        });
        ui.label(decomp.activation_energy.e_a.to_string());
        ui.label(decomp.activation_energy.delta_g.to_string());
        ui.end_row();
    }
}

fn draw_synthesis(
    ui: &mut Ui,
    a: &CompoundId,
    b: &CompoundId,
    chem: &ChemicalWorld,
    selected_cmpd: &mut CompoundId,
) {
    let res = &chem.deriv.synthesis[&(a.clone(), b.clone())];
    ui.horizontal(|ui| {
        selectable_cmpd(ui, chem, a.clone(), selected_cmpd);
        ui.label("+");
        selectable_cmpd(ui, chem, b.clone(), selected_cmpd);
    });
    ui.label("->");
    ui.horizontal(|ui| {
        let mut products = res.products.iter().cloned();
        selectable_cmpd(ui, chem, products.next().unwrap(), selected_cmpd);
        if let Some(other) = products.next() {
            ui.label("+");
            selectable_cmpd(ui, chem, other, selected_cmpd);
        }
    });
    ui.label(res.activation_energy.e_a.to_string());
    ui.label(res.activation_energy.delta_g.to_string());
    ui.end_row();
}

fn reaction_header(ui: &mut Ui) {
    ui.strong("Reactants");
    ui.label(""); // Arrow
    ui.strong("Products");
    ui.strong("Activation energy (kJ/mol)");
    ui.strong("Gibbs free energy (kJ/mol)");
}

fn component_ui(ui: &mut Ui, cmpd: &Compound) -> egui::Response {
    egui::Grid::new("cmpd_info")
        .striped(true)
        .show(ui, |ui| {
            let Compound {
                smiles,
                label,
                mass_kg,
                inchi,
                charge,
                transport,
            } = cmpd;
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

            ui.strong("Mass: ");
            ui.label(display_kilogram(*mass_kg));
            ui.end_row();

            ui.strong("Diameter");
            ui.label(to_metric_prefix(transport.radius_meters(), "m"));
            ui.end_row();

        })
        .response
}

fn display_kilogram(value: f64) -> String {
    to_metric_prefix(value * 1000.0, "g")
}
