use std::{collections::HashMap, hash::Hasher};

use egui::{Color32, DragValue, Pos2, Rect, RichText, Stroke, Ui, Vec2};
use chemtoy_deduct::{ChemicalWorld, Compound, CompoundId, Compounds, Element, Elements, Laws};
use rand::prelude::Distribution;
use sim::*;

mod query_accel;
mod sim;

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 300.0])
            .with_min_inner_size([300.0, 220.0])
            .with_icon(
                // NOTE: Adding an icon is optional
                eframe::icon_data::from_png_bytes(&include_bytes!("../assets/icon-256.png")[..])
                    .expect("Failed to load icon"),
            ),
        ..Default::default()
    };
    eframe::run_native(
        "eframe template",
        native_options,
        Box::new(|cc| Ok(Box::new(ChemToyApp::new(cc)))),
    )
}

// When compiling to web using trunk:
#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;

    // Redirect `log` message to `console.log` and friends:
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("No window")
            .document()
            .expect("No document");

        let canvas = document
            .get_element_by_id("the_canvas_id")
            .expect("Failed to find the_canvas_id")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("the_canvas_id was not a HtmlCanvasElement");

        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(ChemToyApp::new(cc)))),
            )
            .await;

        // Remove the loading text and spinner:
        if let Some(loading_text) = document.get_element_by_id("loading_text") {
            match start_result {
                Ok(_) => {
                    loading_text.remove();
                }
                Err(e) => {
                    loading_text.set_inner_html(
                        "<p> The app has crashed. See the developer console for details. </p>",
                    );
                    panic!("Failed to start eframe: {e:?}");
                }
            }
        }
    });
}

pub struct ChemToyApp {
    chem: ChemicalWorld,
    sim: Sim,
    sim_cfg: SimConfig,
    scene_rect: Rect,
    draw_compound: CompoundId,
    draw_stationary: bool,
    paused: bool,
    slowdown: usize,
    frame_count: usize,
    with_jittered_grid: bool,
    density: f32,

    screen: Screen,
    vis_cfg: VisualizationConfig,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Screen {
    Simulation,
    ChemBook,
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct SaveData {
    example_value: f32,
}

impl Default for SaveData {
    fn default() -> Self {
        Self {
            // Example stuff:
            example_value: 2.7,
        }
    }
}

impl ChemToyApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        /*
        let save_data = cc
            .storage
            .and_then(|storage| eframe::get_value(storage, eframe::APP_KEY))
            .unwrap_or_default();
        */

        /*
        let mut elements = Elements::default();

        let hydrogen = elements.push(Element {
            mass: 1.008,
            symbol: "H".into(),
        });
        let nitrogen = elements.push(Element {
            mass: 14.007,
            symbol: "N".into(),
        });
        let oxygen = elements.push(Element {
            mass: 15.999,
            symbol: "O".into(),
        });
        /*
        let e = elements.push(Element {
            mass: 1e-3,
            symbol: "e".into(),
        });
        */

        let compounds = Compounds::new(vec![
            //Compound::new("e-", -1, 0.0, &[(e, 1)], &elements),
            Compound::new("H", 0, 203.278, &[(hydrogen, 1)], &elements),
            Compound::new("H-", -1, 132.282, &[(hydrogen, 1)], &elements),
            Compound::new("H+", 1, 1516.99, &[(hydrogen, 1)], &elements),
            Compound::new("H2+", 1, 1484.931, &[(hydrogen, 2)], &elements),
            Compound::new("H2-", -1, 237.732, &[(hydrogen, 2)], &elements),
            Compound::new(
                "H2N",
                0,
                199.846,
                &[(hydrogen, 2), (nitrogen, 1)],
                &elements,
            ),
            Compound::new(
                "H2N2",
                0,
                243.88,
                &[(hydrogen, 2), (nitrogen, 2)],
                &elements,
            ),
            Compound::new("H2O", 0, -237.141, &[(hydrogen, 2), (oxygen, 1)], &elements),
            Compound::new(
                "H2O2",
                0,
                -105.445,
                &[(hydrogen, 2), (oxygen, 2)],
                &elements,
            ),
            Compound::new(
                "H3N",
                0,
                -16.367,
                &[(hydrogen, 3), (nitrogen, 1)],
                &elements,
            ),
            Compound::new("H3O+", 1, 606.607, &[(hydrogen, 3), (oxygen, 1)], &elements),
            Compound::new(
                "H4N2",
                0,
                159.232,
                &[(hydrogen, 4), (nitrogen, 2)],
                &elements,
            ),
            Compound::new("HN", 0, 370.565, &[(hydrogen, 1), (nitrogen, 1)], &elements),
            Compound::new(
                "HNO",
                0,
                112.398,
                &[(hydrogen, 1), (nitrogen, 1), (oxygen, 1)],
                &elements,
            ),
            Compound::new(
                "HNO2",
                0,
                -43.934,
                &[(hydrogen, 1), (nitrogen, 1), (oxygen, 2)],
                &elements,
            ),
            Compound::new(
                "HNO3",
                0,
                -73.941,
                &[(hydrogen, 1), (nitrogen, 1), (oxygen, 3)],
                &elements,
            ),
            Compound::new("HO", 0, 34.277, &[(hydrogen, 1), (oxygen, 1)], &elements),
            Compound::new("HO+", 1, 1306.437, &[(hydrogen, 1), (oxygen, 1)], &elements),
            Compound::new(
                "HO-",
                -1,
                -138.698,
                &[(hydrogen, 1), (oxygen, 1)],
                &elements,
            ),
            Compound::new("HO2", 0, 14.43, &[(hydrogen, 1), (oxygen, 2)], &elements),
            Compound::new("N", 0, 455.54, &[(nitrogen, 1)], &elements),
            Compound::new("N+", 1, 1856.796, &[(nitrogen, 1)], &elements),
            Compound::new("N-", -1, 460.677, &[(nitrogen, 1)], &elements),
            Compound::new("N2+", 1, 1501.44, &[(nitrogen, 2)], &elements),
            Compound::new("N2-", -1, 151.015, &[(nitrogen, 2)], &elements),
            Compound::new("N2O", 0, 104.179, &[(nitrogen, 2), (oxygen, 1)], &elements),
            Compound::new(
                "N2O+",
                1,
                1345.131,
                &[(nitrogen, 2), (oxygen, 1)],
                &elements,
            ),
            Compound::new("N2O3", 0, 139.727, &[(nitrogen, 2), (oxygen, 3)], &elements),
            Compound::new("N2O4", 0, 97.787, &[(nitrogen, 2), (oxygen, 4)], &elements),
            Compound::new("N2O5", 0, 118.013, &[(nitrogen, 2), (oxygen, 5)], &elements),
            Compound::new("N3", 0, 432.387, &[(nitrogen, 3)], &elements),
            Compound::new("NO", 0, 86.6, &[(nitrogen, 1), (oxygen, 1)], &elements),
            Compound::new("NO+", 1, 983.978, &[(nitrogen, 1), (oxygen, 1)], &elements),
            Compound::new("NO2", 0, 51.258, &[(nitrogen, 1), (oxygen, 2)], &elements),
            Compound::new(
                "NO2-",
                -1,
                -177.273,
                &[(nitrogen, 1), (oxygen, 2)],
                &elements,
            ),
            Compound::new("NO3", 0, 116.121, &[(nitrogen, 1), (oxygen, 3)], &elements),
            Compound::new("O", 0, 231.736, &[(oxygen, 1)], &elements),
            Compound::new("O+", 1, 1546.912, &[(oxygen, 1)], &elements),
            Compound::new("O-", -1, 91.638, &[(oxygen, 1)], &elements),
            Compound::new("O2+", 1, 1164.315, &[(oxygen, 2)], &elements),
            Compound::new("O2-", -1, -43.663, &[(oxygen, 2)], &elements),
            Compound::new("O3", 0, 163.184, &[(oxygen, 3)], &elements),
            /*
            Compound::new("H₂", 0, 0.0, &[(hydrogen, 2)], &elements),
            Compound::new("H-", -1, 132.282, &[(hydrogen, 1)], &elements),
            Compound::new("H", 0, 203.278, &[(hydrogen, 1)], &elements),
            Compound::new("H₂+", 1, 1484.931, &[(hydrogen, 2)], &elements),
            Compound::new("H+", 1, 1516.990, &[(hydrogen, 1)], &elements),
            Compound::new("O₂", 0, 0.0, &[(oxygen, 2)], &elements),
            Compound::new("O-", -1, 91.638, &[(oxygen, 1)], &elements),
            Compound::new("O", 0, 231.736, &[(oxygen, 1)], &elements),
            Compound::new("O+²", 2, 1164.315, &[(oxygen, 1)], &elements),
            Compound::new("O+", 1, 1546.912, &[(oxygen, 1)], &elements),
            Compound::new(
                "OH-",
                -1,
                -138.698,
                &[(hydrogen, 1), (oxygen, 1)],
                &elements,
            ),
            Compound::new("H₂O", 0, -228.582, &[(hydrogen, 2), (oxygen, 1)], &elements),
            //Compound::new("H₂O", 0, -228.582, &[(hydrogen, 2), (oxygen, 1)], &elements),
            Compound::new(
                "H₂O₂",
                0,
                -105.445,
                &[(hydrogen, 2), (oxygen, 2)],
                &elements,
            ),
            */
        ]);

        let mut chem = ChemicalWorld::from_laws(Laws {
            elements,
            compounds,
        });
        */

        /*
        for comp in &mut chem.laws.compounds.0 {
            if comp.name != "e-" {
                comp.name = comp.display(&chem.laws.elements);
            }
        }
        */

        let chem = chemtoy_deduct::load_builtin();

        let draw_compound = chem.laws.compounds.enumerate().next().unwrap().0;

        let sim = Sim::new();
        let cfg = SimConfig::default();

        Self {
            draw_stationary: false,
            slowdown: 1,
            frame_count: 0,
            draw_compound,
            chem,
            sim,
            sim_cfg: cfg,
            scene_rect: Rect::ZERO,
            paused: false,
            with_jittered_grid: false,
            screen: Screen::Simulation,
            vis_cfg: Default::default(),
            density: 0.5,
        }
    }
}

impl eframe::App for ChemToyApp {
    /*
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
    eframe::set_value(storage, eframe::APP_KEY, &self.save_data);
    }
    */

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("screen").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.screen, Screen::Simulation, "Simulation");
                ui.selectable_value(&mut self.screen, Screen::ChemBook, "Chem Book");
            });
        });

        match self.screen {
            Screen::Simulation => self.update_simulation(ctx, frame),
            Screen::ChemBook => self.update_chembook(ctx, frame),
        }
    }
}

impl ChemToyApp {
    fn update_simulation(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut single_step = false;

        egui::SidePanel::left("cfg").show(ctx, |ui| {
            egui::ScrollArea::both().show(ui, |ui| {
                ui.group(|ui| {
                    ui.strong("Time");
                    let text = if self.paused { "Paused" } else { "Running" };
                    ui.horizontal(|ui| {
                        ui.label("Slowdown: ");
                        ui.add(DragValue::new(&mut self.slowdown).range(1..=usize::MAX));
                    });
                    self.paused ^= ui.button(text).clicked();
                    single_step |= ui.button("Single step").clicked();
                    ui.checkbox(&mut self.sim_cfg.fill_timestep, "Fill timestep");
                });

                ui.group(|ui| {
                    ui.strong("Drawing");
                    ui.horizontal(|ui| {
                        ui.label("Compound: ");
                        egui::ComboBox::new(
                            "compound",
                            "",
                            //&self.chem.laws.compounds[self.draw_compound].name,
                        )
                        .show_index(
                            ui,
                            &mut self.draw_compound.0,
                            self.chem.laws.compounds.0.len(),
                            |i| self.chem.laws.compounds.0[i].name.clone(),
                        );
                    });
                    ui.checkbox(&mut self.draw_stationary, "Stationary");
                });

                ui.group(|ui| {
                    ui.strong("Simulation");
                    ui.horizontal(|ui| {
                        if ui.button("Reset").clicked() {
                            self.sim = Sim::new();
                            if self.with_jittered_grid {
                                jittered_grid(&mut self.sim, &self.sim_cfg, self.draw_compound, self.density);
                            }
                        }
                        ui.checkbox(&mut self.with_jittered_grid, "with particles");
                        ui.add_enabled(self.with_jittered_grid, DragValue::new(&mut self.density).prefix("Init density: "));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Δt: ");
                        ui.add(
                            DragValue::new(&mut self.sim_cfg.dt)
                                .speed(1e-3)
                                .range(0.0..=f32::MAX)
                                .suffix(" units/step"),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.label("Dimensions: ");
                        ui.add(DragValue::new(&mut self.sim_cfg.dimensions.x));
                        ui.label("x");
                        ui.add(DragValue::new(&mut self.sim_cfg.dimensions.y));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Particle radius: ");
                        ui.add(DragValue::new(&mut self.sim_cfg.particle_radius).speed(1e-2));
                    });
                    /*ui.horizontal(|ui| {
                        ui.label("Max collision time: ");
                        ui.add(DragValue::new(&mut self.sim_cfg.max_collision_time).speed(1e-2));
                    });*/
                    ui.horizontal(|ui| {
                        ui.label("Gravity: ");
                        ui.add(DragValue::new(&mut self.sim_cfg.gravity).speed(1e-2));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Speed limit: ");
                        ui.add(DragValue::new(&mut self.sim_cfg.speed_limit).speed(1e-2));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Kinetic energy scale factor: ");
                        ui.add(DragValue::new(&mut self.sim_cfg.kjmol_per_sim_energy).speed(1e-2));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Morse alpha: ");
                        ui.add(DragValue::new(&mut self.sim_cfg.morse_alpha).speed(1e-2));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Morse magnitude: ");
                        ui.add(DragValue::new(&mut self.sim_cfg.morse_mag).speed(1e-2));
                    });

                    /*
                    ui.horizontal(|ui| {
                        ui.label("Morse radius: ");
                        ui.add(DragValue::new(&mut self.sim_cfg.morse_radius).speed(1e-2));
                    });
                    */

                    ui.horizontal(|ui| {
                        ui.label("Coulomb magnitude: ");
                        ui.add(DragValue::new(&mut self.sim_cfg.coulomb_k).speed(1e-2));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Coulomb softening: ");
                        ui.add(DragValue::new(&mut self.sim_cfg.coulomb_softening).speed(1e-2));
                    });



                    /*
                    // TODO: Neglects mass...
                    let potential_energy = self
                        .sim
                        .particles
                        .iter()
                        .map(|particle| {
                            let h = self.sim_cfg.dimensions.y - particle.pos.y;
                            self.chem.laws.compounds[particle.compound].mass
                                * h
                                * self.sim_cfg.gravity
                        })
                        .sum::<f32>();
                    let kinetic_energy = self
                        .sim
                        .particles
                        .iter()
                        .map(|particle| {
                            self.chem.laws.compounds[particle.compound].mass
                                * particle.vel.length_sq()
                                / 2.0
                        })
                        .sum::<f32>();
                    let total_energy = potential_energy + kinetic_energy;
                    ui.label(format!("Potential energy: {potential_energy:.02}"));
                    ui.label(format!("Kinetic energy energy: {kinetic_energy:.02}"));
                    ui.label(format!("Total energy: {total_energy}"));
                    */

                });

                ui.group(|ui| {
                    ui.strong("Visualization");
                    ui.checkbox(
                        &mut self.vis_cfg.show_velocity_vector,
                        "Show Velocity Vector",
                    );
                    ui.checkbox(
                        &mut self.vis_cfg.show_names,
                        "Show Names",
                    );

                });

                ui.group(|ui| {
                    particle_stats(ui, &self.sim, &self.chem.laws);
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::Scene::new()
                .zoom_range(0.0..=100.0)
                .show(ui, &mut self.scene_rect, |ui| {
                    let (rect, resp) = ui.allocate_exact_size(
                        self.sim_cfg.dimensions,
                        egui::Sense::click_and_drag(),
                    );
                    // Background rect
                    ui.painter().rect_filled(rect, 0.0, Color32::DARK_GRAY);

                    // Bounding rect
                    ui.painter().rect_stroke(
                        //Rect::from_min_max(Pos2::ZERO, Pos2::ZERO + self.cfg.dimensions),
                        rect,
                        0.0,
                        Stroke::new(1., Color32::WHITE),
                        egui::StrokeKind::Outside,
                    );

                    draw_particles(
                        ui,
                        rect,
                        &self.sim.particles,
                        &self.sim_cfg,
                        &self.chem.laws,
                        &self.vis_cfg,
                    );

                    //if let Some(drag_pos) = resp.interact_pointer_pos() {
                    if let Some(interact_pos) = resp.interact_pointer_pos() {
                        if resp.clicked() || resp.dragged() {
                            let pos = interact_pos - rect.min.to_vec2();
                            if self.sim.area_is_clear(&self.sim_cfg, pos) {
                                self.sim.particles.push(Particle {
                                    compound: self.draw_compound,
                                    pos,
                                    vel: resp.drag_delta(),
                                    is_stationary: self.draw_stationary,
                                });
                            }
                        }
                    }
                });
        });

        if !self.paused || single_step {
            if self.frame_count % self.slowdown.max(1) == 0 {
                self.sim.step(&self.sim_cfg, &self.chem);
            }
            ctx.request_repaint();
            self.frame_count += 1;
        }
    }

    fn update_chembook(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        chemtoy::update_chembook(ctx, &self.chem, &mut self.draw_compound)
    }
}

fn jittered_grid(sim: &mut Sim, cfg: &SimConfig, compound: CompoundId, density: f32) {
    let margin = cfg.particle_radius * 2.0 / density;
    let spacing = margin * 2.0;
    let total_width = cfg.particle_radius + spacing;
    let nx = (cfg.dimensions.x / total_width) as i32;
    let ny = (cfg.dimensions.y / total_width) as i32;

    let rand_range = cfg.particle_radius * 1e-2;
    let unif = rand::distributions::Uniform::new(-rand_range, rand_range);

    let mut rng = rand::thread_rng();
    for x in 0..nx {
        for y in 0..ny {
            let mut pos = Pos2::new(
                x as f32 * total_width + margin + cfg.particle_radius,
                y as f32 * total_width + margin + cfg.particle_radius,
            );

            pos.x += unif.sample(&mut rng);
            pos.y += unif.sample(&mut rng);

            sim.particles.push(Particle {
                compound,
                pos,
                vel: Vec2::ZERO,
                is_stationary: false,
            });
        }
    }
}

struct VisualizationConfig {
    show_velocity_vector: bool,
    show_names: bool,
}

fn compound_color(compound: CompoundId) -> Color32 {
    let mut hash = std::hash::DefaultHasher::new();
    hash.write_usize(compound.0);
    let bytes = hash.finish().to_le_bytes();
    Color32::from_rgb(bytes[0], bytes[1], bytes[2])
}

fn draw_particles(
    ui: &mut Ui,
    rect: Rect,
    particles: &[Particle],
    cfg: &SimConfig,
    laws: &Laws,
    vis_cfg: &VisualizationConfig,
) {
    for particle in particles {
        let color = compound_color(particle.compound);

        ui.painter().circle_filled(
            particle.pos + rect.min.to_vec2(),
            cfg.particle_radius,
            color,
        );

        if vis_cfg.show_names {
            let compound = &laws.compounds[particle.compound];
            ui.painter().text(
                particle.pos,
                egui::Align2([egui::Align::Center; 2]),
                &compound.name,
                Default::default(),
                Color32::WHITE,
            );
        }

        if vis_cfg.show_velocity_vector {
            ui.painter().arrow(
                particle.pos + rect.min.to_vec2(),
                particle.vel,
                Stroke::new(1.0, Color32::RED),
            );
        }
    }
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            show_velocity_vector: false,
            show_names: false,
        }
    }
}

fn particle_stats(ui: &mut Ui, sim: &Sim, laws: &Laws) {
    let mut counts: HashMap<CompoundId, usize> = Default::default();
    for part in &sim.particles {
        *counts.entry(part.compound).or_default() += 1;
    }

    let total: usize = counts.iter().map(|(_, n)| *n).sum();

    let mut sorted: Vec<(CompoundId, usize)> = counts.into_iter().collect();
    sorted.sort_by_key(|(_, n)| std::cmp::Reverse(*n));

    egui::Grid::new("stats").show(ui, |ui| {
        for (id, n) in sorted {
            let percent = 100.0 * n as f32 / total as f32;
            ui.label(RichText::new(&laws.compounds[id].name).color(compound_color(id)));
            ui.strong(n.to_string());
            ui.strong(format!("{percent:.02}%"));
            ui.end_row();
        }
    });
}
