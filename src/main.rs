use std::collections::HashMap;

use egui::{DragValue, Pos2, Vec2};
use laws::{ChemicalWorld, Compound, CompoundId, Compounds, Element, Elements, Laws};
use query_accel::QueryAccelerator;

mod laws;
mod query_accel;

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
        Box::new(|cc| Ok(Box::new(TemplateApp::new(cc)))),
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
                Box::new(|cc| Ok(Box::new(TemplateApp::new(cc)))),
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

pub struct TemplateApp {
    chem: ChemicalWorld,
    sim: Sim,
    cfg: SimConfig,
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

impl TemplateApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        /*
        let save_data = cc
            .storage
            .and_then(|storage| eframe::get_value(storage, eframe::APP_KEY))
            .unwrap_or_default();
        */

        let mut elements = Elements::default();

        let hydrogen = elements.push(Element {
            mass: 1.008,
            symbol: "H".into(),
        });
        let oxygen = elements.push(Element {
            mass: 15.999,
            symbol: "O".into(),
        });

        let compounds = Compounds::new(vec![
            Compound::new("H₂", 1, 0.0, &[(hydrogen, 2)], &elements),
            Compound::new("H⁻", -1, 132.282, &[(hydrogen, 1)], &elements),
            Compound::new("H", 0, 203.278, &[(hydrogen, 1)], &elements),
            //Compound::new("H₂⁺", 1, 1484.931, &[(hydrogen, 2)]),
            //Compound::new("H⁺", 1, 1516.990, &[(hydrogen, 1)]),

            Compound::new("O₂", 0, 0.0, &[(hydrogen, 2)], &elements),
            Compound::new("O⁻", -1, 91.638, &[(hydrogen, 1)], &elements),
            Compound::new("O", 0, 231.736, &[(hydrogen, 1)], &elements),
            //Compound::new("O⁺²", 1, 1164.315, &[(hydrogen, 1)]),
            //Compound::new("O⁺", 1, 1546.912, &[(hydrogen, 1)]),

            Compound::new("OH⁻", -1, -138.698, &[(hydrogen, 1), (oxygen, 1)], &elements),
            Compound::new("H₂O", 0, -228.582, &[(hydrogen, 2), (oxygen, 1)], &elements),
            Compound::new("H₂O", 0, -228.582, &[(hydrogen, 2), (oxygen, 1)], &elements),
            Compound::new("H₂O₂", 0, -105.445, &[(hydrogen, 2), (oxygen, 2)], &elements),
        ]);

        let chem = ChemicalWorld::from_laws(Laws {
            elements,
            compounds,
        });

        let sim = Sim::new();

        Self { chem, sim, cfg: SimConfig::default() }
    }
}

impl eframe::App for TemplateApp {
    /*
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.save_data);
    }
    */

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("cfg").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Δt: ");
                ui.add(DragValue::new(&mut self.cfg.dt).suffix(" units/step"));
            });
            ui.horizontal(|ui| {
                ui.label("Dimensions: ");
                ui.add(DragValue::new(&mut self.cfg.dimensions.x));
                ui.label("x");
                ui.add(DragValue::new(&mut self.cfg.dimensions.y));
            });
            ui.horizontal(|ui| {
                ui.label("Particle radius: ");
                ui.add(DragValue::new(&mut self.cfg.particle_radius));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label("hi");
        });
    }
}

struct Sim {
    particles: Vec<Particle>,
}

struct Particle {
    compound: CompoundId,
    pos: Pos2,
    vel: Vec2,
}

impl Sim {
    pub fn new() -> Self {
        Self {
            particles: vec![],
        }
    }

    pub fn step(&mut self, cfg: &SimConfig, chem: &ChemicalWorld) {
        // Step particles forwards in time
        for part in &mut self.particles {
            part.pos += part.vel * cfg.dt;
        }

        // Collide particles with walls
        for part in &mut self.particles {
            if part.pos.x < 0.0 {
                part.pos.x *= -1.0;
                part.vel.x *= -1.0;
            }

            if part.pos.y < 0.0 {
                part.pos.y *= -1.0;
                part.vel.y *= -1.0;
            }

            if part.pos.x > cfg.dimensions.x {
                part.pos.x = 2.0 * cfg.dimensions.x - cfg.dimensions.x;
                part.vel.x *= -1.0;
            }

            if part.pos.y > cfg.dimensions.y {
                part.pos.y = 2.0 * cfg.dimensions.y - cfg.dimensions.y;
                part.vel.y *= -1.0;
            }
        }


        // Build a map for the collisions

        let points: Vec<Pos2> = self.particles.iter().map(|p| p.pos).collect();
        let accel = QueryAccelerator::new(&points, cfg.particle_radius);

        for i in 0..self.particles.len() {
            for neighbor in accel.query_neighbors(&points, i, points[i]) {
                let [p1, p2] = &mut self.particles.get_disjoint_mut([i, neighbor]).unwrap();
                let m1 = chem.laws.compounds[p1.compound].mass;
                let m2 = chem.laws.compounds[p2.compound].mass;
                (p1.vel, p2.vel) = elastic_collision(m1, p1.vel, m2, p2.vel);
            }
        }
    }
}

struct SimConfig {
    dimensions: Vec2,
    dt: f32,
    particle_radius: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            dimensions: Vec2::new(100., 100.),
            dt: 1. / 60.,
            particle_radius: 1e-2,
        }
    }
}

fn elastic_collision(m1: f32, v1: Vec2, m2: f32, v2: Vec2) -> (Vec2, Vec2) {
    assert!(m1 > 0.0);
    assert!(m2 > 0.0);
    let denom = m1 + m2;
    let diff = m1 - m2;

    let v1f = (diff * v1 + 2. * m2 * v2) / denom;
    let v2f = (2. * m1 * v1 + diff * v2) / denom;
    (v1f, v2f)
}
