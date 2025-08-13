use std::collections::HashMap;

use egui::{Color32, DragValue, Pos2, Rect, Stroke, Vec2};
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
    scene_rect: Rect,
    draw_compound: CompoundId,
    paused: bool,
    slowdown: usize,
    frame_count: usize,
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
            Compound::new(
                "OH⁻",
                -1,
                -138.698,
                &[(hydrogen, 1), (oxygen, 1)],
                &elements,
            ),
            Compound::new("H₂O", 0, -228.582, &[(hydrogen, 2), (oxygen, 1)], &elements),
            Compound::new("H₂O", 0, -228.582, &[(hydrogen, 2), (oxygen, 1)], &elements),
            Compound::new(
                "H₂O₂",
                0,
                -105.445,
                &[(hydrogen, 2), (oxygen, 2)],
                &elements,
            ),
        ]);

        let chem = ChemicalWorld::from_laws(Laws {
            elements,
            compounds,
        });

        let sim = Sim::new();

        let draw_compound = chem.laws.compounds.enumerate().next().unwrap().0;

        Self {
            slowdown: 1,
            frame_count: 0,
            draw_compound,
            chem,
            sim,
            cfg: SimConfig::default(),
            scene_rect: Rect::ZERO,
            paused: false,
        }
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
        let mut single_step = false;

        egui::SidePanel::left("cfg").show(ctx, |ui| {
            ui.group(|ui| {
                ui.strong("Time");
                let text = if self.paused { "Paused" } else { "Running" };
                ui.horizontal(|ui| {
                    ui.label("Slowdown: ");
                    ui.add(DragValue::new(&mut self.slowdown));
                });
                self.paused ^= ui.button(text).clicked();
                single_step |= ui.button("Single step").clicked();
                ui.checkbox(&mut self.cfg.fill_timestep, "Fill timestep");
            });

            ui.group(|ui| {
                ui.strong("Simulation");
                if ui.button("Reset").clicked() {
                    self.sim = Sim::new();
                }

                ui.horizontal(|ui| {
                    ui.label("Δt: ");
                    ui.add(
                        DragValue::new(&mut self.cfg.dt)
                            .speed(1e-3)
                            .range(0.0..=f32::MAX)
                            .suffix(" units/step"),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Dimensions: ");
                    ui.add(DragValue::new(&mut self.cfg.dimensions.x));
                    ui.label("x");
                    ui.add(DragValue::new(&mut self.cfg.dimensions.y));
                });
                ui.horizontal(|ui| {
                    ui.label("Particle radius: ");
                    ui.add(DragValue::new(&mut self.cfg.particle_radius).speed(1e-2));
                });
                ui.horizontal(|ui| {
                    ui.label("Max collision time: ");
                    ui.add(DragValue::new(&mut self.cfg.max_collision_time).speed(1e-2));
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::Scene::new()
                .zoom_range(1.0..=100.0)
                .show(ui, &mut self.scene_rect, |ui| {
                    let (rect, resp) =
                        ui.allocate_exact_size(self.cfg.dimensions, egui::Sense::click_and_drag());
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

                    for particle in &self.sim.particles {
                        ui.painter().circle_filled(
                            particle.pos + rect.min.to_vec2(),
                            self.cfg.particle_radius,
                            Color32::GRAY,
                        );
                        let compound = &self.chem.laws.compounds[particle.compound];
                        ui.painter().text(
                            particle.pos,
                            egui::Align2([egui::Align::Center; 2]),
                            &compound.name,
                            Default::default(),
                            Color32::WHITE,
                        );

                        ui.painter().arrow(
                            particle.pos + rect.min.to_vec2(),
                            particle.vel,
                            Stroke::new(1.0, Color32::RED),
                        );
                    }

                    //if let Some(drag_pos) = resp.interact_pointer_pos() {
                    if let Some(interact_pos) = resp.interact_pointer_pos() {
                        if resp.clicked() {
                            self.sim.particles.push(Particle {
                                compound: self.draw_compound,
                                pos: interact_pos - rect.min.to_vec2(),
                                vel: resp.drag_delta(),
                            });
                        }
                    }
                });
        });

        if !self.paused || single_step {
            if self.frame_count % self.slowdown == 0 {
                self.sim.step(&self.cfg, &self.chem);
            }
            ctx.request_repaint();
            self.frame_count += 1;
        }
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
        Self { particles: vec![] }
    }

    pub fn step(&mut self, cfg: &SimConfig, chem: &ChemicalWorld) {
        // Build a map for the collisions
        let points: Vec<Pos2> = self.particles.iter().map(|p| p.pos).collect();
        // Arbitrary, must be larger than particle radius.
        // TODO: Tune for perf.
        //let accel = QueryAccelerator::new(&points, cfg.particle_radius * 100.0);

        let mut elapsed = 0.0;
        while elapsed < cfg.dt {
            let mut min_dt = cfg.dt;

            let mut min_particle_indices = None;
            let mut min_boundary_vel_idx = None;
            for i in 0..self.particles.len() {
                // Check time of intersection with neighbors
                //for neighbor in accel.query_neighbors(&points, i, points[i]) {
                for neighbor in i + 1..self.particles.len() {
                    let [p1, p2] = self.particles.get_disjoint_mut([i, neighbor]).unwrap();

                    // TODO: Cache these intersections AND evict the cache ...
                    if let Some(intersection_dt) = time_of_intersection_particles(
                        p2.pos - p1.pos,
                        p2.vel - p1.vel,
                        cfg.particle_radius * 2.0,
                    ) {
                        assert!(intersection_dt >= 0.0);
                        if intersection_dt < min_dt {
                            min_dt = intersection_dt;
                            min_particle_indices = Some((i, neighbor));
                            min_boundary_vel_idx = None;
                        }
                    }
                }

                let particle = &self.particles[i];
                if let Some((boundary_dt, new_vel)) = time_of_intersection_boundary(
                    particle.pos,
                    particle.vel,
                    cfg.dimensions,
                    cfg.particle_radius,
                ) {
                    if boundary_dt < min_dt {
                        min_boundary_vel_idx = Some((i, new_vel));
                        min_particle_indices = None;
                        min_dt = boundary_dt;
                    }
                }
            }

            if min_dt < cfg.max_collision_time {
                // Interact the particles. max_collision_time should be small enough not to neglect any
                // external forces(!)
                if let Some((i, vel)) = min_boundary_vel_idx {
                    self.particles[i].vel = vel;
                }

                if let Some((i, neighbor)) = min_particle_indices {
                    let [p1, p2] = self.particles.get_disjoint_mut([i, neighbor]).unwrap();
                    let m1 = chem.laws.compounds[p1.compound].mass;
                    let m2 = chem.laws.compounds[p2.compound].mass;
                    (p1.vel, p2.vel) = elastic_collision(m1, p1.vel, m2, p2.vel);
                }
            } else {
                // Cowardly move halfway to the goal
                let dt = min_dt / 2.0;
                timestep_particles(&mut self.particles, dt);
                for particle in &mut self.particles {
                    particle.vel.y += 9.8 * 1e-1; // pixels/frame^2
                }
                elapsed += dt;
            }

            if !cfg.fill_timestep {
                break;
            }
        }

        /*
        // Do collisions
        for i in 0..self.particles.len() {
            for neighbor in accel.query_neighbors(&points, i, points[i]) {
                            }
        }
        */

        // Add gravity
    }
}

struct SimConfig {
    dimensions: Vec2,
    dt: f32,
    particle_radius: f32,
    max_collision_time: f32,
    fill_timestep: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            dimensions: Vec2::new(100., 100.),
            dt: 1. / 60.,
            particle_radius: 5.0,
            max_collision_time: 1e-2,
            fill_timestep: true,
        }
    }
}

fn elastic_collision(m1: f32, v1: Vec2, m2: f32, v2: Vec2) -> (Vec2, Vec2) {
    assert!(m1 > 0.0);
    assert!(m2 > 0.0);
    let denom = m1 + m2;
    let diff = m1 - m2;

    let v1f = (diff * v1 + 2. * m2 * v2) / denom;
    let v2f = (2. * m1 * v1 - diff * v2) / denom;
    (v1f, v2f)
}

fn cross2d(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

// WARNING: Got lazy and asked a GPT
fn time_of_intersection_particles(rel_pos: Vec2, rel_vel: Vec2, sum_radii: f32) -> Option<f32> {
    // Intersection means |rel_pos + t * rel_vel| == 0
    // => (rel_pos + t*rel_vel)·(rel_pos + t*rel_vel) == 0
    let a = rel_vel.dot(rel_vel);
    let b = 2.0 * rel_pos.dot(rel_vel);
    let c = rel_pos.dot(rel_pos) - sum_radii.powi(2);

    if a == 0.0 {
        // No relative motion
        if c == 0.0 {
            return Some(0.0); // Already intersecting
        }
        return None; // Never intersect
    }

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None; // No real solution
    }

    let sqrt_d = discriminant.sqrt();
    let t1 = (-b - sqrt_d) / (2.0 * a);
    let t2 = (-b + sqrt_d) / (2.0 * a);

    // We care about the earliest non-negative intersection
    let mut t_min = f32::INFINITY;
    if t1 >= 0.0 {
        t_min = t_min.min(t1);
    }
    if t2 >= 0.0 {
        t_min = t_min.min(t2);
    }

    if t_min.is_infinite() {
        None
    } else {
        Some(t_min)
    }
}

/// Step particles forwards in time
fn timestep_particles(particles: &mut [Particle], dt: f32) {
    for part in particles {
        part.pos += part.vel * dt;
    }
}

fn particle_collisions(particles: &mut [Particle], cfg: &SimConfig) {
    // Collide particles with walls
    for part in particles {
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
}

/// Returns time of intersection and the reflected velocity vector.
fn time_of_intersection_boundary(
    pos: Pos2,
    vel: Vec2,
    dimensions: Vec2,
    radius: f32,
) -> Option<(f32, Vec2)> {
    fn intersect(x: f32, vel: f32, width: f32, radius: f32) -> Option<f32> {
        if vel == 0.0 {
            return None;
        }

        if vel > 0.0 {
            Some((width - x - radius) / vel)
        } else {
            Some((x - radius) / -vel)
        }
    }

    let xtime = intersect(pos.x, vel.x, dimensions.x, radius);
    let ytime = intersect(pos.y, vel.y, dimensions.y, radius);

    if let Some(xtime) = xtime {
        //assert!(xtime >= 0.0);
        if xtime < 0.0 {
            eprintln!("WARNING: xtime = {xtime}");
        }
    }

    if let Some(ytime) = ytime {
        //assert!(ytime >= 0.0);
        if ytime < 0.0 {
            eprintln!("WARNING: ytime = {ytime}");
        }
    }

    match (xtime, ytime) {
        (Some(xtime), Some(ytime)) if xtime < ytime => Some((xtime, Vec2::new(-vel.x, vel.y))),
        (Some(xtime), None) => Some((xtime, Vec2::new(-vel.x, vel.y))),
        (None, Some(ytime)) | (Some(_), Some(ytime)) => Some((ytime, Vec2::new(vel.x, -vel.y))),
        (None, None) => None,
    }
}
