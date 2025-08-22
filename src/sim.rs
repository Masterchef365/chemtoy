use crate::laws::{ChemicalWorld, CompoundId, Laws};
use crate::query_accel::QueryAccelerator;
use egui::{Pos2, Vec2};
use rand::prelude::Distribution;
use rand::Rng;

pub struct Sim {
    pub particles: Vec<Particle>,
}

#[derive(Clone, Copy)]
pub struct Particle {
    pub compound: CompoundId,
    pub pos: Pos2,
    pub vel: Vec2,
    /// Should this particle decompose ASAP, and with how much energy?
    pub to_decompose: Option<f32>,
}

pub struct SimConfig {
    pub dimensions: Vec2,
    pub dt: f32,
    pub particle_radius: f32,
    pub max_collision_time: f32,
    pub fill_timestep: bool,
    pub gravity: f32,
    pub speed_limit: f32,
    pub ke_scale_factor: f32,
}

impl Sim {
    pub fn new() -> Self {
        Self { particles: vec![] }
    }

    pub fn step(&mut self, cfg: &SimConfig, chem: &ChemicalWorld) {
        // Arbitrary, must be larger than particle radius.
        // TODO: Tune for perf.

        // Build a map for the collisions

        let mut elapsed = 0.0;
        let mut remaining_loops = 1000;
        'timeloop: while elapsed < cfg.dt {
            if remaining_loops == 0 {
                break;
            }
            remaining_loops -= 1;

            let speed_limit_sq = cfg.speed_limit.powi(2);
            for particle in &mut self.particles {
                if particle.vel.length_sq() > speed_limit_sq {
                    //eprintln!("OVER SPEED LIMIT {:?}", particle.vel);
                    particle.vel = particle.vel.normalized() * cfg.speed_limit;
                }
            }

            let points: Vec<Pos2> = self.particles.iter().map(|p| p.pos).collect();
            let accel = QueryAccelerator::new(&points, cfg.speed_limit * 2.0);

            let mut min_dt = cfg.dt;

            let mut min_particle_indices = None;
            let mut min_boundary_vel_idx = None;
            for i in 0..self.particles.len() {
                // Check time of intersection with neighbors
                for neighbor in accel.query_neighbors_fast(i, points[i]) {
                    //for neighbor in i + 1..self.particles.len() {
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
                    let c1 = &chem.laws.compounds[p1.compound];
                    let c2 = &chem.laws.compounds[p2.compound];

                    let m1 = c1.mass;
                    let m2 = c2.mass;

                    let rel_pos = p2.pos - p1.pos;
                    let rel_dir = rel_pos.normalized();
                    let rel_vel = p2.vel - p1.vel;

                    // Velocity at point of contact
                    let vel_component = rel_vel.dot(rel_dir).abs();

                    let total_mass = m1 + m2;
                    //const KG_PER_DALTON: f32 = 1.6605390e-27;

                    let kinetic_energy = vel_component.powi(2) * total_mass / 2.0;

                    if kinetic_energy * cfg.ke_scale_factor > 500.0 {
                        p1.to_decompose = Some(kinetic_energy);
                        p2.to_decompose = Some(kinetic_energy);
                    }

                    //let (v1, v2) = elastic_collision(m1, , m2, 0.0);
                    p2.vel += rel_dir * (vel_component * 2.0 * m1 / total_mass);
                    p1.vel += -rel_dir * (vel_component * 2.0 * m2 / total_mass);
                }
            } else {
                // Cowardly move halfway to the goal
                let dt = min_dt * 0.9;
                timestep_particles(&mut self.particles, dt);
                for particle in &mut self.particles {
                    particle.vel.y += cfg.gravity * dt; // pixels/frame^2
                }
                if remaining_loops == 0 {
                    dbg!(elapsed, dt);
                }
                elapsed += dt;
            }

            // Decompose one particle if possible
            let margin = 1e-2;

            'particles: for i in 0..self.particles.len() {
                if let Some(kinetic_energy) = self.particles[i].to_decompose {
                    //let compound_id = self.particles[i].compound;
                    //let compound = &chem.laws.compounds[compound_id];

                    let all_products = &chem.deriv.decompositions[&self.particles[i].compound];

                    let threshold_energy = kinetic_energy * cfg.ke_scale_factor;
                    let Some(last_product_idx) = all_products.nearest_energy(threshold_energy) else {
                        continue 'particles;
                    };

                    let product_idx = rand::thread_rng().gen_range(0..=last_product_idx);

                    let particle = self.particles[i];
                    let products = &all_products.0[product_idx];

                    let mut children: Vec<Particle> = products
                        .compounds
                        .iter()
                        .map(|(&compound, &n)| {
                            (0..n).map(move |_| Particle {
                                compound,
                                ..particle
                            })
                        })
                        .flatten()
                        .collect();

                    // Is anything nearby? Then we can't split.
                    for neighbor in accel.query_neighbors_fast(i, points[i]) {
                        if neighbor == i {
                            continue;
                        }

                        let distance = self.particles[neighbor].pos.distance(self.particles[i].pos);
                        if distance < (cfg.particle_radius + margin) * 2.0 * (children.len() as f32) {
                            continue 'particles;
                        }
                    }

                    self.particles[i].to_decompose = None;

                    // Determine where to put the new particles
                    let mut direction = self.particles[i].vel;
                    if direction.length_sq() == 0.0 {
                        direction = Vec2::Y;
                    } else {
                        direction = direction.normalized();
                    }

                    let direction = direction.rot90();
                    for (idx, particle) in children.iter_mut().enumerate() {
                        let i = idx as f32 * 2.0 - 1.0;
                        particle.pos += direction * i * cfg.particle_radius * 2.0;
                    }

                    if let Some((first, xs)) = children.split_first() {
                        self.particles[i] = *first;
                        for particle in xs {
                            self.particles.push(*particle);
                        }
                    }

                    continue 'timeloop;
                }
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

    /// Returns true if a particle can be placed here
    /// TODO: slow!
    pub fn area_is_clear(&mut self, cfg: &SimConfig, pos: Pos2) -> bool {
        let thresh_sq = (cfg.particle_radius * 2.0).powi(2);
        self.particles
            .iter()
            .all(|p| p.pos.distance_sq(pos) > thresh_sq)
    }
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            dimensions: Vec2::new(500., 500.),
            dt: 1. / 60.,
            particle_radius: 5.0,
            max_collision_time: 1e-2,
            fill_timestep: true,
            gravity: 9.8,
            speed_limit: 500.0,
            ke_scale_factor: 1.2432348, // Arbitrary
        }
    }
}

fn elastic_collision(m1: f32, v1: f32, m2: f32, v2: f32) -> (f32, f32) {
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
            //eprintln!("WARNING: xtime = {xtime}");
        }
    }

    if let Some(ytime) = ytime {
        //assert!(ytime >= 0.0);
        if ytime < 0.0 {
            //eprintln!("WARNING: ytime = {ytime}");
        }
    }

    match (xtime, ytime) {
        (Some(xtime), Some(ytime)) if xtime < ytime => Some((xtime, Vec2::new(-vel.x, vel.y))),
        (Some(xtime), None) => Some((xtime, Vec2::new(-vel.x, vel.y))),
        (None, Some(ytime)) | (Some(_), Some(ytime)) => Some((ytime, Vec2::new(vel.x, -vel.y))),
        (None, None) => None,
    }
}
