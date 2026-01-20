use std::cmp::Reverse;
use std::ops::Neg;

use chemtoy_deduct::{ChemicalWorld, CompoundId};
use crate::query_accel::QueryAccelerator;
use egui::{Pos2, Vec2};
use rand::prelude::Distribution;
use rand::seq::{IteratorRandom, SliceRandom};
use rand::Rng;

pub struct Sim {
    pub particles: Vec<Particle>,
}

#[derive(Clone, Copy)]
pub struct Particle {
    pub compound: CompoundId,
    pub pos: Pos2,
    pub vel: Vec2,
    pub is_stationary: bool,
}

pub struct SimConfig {
    pub dimensions: Vec2,
    pub dt: f32,
    pub particle_radius: f32,
    //pub max_collision_time: f32,
    pub fill_timestep: bool,
    pub gravity: f32,
    pub speed_limit: f32,
    pub temperature: f32,

    pub coulomb_softening: f32,
    pub coulomb_k: f32,
    //pub morse_alpha: f32,
    pub max_interaction_dist: f32,
    pub vanderwaals_mag: f32,
}

fn morse_force(r0: f32, k: f32, r: f32) -> f32 {
    let exp = (-k * (r - r0)).exp();
    
    2.0 * k * exp * (1.0 - exp)
}

impl Sim {
    pub fn new() -> Self {
        Self { particles: vec![] }
    }

    /// Steps forward by as much time as possible up to cfg.dt, returning the actual dt if time was advanced. If cfg.fill_timestep is false, acts like single_step().
    pub fn step(&mut self, cfg: &SimConfig, chem: &ChemicalWorld) -> f32 {
        // Arbitrary, must be larger than particle radius.
        // TODO: Tune for perf.

        // Build a map for the collisions

        let mut elapsed = 0.0;
        let mut remaining_loops = 1000;
        while elapsed < cfg.dt {
            if remaining_loops == 0 {
                break;
            }
            remaining_loops -= 1;

            let dt = self.single_step(cfg, chem);
            elapsed += dt;

            if !cfg.fill_timestep {
                break;
            }
        }

        elapsed
    }

    pub fn single_step(&mut self, cfg: &SimConfig, chem: &ChemicalWorld) -> f32 {
        let points: Vec<Pos2> = self.particles.iter().map(|p| p.pos).collect();
        let accel = QueryAccelerator::new(&points, cfg.max_interaction_dist.max(cfg.particle_radius));

        boundaries(&mut self.particles, cfg, chem, cfg.dt);

        let mut new_particles: Vec<Particle> = vec![];
        let mut removed_particles = vec![];

        let mut changed = vec![false; self.particles.len()];

        for i in 0..self.particles.len() {
            // Inter-particle forces
            let mut k = None;
            for j in accel.query_neighbors(&points, i, points[i]) {
                interact(&mut self.particles, i, j, k, cfg, chem, &mut new_particles, &mut removed_particles, &mut changed);
                // We store an extra neighbor for 3 body interactions
                k = Some(j);
            }

            // Stationary particles
            if self.particles[i].is_stationary {
                self.particles[i].vel = Vec2::ZERO;
            }

            // Gravity
            self.particles[i].vel.y += cfg.gravity * cfg.dt;

            // Time step
            let vel = self.particles[i].vel;
            self.particles[i].pos += vel * cfg.dt;
        }

        removed_particles.sort_unstable_by_key(|f| Reverse(*f));
        removed_particles.dedup();
        for i in removed_particles {
            self.particles.swap_remove(i);
        }

        self.particles.extend_from_slice(&new_particles);

        cfg.dt
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
            coulomb_softening: 0.1,
            dimensions: Vec2::new(500., 500.),
            dt: 1. / 60.,
            particle_radius: 5.0,
            //max_collision_time: 1e-2,
            fill_timestep: true,
            gravity: 9.8,
            speed_limit: 500.0,
            temperature: 100., // Arbitrary
            coulomb_k: 1e3,
            vanderwaals_mag: 1e2,
            //morse_alpha: 1.0,
            max_interaction_dist: 15.0,
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

fn elastic_collision_vect(m1: f32, v1: Vec2, m2: f32, v2: Vec2) -> (Vec2, Vec2) {
    assert!(m1 > 0.0);
    assert!(m2 > 0.0);
    let denom = m1 + m2;
    let diff = m1 - m2;

    let v1f = (diff * v1 + 2. * m2 * v2) / denom;
    let v2f = (2. * m1 * v1 - diff * v2) / denom;
    (v1f, v2f)
}

fn reflect(v1: Vec2, v2: Vec2) -> Vec2 {
    v1 - 2.0 * v1.dot(v2) * v2
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

#[derive(Clone, Copy)]
struct Intersection {
    time: f32,
    data: IntersectionData,
    index: usize,
}

#[derive(Clone, Copy)]
enum IntersectionData {
    Wall { mirrored_velocity: Vec2 },
    Particle { neighbor: usize },
}

fn boundaries(particles: &mut [Particle], cfg: &SimConfig, chem: &ChemicalWorld, dt: f32) {
    // Boundaries
    for part in particles.iter_mut() {
        for i in 0..2 {
            let margin = cfg.dimensions[i] / 1000.;

            if part.pos[i] > cfg.dimensions[i] - cfg.particle_radius {
                if part.vel[i] > 0.0 {
                    part.vel[i] = -part.vel[i].abs();
                    part.pos[i] = cfg.dimensions[i] - cfg.particle_radius - margin;
                }
            } else if part.pos[i] < cfg.particle_radius {
                if part.vel[i] < 0.0 {
                    part.vel[i] = part.vel[i].abs();
                    part.pos[i] = cfg.particle_radius + margin;
                }
            }

            part.pos[i] = part.pos[i].clamp(cfg.particle_radius, cfg.dimensions[i] - cfg.particle_radius);
        }
    }
}

fn interact(particles: &mut [Particle], i: usize, j: usize, k: Option<usize>, cfg: &SimConfig, chem: &ChemicalWorld, add_list: &mut Vec<Particle>, remove_list: &mut Vec<usize>, changed: &mut [bool]) -> Option<Particle> {
    // Medium-range interactions
    let cmpd_i = &chem.laws.compounds[particles[i].compound];
    let cmpd_j = &chem.laws.compounds[particles[j].compound];

    let diff = particles[j].pos - particles[i].pos;
    let r2 = diff.length_sq();

    let d = cfg.particle_radius * 2.0;
    let d2 = d.powi(2);

    let charge = (cmpd_i.charge * cmpd_j.charge) as f32;
    //let coulomb_force = force / diff.length_sq();
    let coulomb_force = charge * (-r2 / d2).exp();

    let vanderwalls = -(-r2 / d2).exp();

    let force = coulomb_force * cfg.coulomb_k + vanderwalls * cfg.vanderwaals_mag;

    let force = force * diff.normalized();

    particles[i].vel -= force * cfg.dt;
    particles[j].vel += force * cfg.dt;

    let r = r2.sqrt();

    // Collision
    let rvel = particles[j].vel - particles[i].vel;
    let may_collide = rvel.dot(diff) < 0.0;

    if r < d && may_collide {
        if let Some(k) = k {
            if !(changed[i] || changed[j]) && react(particles, i, j, k, cfg, chem) {
                changed[i] = true;
                changed[j] = true;
                remove_list.push(i);
                return None;
            }
        }

        if !(changed[i] || changed[j]) {
            if let Some(new_particle) = decompose(particles, i, j, cfg, chem) {
                changed[i] = true;
                changed[j] = true;
                add_list.push(new_particle);
            }
        }

        // Scattering
        if particles[i].is_stationary && !particles[j].is_stationary {
            let v = reflect(particles[j].vel, diff.normalized());
            particles[j].vel = v; 
        }

        if !particles[i].is_stationary && !particles[j].is_stationary {
            let (vi, vj) = elastic_collision_vect(cmpd_i.mass, particles[i].vel, cmpd_j.mass, particles[j].vel);
            particles[i].vel = vi; 
            particles[j].vel = vj; 
        }
    }

    None
}

fn inelastic_collision(m1: f32, v1: Vec2, m2: f32, v2: Vec2) -> Vec2 {
    (m1 * v1 + m2 * v2) / (m1 + m2)
}

fn restitution_collision(m1: f32, v1: Vec2, m2: f32, v2: Vec2, e: f32) -> (Vec2, Vec2) {
    let total_mass = m1 + m2;
    let total_momentum = m1 * v1 + m2 * v2;
    (
        (e * m2 * (v2 - v1) + total_momentum) / total_mass,
        (e * m1 * (v1 - v2) + total_momentum) / total_mass,
    )
}

fn kinetic_energy(vel: Vec2, mass: f32) -> f32 {
    vel.length_sq() * mass * 0.5
}

/// Returns true if the particle at index `i` should be deleted.
/// Particle j will become the product
/// Particle k will receive any excess kinetic energy
fn react(particles: &mut [Particle], i: usize, j: usize, k: usize, cfg: &SimConfig, chem: &ChemicalWorld) -> bool {
    let cmpd_i = &chem.laws.compounds[particles[i].compound];
    let cmpd_j = &chem.laws.compounds[particles[j].compound];
    let cmpd_k = &chem.laws.compounds[particles[k].compound];

    /*
    let total_ke_init = 
        kinetic_energy(particles[i].vel, cmpd_i.mass)
        + kinetic_energy(particles[k].vel, cmpd_k.mass)
        + kinetic_energy(particles[j].vel, cmpd_j.mass);
    */

    let ke_init = kinetic_energy(particles[i].vel, cmpd_i.mass) + kinetic_energy(particles[j].vel, cmpd_j.mass);

    let Some(product) = chem.deriv.synthesis.lookup(particles[i].compound, particles[j].compound) else { return false; };
    let new_cmpd_j = &chem.laws.compounds[product];

    let dg = new_cmpd_j.std_free_energy - cmpd_i.std_free_energy - cmpd_j.std_free_energy;
    let p = (dg / cfg.temperature).neg().exp().clamp(0.0, 1.0);
    if rand::thread_rng().gen_bool(1f64 - p as f64) {
        return false;
    }

    particles[j].vel = inelastic_collision(cmpd_i.mass, particles[i].vel, cmpd_j.mass, particles[j].vel);
    particles[j].compound = product;

    let ke_end = kinetic_energy(particles[j].vel, new_cmpd_j.mass);

    let de = ke_init - ke_end;

    let ke_k = kinetic_energy(particles[k].vel, cmpd_k.mass);

    if ke_k == 0.0 || de + ke_k <= 0.0 {
        return false;
    }

    let e = ((de + ke_k)/ke_k).sqrt();

    particles[k].vel *= e;

    /*
    let total_ke_end = 
        kinetic_energy(particles[k].vel, cmpd_k.mass)
        + kinetic_energy(particles[j].vel, new_cmpd_j.mass);
    */

    //dbg!(total_ke_end - total_ke_init);
    
    true
}

// Product i will be decomposed into particles a and b. It will be replaced with particle a, and
// Some(b) will be returned.
fn decompose(particles: &mut [Particle], i: usize, j: usize, cfg: &SimConfig, chem: &ChemicalWorld) -> Option<Particle> {
    let cmpd_i = &chem.laws.compounds[particles[i].compound];
    let cmpd_j = &chem.laws.compounds[particles[j].compound];

    let productsets = &chem.deriv.decompositions[&particles[i].compound];
    let mut rng = rand::thread_rng();
    let productset = productsets.products.iter().filter(|x| x.count() == 2).choose(&mut rng)?;
    let mut compounds = productset.compounds.keys().copied();
    let product_a = compounds.next().unwrap();
    let product_b = compounds.next().unwrap_or(product_a);

    let cmpd_a = &chem.laws.compounds[product_a];
    let cmpd_b = &chem.laws.compounds[product_a];

    let dg = cmpd_a.std_free_energy + cmpd_b.std_free_energy - cmpd_i.std_free_energy;

    let p = (dg / cfg.temperature).neg().exp().clamp(0.0, 1.0);
    if rand::thread_rng().gen_bool(1f64 - p as f64) {
        return None;
    }

    let (vel_i, vel_j) = elastic_collision_vect(cmpd_i.mass, particles[i].vel, cmpd_j.mass, particles[j].vel);
    particles[j].vel = vel_j;
    particles[i].vel = vel_i;

    particles[i].compound = product_a;

    let pos = particles[i].pos - particles[i].vel.normalized() * cfg.particle_radius * 2.0;
    let pos2 = particles[i].pos - particles[i].vel.normalized() * cfg.particle_radius * 4.0;
    particles[j].pos = pos2;

    Some(Particle { compound: product_b, pos, vel: Vec2::ZERO, is_stationary: false })
}
