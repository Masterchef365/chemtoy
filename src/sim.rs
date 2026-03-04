use std::cmp::Reverse;
use std::ops::Neg;

use crate::query_accel::QueryAccelerator;
use chemtoy_deduct::{ChemicalWorld, CompoundId};
use egui::{Pos2, Vec2};
use rand::prelude::Distribution;
use rand::seq::{IteratorRandom, SliceRandom};
use rand::Rng;

pub struct Sim {
    pub particles: Vec<Particle>,
}

// kJ/K
pub const BOLTZMANN: f64 = 1.381e-23;

#[derive(Clone)]
pub struct Particle {
    pub compound: CompoundId,
    pub pos: Pos2,
    pub vel: Vec2,
    pub is_stationary: bool,
}

pub struct SimConfig {
    pub dimensions: Vec2,
    /// Time step (seconds) per frame
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

    /// Scale exponent; meters_per_unit = 10^{-scale_exp}
    pub scale_exp: f32,
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
        let accel =
            QueryAccelerator::new(&points, cfg.max_interaction_dist.max(cfg.particle_radius));

        boundaries(&mut self.particles, cfg, chem, cfg.dt);

        let mut new_particles: Vec<Particle> = vec![];
        let mut removed_particles = vec![];

        let mut changed = vec![false; self.particles.len()];

        for i in 0..self.particles.len() {
            // Inter-particle forces
            let mut k = None;
            for j in accel.query_neighbors(&points, i, points[i]) {
                interact(
                    &mut self.particles,
                    i,
                    j,
                    k,
                    cfg,
                    chem,
                    &mut new_particles,
                    &mut removed_particles,
                    &mut changed,
                );
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

            part.pos[i] =
                part.pos[i].clamp(cfg.particle_radius, cfg.dimensions[i] - cfg.particle_radius);
        }
    }
}

fn interact(
    particles: &mut [Particle],
    i: usize,
    j: usize,
    k: Option<usize>,
    cfg: &SimConfig,
    chem: &ChemicalWorld,
    add_list: &mut Vec<Particle>,
    remove_list: &mut Vec<usize>,
    changed: &mut [bool],
) -> Option<Particle> {
    // Medium-range interactions
    let cmpd_i = &chem.deriv.compound_lookup[&particles[i].compound];
    let cmpd_j = &chem.deriv.compound_lookup[&particles[j].compound];

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
            let (vi, vj) = elastic_collision_vect(
                cmpd_i.mass_kg,
                particles[i].vel,
                cmpd_j.mass_kg,
                particles[j].vel,
            );
            particles[i].vel = vi;
            particles[j].vel = vj;
        }
    }

    None
}

fn inelastic_collision(m1: f32, v1: Vec2, m2: f32, v2: Vec2) -> Vec2 {
    (m1 * v1 + m2 * v2) / (m1 + m2)
}

fn kinetic_energy(vel: Vec2, mass: f32) -> f32 {
    vel.length_sq() * mass * 0.5
}

/// Returns true if the particle at index `i` should be deleted.
/// Particle j will become the product
/// Particle k will receive any excess kinetic energy
fn react(
    particles: &mut [Particle],
    i: usize,
    j: usize,
    k: usize,
    cfg: &SimConfig,
    chem: &ChemicalWorld,
) -> bool {
    let cmpd_i = &chem.deriv.compound_lookup[&particles[i].compound];
    let cmpd_j = &chem.deriv.compound_lookup[&particles[j].compound];
    let cmpd_k = &chem.deriv.compound_lookup[&particles[k].compound];

    /*
    let total_ke_init =
        kinetic_energy(particles[i].vel, cmpd_i.mass)
        + kinetic_energy(particles[k].vel, cmpd_k.mass)
        + kinetic_energy(particles[j].vel, cmpd_j.mass);
    */

    let ke_init = kinetic_energy(particles[i].vel, cmpd_i.mass_kg)
        + kinetic_energy(particles[j].vel, cmpd_j.mass_kg);

    let Some(products) = chem
        .deriv
        .synthesis
        .get(&(particles[i].compound.clone(), particles[j].compound.clone()))
    else {
        return false;
    };

    let e_a = products.activation_energy.e_a * cfg.si_per_sim_units_energy();
    let ke_rel = (cmpd_i.mass_kg + cmpd_j.mass_kg) * (particles[i].vel - particles[j].vel).length_sq();

    if ke_rel * cfg.si_per_sim_units_energy() < e_a {
        return false;
    }

    let mut product_iter = products.products.iter();
    let product_j = product_iter.next().unwrap();
    let product_i = product_iter.next();

    let new_cmpd_j = &chem.deriv.compound_lookup[&product_j];
    let new_cmpd_i = product_i.map(|i| &chem.deriv.compound_lookup[i]);

    particles[j].vel = inelastic_collision(
        cmpd_i.mass_kg,
        particles[i].vel,
        cmpd_j.mass_kg,
        particles[j].vel,
    );
    particles[j].compound = new_cmpd_j.smiles.clone();

    if let Some(i_cmpd) = new_cmpd_i {
        particles[i].compound = i_cmpd.smiles.clone();
    }

    let ke_end = kinetic_energy(particles[j].vel, new_cmpd_j.mass_kg)
        + new_cmpd_i
            .map(|cmpd| kinetic_energy(particles[i].vel, cmpd.mass_kg))
            .unwrap_or(0.0);

    let de = ke_init - ke_end + products.activation_energy.delta_g * cfg.si_per_sim_units_energy();

    let ke_k = kinetic_energy(particles[k].vel, cmpd_k.mass_kg);

    if ke_k == 0.0 || de + ke_k <= 0.0 {
        return false;
    }

    let e = ((de + ke_k) / ke_k).sqrt();

    particles[k].vel *= e;

    /*
    let total_ke_end =
        kinetic_energy(particles[k].vel, cmpd_k.mass)
        + kinetic_energy(particles[j].vel, new_cmpd_j.mass);
    */

    //dbg!(total_ke_end - total_ke_init);

    new_cmpd_i.is_none()
}

// Product i will be decomposed into particles a and b. It will be replaced with particle a, and
// Some(b) will be returned.
fn decompose(
    particles: &mut [Particle],
    i: usize,
    j: usize,
    cfg: &SimConfig,
    chem: &ChemicalWorld,
) -> Option<Particle> {
    let cmpd_i = &chem.deriv.compound_lookup[&particles[i].compound];
    let cmpd_j = &chem.deriv.compound_lookup[&particles[j].compound];

    let productsets = &chem.deriv.decompositions.get(&particles[i].compound)?;
    let mut rng = rand::thread_rng();
    let productset = productsets.choose(&mut rng)?;
    let compounds = &productset.products;
    let product_a = compounds.get(0)?;
    let product_b = compounds.get(1)?;

    let (vel_i, vel_j) = elastic_collision_vect(
        cmpd_i.mass_kg,
        particles[i].vel,
        cmpd_j.mass_kg,
        particles[j].vel,
    );
    particles[j].vel = vel_j;
    particles[i].vel = vel_i;

    particles[i].compound = product_a.clone();

    let pos = particles[i].pos - particles[i].vel.normalized() * cfg.particle_radius * 2.0;
    let pos2 = particles[i].pos - particles[i].vel.normalized() * cfg.particle_radius * 4.0;
    particles[j].pos = pos2;

    Some(Particle {
        compound: product_b.clone(),
        pos,
        vel: Vec2::ZERO,
        is_stationary: false,
    })
}

impl SimConfig {
    /// Meters per in-simulation unit
    pub fn meters_per_unit(&self) -> f32 {
        10_f32.powf(self.scale_exp)
    }

    /// Multiply by sim energy units to get energy per reaction in Joules
    pub fn si_per_sim_units_energy(&self) -> f32 {
        self.meters_per_unit().powi(2)
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
            scale_exp: -13.0,
        }
    }
}
