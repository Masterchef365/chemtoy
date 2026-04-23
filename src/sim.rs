use std::cmp::Reverse;

use crate::query_accel::QueryAccelerator;
use chemtoy_deduct::{ChemicalWorld, CompoundId};
use glam::DVec2;
use rand::seq::SliceRandom;
use rand::Rng;

pub struct Sim {
    pub particles: Vec<Particle>,
}

// kJ/K
pub const BOLTZMANN: f64 = 1.381e-23;

#[derive(Clone, Debug)]
pub struct Particle {
    pub compound: CompoundId,
    pub pos: DVec2,
    pub vel: DVec2,
    pub is_stationary: bool,
}

pub struct SimConfig {
    pub dimensions: DVec2,
    //pub max_collision_time: f64,
    pub fill_timestep: bool,
    pub gravity: f64,
    //pub speed_limit: f64,
    //pub temperature: f64,

    //pub coulomb_softening: f64,
    pub coulomb_k: f64,
    pub vanderwaals_mag: f64,

    /// Scale exponent; meters_per_unit = 10^{-scale_exp}
    pub scale_exp: f64,
    /// Time step (seconds) per frame = 10^-dt_exp
    pub max_dt_exp: f64,
    /// Maximum number of iterations before bailout
    pub max_iters: usize,
    /// Move within this percentage tolerance of the actual predicted position for collisions
    pub collision_margin: f64,
}

impl Sim {
    pub fn new() -> Self {
        Self { particles: vec![] }
    }

    /// Steps forward by as much time as possible up to cfg.dt, returning the actual dt if time was advanced. If cfg.fill_timestep is false, acts like single_step().
    pub fn step(&mut self, cfg: &SimConfig, chem: &ChemicalWorld) -> f64 {
        let mut elapsed = 0.0;
        let mut remaining_loops = cfg.max_iters;
        while elapsed < cfg.max_dt() && remaining_loops > 0 {
            remaining_loops -= 1;

            let dt = self.single_step(cfg, chem);
            elapsed += dt;

            if !cfg.fill_timestep {
                break;
            }
        }

        elapsed
    }

    pub fn single_step(&mut self, cfg: &SimConfig, chem: &ChemicalWorld) -> f64 {
        let dt = self.apply_next_event(cfg, chem).unwrap_or(cfg.max_dt());

        // Integrate acceleration due to gravity
        for part in &mut self.particles {
            part.vel.y += cfg.gravity * dt;
        }

        dt
    }

    pub fn apply_next_event(&mut self, cfg: &SimConfig, chem: &ChemicalWorld) -> Option<f64> {
        if let Some((mut dt, action)) = soonest_event(&self.particles, cfg, chem) {
            let dt_too_large = dt > cfg.max_dt();
            if dt_too_large {
                dt = dt.min(cfg.max_dt());
            } else {
                dt = dt * (1.0 - cfg.collision_margin);
            }

            // Integrate position
            for part in &mut self.particles {
                part.pos += part.vel * dt;
            }

            if !dt_too_large {
                action.apply(&mut self.particles, chem);
            }

            Some(dt)
        } else {
            None
        }
    }

    /// Returns true if a particle can be placed here
    /// TODO: slow and bad but sufficient!
    pub fn area_is_clear(&mut self, chem: &ChemicalWorld, cfg: &SimConfig, pos: DVec2) -> bool {
        let thresh_sq = (max_radius_meters(chem) * 2.0).powi(2);
        self.particles
            .iter()
            .all(|p| p.pos.distance_squared(pos) > thresh_sq)
    }
}

/// pos_diff = P2-P1
fn elastic_collision_vect(m1: f64, v1: DVec2, m2: f64, v2: DVec2, pos_diff: DVec2) -> (DVec2, DVec2) {
    assert!(m1 > 0.0);
    assert!(m2 > 0.0);

    let rvel = v2 - v1;
    let mtot = m1 + m2;

    let v2i = rvel.project_onto_normalized(pos_diff);
    let rem = rvel - v2i;

    let v1f = 2.0 * m2 * v2i / mtot;
    let v2f = (m2 - m1) * v2i / mtot;

    (v1f + v1, v2f + v1 + rem)
}

fn reflect(v1: DVec2, v2: DVec2) -> DVec2 {
    v1 - 2.0 * v1.dot(v2) * v2
}

/*
fn boundaries(particles: &mut [Particle], cfg: &SimConfig, chem: &ChemicalWorld, dt: f64) {
    // Boundaries
    for part in particles.iter_mut() {
        let comp = &chem.deriv.compound_lookup[&part.compound];
        let radius = comp.transport.radius_meters();
        for i in 0..2 {
            let margin = cfg.dimensions[i] / 1000.;

            if part.pos[i] > cfg.dimensions[i] - radius {
                if part.vel[i] > 0.0 {
                    part.vel[i] = -part.vel[i].abs();
                    part.pos[i] = cfg.dimensions[i] - radius - margin;
                }
            } else if part.pos[i] < radius {
                if part.vel[i] < 0.0 {
                    part.vel[i] = part.vel[i].abs();
                    part.pos[i] = radius + margin;
                }
            }

            part.pos[i] = part.pos[i].clamp(radius, cfg.dimensions[i] - radius);
        }
    }
}
*/

fn interact(
    particles: &mut [Particle],
    i: usize,
    j: usize,
    cfg: &SimConfig,
    chem: &ChemicalWorld,
    add_list: &mut Vec<Particle>,
    remove_list: &mut Vec<usize>,
) -> Option<Particle> {
    // Medium-range interactions
    let cmpd_i = &chem.deriv.compound_lookup[&particles[i].compound];
    let cmpd_j = &chem.deriv.compound_lookup[&particles[j].compound];

    let diff = particles[j].pos - particles[i].pos;
    let r2 = diff.length_squared();

    let d = cmpd_i.transport.radius_meters() + cmpd_j.transport.radius_meters();
    //let d2 = d.powi(2);

    //let charge = (cmpd_i.charge * cmpd_j.charge) as f64;
    //let coulomb_force = charge * (-r2 / d2).exp();

    //let vanderwalls = -(-r2 / d2).exp();

    //let force = coulomb_force * cfg.coulomb_k + vanderwalls * cfg.vanderwaals_mag;

    //let force = force * diff.normalize();

    //particles[i].vel -= force * cfg.dt();
    //particles[j].vel += force * cfg.dt();

    let r = r2.sqrt();

    // Collision
    let rvel = particles[j].vel - particles[i].vel;
    let may_collide = true;//rvel.dot(diff) < 0.0;

    if r < d && may_collide {
        // Scattering
        if particles[i].is_stationary && !particles[j].is_stationary {
            let v = reflect(particles[j].vel, diff.normalize());
            particles[j].vel = v;
        }

        if !particles[i].is_stationary && !particles[j].is_stationary {
            let (vi, vj) = elastic_collision_vect(
                cmpd_i.mass_kg,
                particles[i].vel,
                cmpd_j.mass_kg,
                particles[j].vel,
                particles[j].pos - particles[i].pos,
            );
            particles[i].vel = vi;
            particles[j].vel = vj;
        }
    }

    None
}

fn inelastic_collision(m1: f64, v1: DVec2, m2: f64, v2: DVec2) -> DVec2 {
    (m1 * v1 + m2 * v2) / (m1 + m2)
}

fn kinetic_energy(vel: DVec2, mass: f64) -> f64 {
    vel.length_squared() * mass * 0.5
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
    let ke_rel =
        (cmpd_i.mass_kg + cmpd_j.mass_kg) * (particles[i].vel - particles[j].vel).length_squared();

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
        particles[j].pos - particles[i].pos,
    );
    particles[j].vel = vel_j;
    particles[i].vel = vel_i;

    particles[i].compound = product_a.clone();

    let max_radius = cmpd_i.transport.radius_meters().max(cmpd_j.transport.radius_meters());

    let pos = particles[i].pos - particles[i].vel.normalize() * max_radius * 2.0;
    let pos2 = particles[i].pos - particles[i].vel.normalize() * max_radius * 4.0;
    particles[j].pos = pos2;

    // TODO: Compensate offset momentum here

    Some(Particle {
        compound: product_b.clone(),
        pos,
        vel: DVec2::ZERO,
        is_stationary: false,
    })
}

impl SimConfig {
    /// Meters per in-simulation unit
    pub fn meters_per_unit(&self) -> f64 {
        10_f64.powf(self.scale_exp)
    }

    /// Multiply by sim energy units to get energy per reaction in Joules
    pub fn si_per_sim_units_energy(&self) -> f64 {
        self.meters_per_unit().powi(2)
    }

    pub fn max_dt(&self) -> f64 {
        10_f64.powf(self.max_dt_exp)
    }
}

impl Default for SimConfig {
    fn default() -> Self {
        let scale_exp = -11.0;
        let dt_exp = -14.0;
        Self {
            max_iters: 1000,
            //coulomb_softening: 0.1,
            dimensions: DVec2::new(500., 500.) * 10_f64.powf(scale_exp),
            //max_collision_time: 1e-2,
            fill_timestep: true,
            gravity: 9.8,
            //speed_limit: 500.0,
            //temperature: 100., // Arbitrary
            coulomb_k: 1e3,
            vanderwaals_mag: 0.0,
            max_dt_exp: dt_exp,
            scale_exp,
            collision_margin: 0.01,
        }
    }
}

fn max_radius_meters(chem: &ChemicalWorld) -> f64 {
    chem.laws
        .species
        .iter()
        .max_by(|a, b| {
            a.transport
                .diameter_angstroms
                .partial_cmp(&b.transport.diameter_angstroms)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
        .transport
        .radius_meters()
}

#[derive(Debug)]
enum SimEvent {
    WallCollision {
        particle: usize,
        normal: usize,
    },
    ParticleCollision {
        part_a: usize,
        part_b: usize,
    }
}

fn soonest_event(particles: &[Particle], cfg: &SimConfig, chem: &ChemicalWorld) -> Option<(f64, SimEvent)> {
    let mut soonest_time = f64::MAX;
    let mut event = None;

    // Wall collisions
    for i in 0..particles.len() {
        let part = &particles[i];
        let cmpd = &chem.deriv.compound_lookup[&part.compound];
        let radius = cmpd.transport.radius_meters();

        for dim in 0..2 {
            if part.vel[dim] == 0.0 {
                continue;
            }

            let event_time = if part.vel[dim] > 0.0 {
                (cfg.dimensions[dim] - part.pos[dim] - radius) / part.vel[dim]
            } else {
                (part.pos[dim] - radius) / -part.vel[dim]
            };

            if event_time > 0.0 && event_time < soonest_time {
                soonest_time = event_time;
                event = Some(SimEvent::WallCollision { particle: i, normal: dim })
            }
        }
    }

    // Particle collisions
    for i in 0..particles.len() {
        let r_i = &chem.deriv.compound_lookup[&particles[i].compound].transport.radius_meters();

        for j in i + 1..particles.len() {
            let rel_pos = particles[j].pos - particles[i].pos;
            let rel_vel = particles[j].vel - particles[i].vel;
            let r_j = &chem.deriv.compound_lookup[&particles[j].compound].transport.radius_meters();

            if let Some(event_time) = time_of_intersection_particles(rel_pos, rel_vel, r_i + r_j) {
                if event_time > 0.0 && event_time < soonest_time {
                    soonest_time = event_time;
                    event = Some(SimEvent::ParticleCollision { part_a: i, part_b: j });
                }
            }
        }
    }

    event.map(|act| (soonest_time, act))
}

impl SimEvent {
    pub fn apply(&self, particles: &mut [Particle], chem: &ChemicalWorld) {
        match self {
            Self::WallCollision { particle, normal } => {
                particles[*particle].vel[*normal] *= -1.0;
            },
            Self::ParticleCollision { part_a: i, part_b: j } => {
                if !react_particles(particles, *i, *j, chem) {
                    scatter_particles(particles, *i, *j, chem);
                }
            },
        }
    }
}

fn time_of_intersection_particles(rel_pos: DVec2, rel_vel: DVec2, sum_radii: f64) -> Option<f64> {
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
    let mut t_min = f64::INFINITY;
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

fn scatter_particles(particles: &mut [Particle], i: usize, j: usize, chem: &ChemicalWorld) {
    let m_i = chem.deriv.compound_lookup[&particles[i].compound].mass_kg;
    let m_j = chem.deriv.compound_lookup[&particles[j].compound].mass_kg;

    let dp = (particles[j].pos - particles[i].pos).normalize_or_zero();
    let (v_i, v_j) = elastic_collision_vect(
        m_i, particles[i].vel,
        m_j, particles[j].vel,
        dp,
    );

    particles[i].vel = v_i;
    particles[j].vel = v_j;
}

fn react_particles(particles: &mut [Particle], i: usize, j: usize, chem: &ChemicalWorld) -> bool {
    false
}
