use std::cmp::Reverse;

use crate::query_accel::QueryAccelerator;
use crate::MOL;
use chemtoy_deduct::{ChemicalWorld, Compound, CompoundId, Synthesis, SynthesisOutputs};
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
                action.apply(&mut self.particles, chem, cfg);
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
fn elastic_collision_vect(
    m1: f64,
    v1: DVec2,
    m2: f64,
    v2: DVec2,
    pos_diff: DVec2,
) -> (DVec2, DVec2) {
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
    WallCollision { particle: usize, normal: usize },
    ParticleCollision { part_a: usize, part_b: usize },
}

fn soonest_event(
    particles: &[Particle],
    cfg: &SimConfig,
    chem: &ChemicalWorld,
) -> Option<(f64, SimEvent)> {
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
                event = Some(SimEvent::WallCollision {
                    particle: i,
                    normal: dim,
                })
            }
        }
    }

    // Particle collisions
    for i in 0..particles.len() {
        let r_i = &chem.deriv.compound_lookup[&particles[i].compound]
            .transport
            .radius_meters();

        for j in i + 1..particles.len() {
            let rel_pos = particles[j].pos - particles[i].pos;
            let rel_vel = particles[j].vel - particles[i].vel;
            let r_j = &chem.deriv.compound_lookup[&particles[j].compound]
                .transport
                .radius_meters();

            if let Some(event_time) = time_of_intersection_particles(rel_pos, rel_vel, r_i + r_j) {
                if event_time > 0.0 && event_time < soonest_time {
                    soonest_time = event_time;
                    event = Some(SimEvent::ParticleCollision {
                        part_a: i,
                        part_b: j,
                    });
                }
            }
        }
    }

    event.map(|act| (soonest_time, act))
}

impl SimEvent {
    pub fn apply(&self, particles: &mut Vec<Particle>, chem: &ChemicalWorld, cfg: &SimConfig) {
        match self {
            Self::WallCollision { particle, normal } => {
                particles[*particle].vel[*normal] *= -1.0;
            }
            Self::ParticleCollision {
                part_a: i,
                part_b: j,
            } => {
                if let Some(delta_e) = react_particles(particles, *i, *j, chem, cfg) {
                    adjust_global_energy(particles, chem, delta_e);
                } else {
                    scatter_particles(particles, *i, *j, chem);
                }
            }
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
    let (v_i, v_j) = elastic_collision_vect(m_i, particles[i].vel, m_j, particles[j].vel, dp);

    particles[i].vel = v_i;
    particles[j].vel = v_j;
}

/// Returns Some(delta E) if a reaction occured
fn react_particles(
    particles: &mut Vec<Particle>,
    i: usize,
    j: usize,
    chem: &ChemicalWorld,
    cfg: &SimConfig,
) -> Option<f64> {
    let synth = &chem
        .deriv
        .synthesis
        .get(&(particles[i].compound.clone(), particles[j].compound.clone()))?;

    match &synth.products {
        SynthesisOutputs::Single(product) => {
            if !exceeds_rxn_barrier(particles, i, j, &synth, chem) {
                return None;
            }

            let midpt = (particles[i].pos + particles[j].pos) / 2.0;

            let Some(new_pos) =
                smart_insert_particle(particles, product.clone(), chem, cfg, midpt, &[i, j])
            else {
                return None;
            };

            let m_i = chem.deriv.compound_lookup[&particles[i].compound].mass_kg;
            let m_j = chem.deriv.compound_lookup[&particles[j].compound].mass_kg;

            let ke_init = (m_i * particles[i].vel.length_squared() + m_j * particles[j].vel.length_squared()) / 2.0;

            let vel = inelastic_collision(m_i, particles[i].vel, m_j, particles[j].vel);

            let ke_final = (m_i + m_j) * vel.length_squared() / 2.0;

            particles.swap_remove(j);
            particles.swap_remove(i);
            particles.push(Particle {
                compound: product.clone(),
                pos: new_pos,
                vel,
                is_stationary: false,
            });

            let delta_g = synth.activation_energy.delta_g / MOL;

            Some(ke_final - ke_init - delta_g)
        }
        SynthesisOutputs::Exchange(k, l) => {
            /*
            let (k_vel, l_vel, delta_e) = exchange_reaction(particles, i, j, k.clone(), l.clone(), &synth, chem)?;

            particles[i].vel = k_vel;
            particles[i].vel = k_vel;

            Some(delta_e)
            */
            None
        }
    }
}

// Returns Some(motion_vector, delta E) if the particles can be fused, where delta E
// is the amount of energy lost to the environment
fn exceeds_rxn_barrier(
    particles: &mut Vec<Particle>,
    i: usize,
    j: usize,
    synth: &Synthesis,
    chem: &ChemicalWorld,
) -> bool {
    let cmpd_i = &chem.deriv.compound_lookup[&particles[i].compound];
    let cmpd_j = &chem.deriv.compound_lookup[&particles[j].compound];

    let ke_i = cmpd_i.mass_kg * particles[i].vel.length_squared() / 2.0;
    let ke_j = cmpd_j.mass_kg * particles[j].vel.length_squared() / 2.0;

    let initial_ke = ke_i + ke_j;

    let e_a_per_particle = synth.activation_energy.e_a / MOL;

    initial_ke >= e_a_per_particle
}

fn inelastic_collision(m1: f64, v1: DVec2, m2: f64, v2: DVec2) -> DVec2 {
    (m1 * v1 + m2 * v2) / (m1 + m2)
}

fn adjust_global_energy(particles: &mut [Particle], chem: &ChemicalWorld, delta_e: f64) {
    let total_ke = particles
        .iter()
        .map(|part| chem.deriv.compound_lookup[&part.compound].mass_kg * part.vel.length_squared())
        .sum::<f64>()
        / 2.0;

    if total_ke + delta_e <= 0.0 {
        return;
    }

    let vel_frac = ((total_ke + delta_e) / total_ke).sqrt();

    for part in particles {
        part.vel *= vel_frac;
    }
}

/// Returns a position for the new particle if there is one, preferring positions close to origin
fn smart_insert_particle(
    particles: &mut Vec<Particle>,
    cmpd: CompoundId,
    chem: &ChemicalWorld,
    cfg: &SimConfig,
    mut proposed_pos: DVec2,
    mask: &[usize],
) -> Option<DVec2> {
    let radius = chem.deriv.compound_lookup[&cmpd].transport.radius_meters();

    if particles.is_empty() {
        return Some(proposed_pos);
    }

    const N_RETRIES: usize = 3;

    'proposals: for _ in 0..N_RETRIES {
        // Stay within boundaries
        proposed_pos = proposed_pos.clamp(DVec2::splat(radius), cfg.dimensions - radius);

        // Move away from overlapping particles
        'particles: for (idx, other_particle) in particles.iter_mut().enumerate() {
            if mask.contains(&idx) {
                continue 'particles;
            }

            let other_radius = chem.deriv.compound_lookup[&other_particle.compound]
                .transport
                .radius_meters();

            let dist = other_particle.pos.distance(proposed_pos);
            let total_radii = radius + other_radius;
            if dist <= total_radii {
                proposed_pos +=
                    (proposed_pos - other_particle.pos).normalize_or_zero() * total_radii;
                continue 'proposals;
            }
        }

        return Some(proposed_pos);
    }

    None
}
