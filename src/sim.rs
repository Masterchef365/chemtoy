use chemtoy_deduct::{ChemicalWorld, CompoundId};
use crate::query_accel::QueryAccelerator;
use egui::{Pos2, Vec2};
use rand::prelude::Distribution;
use rand::seq::SliceRandom;
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
    //pub max_collision_time: f32,
    pub fill_timestep: bool,
    pub gravity: f32,
    pub speed_limit: f32,
    pub kjmol_per_sim_energy: f32,

    pub coulomb_softening: f32,
    pub coulomb_k: f32,
    pub morse_alpha: f32,
    pub morse_radius: f32,
    pub morse_mag: f32,
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

            if let Some(dt) = self.single_step(cfg, chem) {
                elapsed += dt;
            }

            if !cfg.fill_timestep {
                break;
            }
        }

        elapsed
    }

    pub fn single_step(&mut self, cfg: &SimConfig, chem: &ChemicalWorld) -> Option<f32> {

        let points: Vec<Pos2> = self.particles.iter().map(|v| v.pos).collect();
        let accel = QueryAccelerator::new(points.as_slice(), cfg.particle_radius * 2.0);

        /*
        if self.try_collide(cfg, chem, &accel) {
            return None;
        }

        if self.try_decompose(cfg, chem, &accel) {
            return None;
        }
        */

        /*
        let max_dist = cfg.particle_radius / 2.0;

        let mut max_rel_vel: f32 = 0.0;
        for i in 0..self.particles.len() {
            for j in i + 1..self.particles.len() {
                let rvel = (self.particles[i].vel - self.particles[j].vel).length_sq();
                max_rel_vel = max_rel_vel.max(rvel);
            }
        }
        max_rel_vel = max_rel_vel.sqrt();

        let max_dt = max_dist / max_rel_vel;

        let dt = cfg.dt.min(max_dt);
        */
        let mut dt = cfg.dt;

        for i in 0..self.particles.len() {
            for neighbor in accel.query_neighbors_fast(i, points[i]) {
                let us = &self.particles[i];
                let neigh = &self.particles[neighbor];

                let rvel = us.vel - neigh.vel;
                let rpos = us.pos - neigh.pos;

                let adj_rpos = rpos.length() - cfg.particle_radius * 2.;
                let rvel = rvel.length();

                if us.vel.dot(neigh.vel) < 0.0 {
                    let min_intersect_time = adj_rpos / rvel;

                    if min_intersect_time > 0.0 {
                        dt = dt.min(min_intersect_time / 2.0);
                    }

                    /*
                    if min_intersect_time < cfg.dt / 100. {
                        /*
                        //if !self.handle_collision_particle(cfg, chem, i, neighbor) {
                            let m1 = chem.laws.compounds[self.particles[i].compound].mass;
                            let m2 = chem.laws.compounds[self.particles[neighbor].compound].mass;
                            let v1 = self.particles[i].vel;
                            let v2 = self.particles[neighbor].vel;

                            let (v1, v2) = elastic_collision_vect(m1, v1, m2, v2);
                            self.particles[i].vel = v1;
                            self.particles[neighbor].vel = v2;
                        //}
                        */
                    } else {
                    }
                    */
                }
            }
        }

        dt = dt.max(cfg.dt * 1e-3);

        integrate_velocity(&mut self.particles, cfg, chem, dt);

        // Gravity
        for part in self.particles.iter_mut() {
            part.vel += Vec2::Y * cfg.gravity * dt;
        }

        Some(dt)
    }

    fn enforce_speed_limit(&mut self, cfg: &SimConfig) {
        let speed_limit_sq = cfg.speed_limit.powi(2);
        for particle in &mut self.particles {
            if particle.vel.length_sq() > speed_limit_sq {
                //eprintln!("OVER SPEED LIMIT {:?}", particle.vel);
                particle.vel = particle.vel.normalized() * cfg.speed_limit * 0.9;
            }
        }
    }

    fn calculate_min_intersection(
        &mut self,
        cfg: &SimConfig,
        chem: &ChemicalWorld,
        accel: &QueryAccelerator,
    ) -> Option<Intersection> {
        let mut min_dt = cfg.dt;

        let mut min_particle_indices = None;
        let mut min_boundary_vel_idx = None;
        for i in 0..self.particles.len() {
            // Check time of intersection with neighbors
            for neighbor in accel.query_neighbors_fast(i, self.particles[i].pos) {
                //for neighbor in i + 1..self.particles.len() {
                let [p1, p2] = self.particles.get_disjoint_mut([i, neighbor]).unwrap();

                // TODO: Cache these intersections AND evict the cache ...
                if let Some(intersection_dt) = time_of_intersection_particles(
                    p2.pos - p1.pos,
                    p2.vel - p1.vel,
                    cfg.particle_radius * 2.0,
                ) {
                    assert!(intersection_dt >= 0.0);
                    if intersection_dt <= min_dt {
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

        // TODO: This is silly.
        let (index, data) = match (min_particle_indices, min_boundary_vel_idx) {
            (Some((index, neighbor)), None) => (index, IntersectionData::Particle { neighbor }),
            (None, Some((index, mirrored_velocity))) => {
                (index, IntersectionData::Wall { mirrored_velocity })
            }
            (None, None) => return None,
            _ => unreachable!(),
        };

        Some(Intersection {
            time: min_dt,
            data,
            index,
        })
    }

    fn handle_collision(
        &mut self,
        cfg: &SimConfig,
        chem: &ChemicalWorld,
        intersection: Intersection,
    ) {
        match intersection.data {
            IntersectionData::Wall { mirrored_velocity } => {
                self.particles[intersection.index].vel = mirrored_velocity;
            }
            IntersectionData::Particle { neighbor } => {
                self.handle_collision_particle(cfg, chem, intersection.index, neighbor);
            }
        }
    }

    fn handle_collision_particle(
        &mut self,
        cfg: &SimConfig,
        chem: &ChemicalWorld,
        i: usize,
        neighbor: usize,
    ) -> bool {
        // Synthesis
        // TODO: Make the sorted keys a type...
        let mut keys = [
            self.particles[i].compound,
            self.particles[neighbor].compound,
        ];
        keys.sort_by_key(|CompoundId(i)| *i);
        let [a, b] = keys;

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

        let kinetic_energy_component = vel_component.powi(2) * total_mass / 2.0;

        if kinetic_energy_component * cfg.kjmol_per_sim_energy > 500.0 {
            p1.to_decompose = Some(kinetic_energy_component);
            p2.to_decompose = Some(kinetic_energy_component);
        }

        //let (v1, v2) = elastic_collision(m1, , m2, 0.0);

        // Do the synthesizing
        if let Some(product_id) = chem.deriv.synthesis.get(&(a, b)) {
            let new_vel = (p1.vel * m1 + p2.vel * m2) / total_mass;

            let res = &chem.laws.compounds[*product_id];
            let delta_g = c1.std_free_energy + c2.std_free_energy - res.std_free_energy;
            let ke = new_vel.length_sq() * total_mass / 2.0;
            let ke = ke * cfg.kjmol_per_sim_energy;

            /*
            let scale_factor = if ke > 0.0 {
                ((ke + delta_g) / ke).sqrt()
            } else {
                1.0
            };
            */

            let p = (ke - delta_g).min(0.0).exp() as f64;

            if rand::thread_rng().gen_bool(p) {
                self.particles[i].compound = *product_id;
                //self.particles[i].vel = new_vel * scale_factor;

                self.particles.remove(neighbor);

                return true;
            }
        } 

        false
    }

    fn try_collide(
        &mut self,
        cfg: &SimConfig,
        chem: &ChemicalWorld,
        accel: &QueryAccelerator,
    ) -> bool {
        for i in 0..self.particles.len() {
            for neighbor in accel.query_neighbors_fast(i, self.particles[i].pos) {
                if self.particles[i].pos.distance(self.particles[neighbor].pos)
                    < cfg.particle_radius * 2.0
                {
                    if self.handle_collision_particle(cfg, chem, i, neighbor) {
                        return true;
                    }
                }
            }
        }

        false
    }

    fn try_decompose(
        &mut self,
        cfg: &SimConfig,
        chem: &ChemicalWorld,
        accel: &QueryAccelerator,
    ) -> bool {
        // Decompose one particle if possible
        let margin = 1e-2;

        'particles: for i in 0..self.particles.len() {
            let Some(particle_energy) = self.particles[i].to_decompose else {
                continue 'particles;
            };

            //let compound_id = self.particles[i].compound;
            //let compound = &chem.laws.compounds[compound_id];

            let all_products = &chem.deriv.decompositions[&self.particles[i].compound];

            let particle_energy_kjmol = particle_energy * cfg.kjmol_per_sim_energy;
            let Some(last_product_idx) = all_products.nearest_energy(particle_energy_kjmol) else {
                continue 'particles;
            };

            let product_idx = rand::thread_rng().gen_range(0..=last_product_idx);

            let particle = self.particles[i];
            let products = &all_products.products[product_idx];

            let delta_g = products.total_std_free_energy
                - chem.laws.compounds[particle.compound].std_free_energy;

            /*
            let velocity_scaling = if particle_energy_kjmol > 0.0 {
                ((particle_energy_kjmol - delta_g).max(0.0) / particle_energy_kjmol).sqrt()
            } else {
                1.0
            };
            */

            let ke = particle_energy;
            let p = (ke - delta_g).min(0.0).exp() as f64;

            if !rand::thread_rng().gen_bool(p) {
                continue 'particles;
            }


            //products.total_std_free_energy

            let mut children: Vec<Particle> = products
                .compounds
                .iter()
                .map(|(&compound, &n)| {
                    (0..n).map(move |_| Particle {
                        compound,
                        vel: particle.vel,
                        //vel: particle.vel * velocity_scaling,
                        ..particle
                    })
                })
                .flatten()
                .collect();

            let spacing = (cfg.particle_radius + margin) * 2.0;
            let our_radius = spacing * children.len() as f32;
            let safe_distance = our_radius + cfg.particle_radius;

            // Is anything nearby? Then we can't split.
            for neighbor in accel.query_neighbors_fast(i, self.particles[i].pos) {
                //for neighbor in 0..self.particles.len() {
                if neighbor == i {
                    continue;
                }

                let distance = self.particles[neighbor].pos.distance(self.particles[i].pos);
                if distance < safe_distance
                    || particle.pos.x < safe_distance
                    || cfg.dimensions.x - particle.pos.x < safe_distance
                    || particle.pos.y < safe_distance
                    || cfg.dimensions.y - particle.pos.y < safe_distance
                {
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
                particle.pos += direction * idx as f32 * spacing;
            }

            for (child_idx, child) in children.iter().enumerate() {
                for neighbor in accel.query_neighbors_fast(i, self.particles[i].pos) {
                    //for neighbor in 0..self.particles.len() {
                    if neighbor == i {
                        continue;
                    }

                    let distance = child.pos.distance(self.particles[neighbor].pos);

                    if distance < cfg.particle_radius * 2.0 {
                        println!("Child {child_idx} of {i} intersected {neighbor}");
                    }
                }
            }

            if let Some((first, xs)) = children.split_first() {
                self.particles[i] = *first;
                for particle in xs {
                    self.particles.push(*particle);
                }
            }

            return true;
        }

        false
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
            kjmol_per_sim_energy: 1e-2, // Arbitrary
            coulomb_k: 1e5,
            morse_mag: 1e5,
            morse_alpha: 1.0,
            morse_radius: 10.0,
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

fn morse_potential_deriv(cfg: &SimConfig, dist: f32) -> f32 {
    // https://en.wikipedia.org/wiki/Morse_potential
    let re = cfg.particle_radius; //cfg.morse_radius;
    let a = 1.0 / re; //cfg.morse_alpha;
    let d = cfg.morse_mag;
    let exp = (-a * (dist - re)).exp();
    //let morse = morse_mag * ((1. - exp).powi(2) - 1.0);
    2.0 * d * a * (1.0 - exp) * exp
}

fn acceleration(particles: &[Particle], cfg: &SimConfig, chem: &ChemicalWorld) -> Vec<Vec2> {
    // Particle velocity integration
    let mut accel = vec![Vec2::ZERO; particles.len()];
    for (i, acc) in accel.iter_mut().enumerate() {
        for j in (0..particles.len()).filter(|&j| j != i) {
            let [pi, pj] = [particles[i], particles[j]];
            let diff = pj.pos - pi.pos; // i -> j
            let n = diff.normalized();
            let dist = diff.length();

            let ci = &chem.laws.compounds[pi.compound];
            let cj = &chem.laws.compounds[pj.compound];
            let coulomb = ((ci.charge * cj.charge) as f32 * cfg.coulomb_k)
                / (diff.length_sq() + cfg.coulomb_softening);

            let morse = morse_potential_deriv(cfg, dist);

            let force = morse + coulomb;

            let dp = force * n;
            *acc -= dp / ci.mass;
        }
    }
    accel
}

fn integrate_velocity(particles: &mut [Particle], cfg: &SimConfig, chem: &ChemicalWorld, dt: f32) {
    let k1 = acceleration(particles, cfg, chem);

    let mut y1 = particles.to_vec();
    y1.iter_mut()
        .zip(&k1)
        .for_each(|(part, acc)| part.vel += *acc * dt / 2.0);
    y1.iter_mut()
        .for_each(|part| part.pos += part.vel * dt / 2.0);
    let k2 = acceleration(&y1, cfg, chem);

    let mut y2 = particles.to_vec();
    y2.iter_mut()
        .zip(&k2)
        .for_each(|(part, acc)| part.vel += *acc * dt / 2.0);
    y2.iter_mut()
        .for_each(|part| part.pos += part.vel * dt / 2.0);
    let k3 = acceleration(&y2, cfg, chem);

    let mut y3 = particles.to_vec();
    y3.iter_mut()
        .zip(&k2)
        .for_each(|(part, acc)| part.vel += *acc * dt);
    y3.iter_mut().for_each(|part| part.pos += part.vel * dt);
    let k4 = acceleration(&y3, cfg, chem);

    for (((part, y1), y2), y3) in particles.iter_mut().zip(&y1).zip(&y2).zip(&y3) {
        part.pos += (dt / 6.0) * (part.vel + 2.0 * y1.vel + 2.0 * y2.vel + y3.vel);
    }

    for ((((part, k1), k2), k3), k4) in particles.iter_mut().zip(&k1).zip(&k2).zip(&k3).zip(&k4) {
        part.vel += (dt / 6.0) * (*k1 + *k2 * 2.0 + *k3 * 2.0 + *k4);
    }

    //let accel = acceleration(particles, cfg, chem);
    //for (part, acc) in particles.iter_mut().zip(&accel) {
        //part.vel += *acc * dt;
    for part in particles.iter_mut() {
        part.pos += part.vel * dt;
    }

    boundaries(particles, cfg, chem, dt);
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
        }
    }
}
