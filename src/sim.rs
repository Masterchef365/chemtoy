use egui::{Color32, DragValue, Pos2, Rect, Stroke, Vec2};
use crate::laws::{ChemicalWorld, Compound, CompoundId, Compounds, Element, Elements, Laws};
use crate::query_accel::QueryAccelerator;
use rand::prelude::Distribution;


pub struct Sim {
    pub particles: Vec<Particle>,
}

pub struct Particle {
    pub compound: CompoundId,
    pub pos: Pos2,
    pub vel: Vec2,
}

pub struct SimConfig {
    pub dimensions: Vec2,
    pub dt: f32,
    pub particle_radius: f32,
    pub max_collision_time: f32,
    pub fill_timestep: bool,
    pub gravity: f32,
    pub speed_limit: f32,
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

        let speed_limit_sq = cfg.speed_limit.powi(2);
        for particle in &mut self.particles {
            if particle.vel.length_sq() > speed_limit_sq {
                eprintln!("OVER SPEED LIMIT {:?}", particle.vel);
                particle.vel = particle.vel.normalized() * cfg.speed_limit;
            }
        }

        let accel = QueryAccelerator::new(&points, cfg.speed_limit * 2.0);

        let mut elapsed = 0.0;
        let mut remaining_loops = 1000;
        while elapsed < cfg.dt {
            if remaining_loops == 0 {
                break;
            }
            remaining_loops -= 1;

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
                    let m1 = chem.laws.compounds[p1.compound].mass;
                    let m2 = chem.laws.compounds[p2.compound].mass;

                    let rel_pos = p2.pos - p1.pos;
                    let rel_dir = rel_pos.normalized();
                    let rel_vel = p2.vel - p1.vel;

                    let vel_component = rel_vel.dot(rel_dir).abs();
                    //let (v1, v2) = elastic_collision(m1, , m2, 0.0);
                    p2.vel += rel_dir * (vel_component * 2.0 * m1 / (m2 + m1));
                    p1.vel += -rel_dir * (vel_component * 2.0 * m2 / (m2 + m1));
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

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            dimensions: Vec2::new(100., 100.),
            dt: 1. / 60.,
            particle_radius: 10.0,
            max_collision_time: 1e-2,
            fill_timestep: true,
            gravity: 9.8,
            speed_limit: 500.0,
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


