use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::initialization::Initialization;

pub struct ZeroCentered {
    rng: Box<dyn RngCore>,
}

impl ZeroCentered {
    pub fn new(seed: Option<u64>) -> Self {
        match seed {
            Some(seed) => Self {
                rng: Box::new(ChaCha8Rng::seed_from_u64(seed)),
            },
            None => Self {
                rng: Box::new(rand::thread_rng()),
            },
        }
    }
}

impl Initialization for ZeroCentered {
    fn init(&mut self) -> f64 {
        self.rng.gen_range(-1.0..1.0)
    }
}
