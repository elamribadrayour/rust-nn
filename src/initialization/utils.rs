use crate::config::ConfigInitialization;
use crate::initialization::*;

pub fn get_initialization(config: &ConfigInitialization) -> Box<dyn Initialization> {
    match config.method.as_str() {
        "zero-centered" => Box::new(ZeroCentered::new(config.seed)),
        "uniform-distribution" => Box::new(UniformDistribution::new(config.seed)),
        _ => panic!("Invalid initialization method"),
    }
}
