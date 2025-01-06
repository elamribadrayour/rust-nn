use super::{Activation, ReLU, Sigmoid};

pub fn get_activation(name: &str) -> Box<dyn Activation> {
    match name {
        "relu" => Box::new(ReLU {}) as Box<dyn Activation>,
        "sigmoid" => Box::new(Sigmoid {}) as Box<dyn Activation>,
        _ => panic!("activation function should be sigmoid or relu"),
    }
}
