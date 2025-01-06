use std::collections::HashMap;

use crate::activation::*;

pub fn get_activation(name: &str, params: HashMap<String, f64>) -> Box<dyn Activation> {
    match name {
        "relu" => Box::new(ReLU {}) as Box<dyn Activation>,
        "linear" => Box::new(Linear {}) as Box<dyn Activation>,
        "sigmoid" => Box::new(Sigmoid {}) as Box<dyn Activation>,
        "heaviside" => Box::new(Heaviside {}) as Box<dyn Activation>,
        "gaussian" => Box::new(Gaussian::new(params)) as Box<dyn Activation>,
        "multiquadratics" => Box::new(Multiquadratics::new(params)) as Box<dyn Activation>,
        _ => panic!("activation function should be sigmoid or relu"),
    }
}
