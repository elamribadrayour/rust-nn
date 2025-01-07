use std::collections::HashMap;

use crate::activation::*;

pub fn get_activation(name: &str, params: HashMap<String, f64>) -> Box<dyn Activation> {
    match name {
        "relu" => Box::new(ReLU {}),
        "tanh" => Box::new(Tanh {}),
        "linear" => Box::new(Linear {}),
        "sigmoid" => Box::new(Sigmoid {}),
        "heaviside" => Box::new(Heaviside {}),
        "gaussian" => Box::new(Gaussian::new(params)),
        "multiquadratics" => Box::new(Multiquadratics::new(params)),
        _ => panic!("activation function should be sigmoid or relu"),
    }
}
