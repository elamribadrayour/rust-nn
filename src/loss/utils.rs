use std::collections::HashMap;

use crate::loss::*;

pub fn get_loss(name: &str, _: HashMap<String, f64>) -> Box<dyn Loss> {
    match name {
        "mse" => Box::new(Mse {}),
        "crossentropy" => Box::new(CrossEntropy {}),
        "binary-crossentropy" => Box::new(BinaryCrossEntropy {}),
        _ => panic!("loss function should be mse"),
    }
}
