use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct Config {
    pub epochs: usize,
    pub learning_rate: f64,
    pub layers: Vec<ConfigLayer>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ConfigLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub activation: String,
}

impl Config {
    pub fn new(path: &str) -> Self {
        let file = std::fs::File::open(path).expect("Unable to open file");
        serde_json::from_reader(file).expect("Unable to parse JSON")
    }
}
