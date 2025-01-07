use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct ConfigInitialization {
    pub method: String,
    pub seed: Option<u64>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Config {
    pub lr: f64,
    pub epochs: usize,
    pub loss: String,
    pub layers: Vec<ConfigLayer>,
    pub initialization: ConfigInitialization,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ConfigLayer {
    pub name: String,
    pub input_size: usize,
    pub output_size: usize,
    pub activation: String,
}

impl Config {
    pub fn new(path: &str) -> Self {
        let file = std::fs::File::open(path).expect("Unable to open file");
        serde_json::from_reader(file).expect("Unable to parse JSON")
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.lr <= 0.0 {
            return Err("learning rate must be positive".to_string());
        }

        if self.loss.is_empty() {
            return Err("loss function must be specified".to_string());
        }

        if self
            .layers
            .iter()
            .zip(self.layers.iter().skip(1))
            .any(|(l, n)| l.output_size != n.input_size)
        {
            return Err("input and output size must be the same".to_string());
        }

        if self.layers.iter().any(|l| l.validate().is_err()) {
            return Err("layer validation failed".to_string());
        }

        Ok(())
    }
}

impl ConfigLayer {
    pub fn validate(&self) -> Result<(), String> {
        if self.input_size == 0 || self.output_size == 0 {
            return Err("input and output size must be positive".to_string());
        }

        if self.activation.is_empty() {
            return Err("activation function must be specified".to_string());
        }

        Ok(())
    }
}

impl ConfigInitialization {
    pub fn validate(&self) -> Result<(), String> {
        if self.method.is_empty() {
            return Err("initialization method must be specified".to_string());
        }

        Ok(())
    }
}
