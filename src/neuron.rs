use rand::{Rng, RngCore};

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
    pub fn new(rng: &mut dyn RngCore, size: usize) -> Self {
        Self {
            bias: rng.gen_range(0.0..1.0),
            weights: (0..size).map(|_| rng.gen_range(0.0..1.0)).collect(),
        }
    }

    pub fn forward(&self, inputs: &[f64]) -> f64 {
        assert_eq!(
            self.weights.len(),
            inputs.len(),
            "weights length != inputs length"
        );
        self.weights
            .iter()
            .zip(inputs.iter())
            .map(|(x, y)| x * y)
            .sum::<f64>()
            + self.bias
    }

    pub fn backward(&mut self, inputs: &[f64], output_grad: f64, learning_rate: f64) -> Vec<f64> {
        let mut input_grad = vec![0.0; inputs.len()];

        inputs.iter().enumerate().for_each(|(i, input)| {
            input_grad[i] = self.weights[i] * output_grad;
            self.weights[i] -= learning_rate * input * output_grad;
        });

        self.bias -= learning_rate * output_grad;

        input_grad
    }
}
