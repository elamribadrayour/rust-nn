use crate::initialization::Initialization;

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Neuron {
    pub fn new(size: usize, initialization: &mut Box<dyn Initialization>) -> Self {
        Self {
            bias: initialization.init(),
            weights: (0..size).map(|_| initialization.init()).collect(),
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

    pub fn backward(&mut self, inputs: &[f64], output_grad: f64, lr: f64) -> Vec<f64> {
        let mut input_grad = vec![0.0; inputs.len()];

        let len = inputs.len();
        for i in 0..len {
            input_grad[i] = self.weights[i] * output_grad;
            self.weights[i] -= lr * inputs[i] * output_grad;
        }

        self.bias -= lr * output_grad;

        input_grad
    }
}
