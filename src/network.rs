use rand::RngCore;

use crate::{config::Config, layer::Layer};

pub struct Network {
    pub config: Config,
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(rng: &mut dyn RngCore, config: Config) -> Self {
        Self {
            config: config.clone(),
            layers: config.layers.iter().map(|c| Layer::new(rng, c)).collect(),
        }
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.layers
            .iter()
            .fold(inputs.to_owned(), |outputs, l| l.forward(&outputs))
    }

    pub fn activations(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut output = inputs.to_vec();
        let mut outputs: Vec<Vec<f64>> = Vec::new();

        self.layers.iter().for_each(|l| {
            output = l.forward(&output);
            outputs.push(output.clone());
        });

        outputs
    }

    pub fn backward(&mut self, inputs: &[f64], target: &[f64], learning_rate: f64) {
        let activations = self.activations(inputs);

        // Calculate the gradient of the loss with respect to the output of the network
        // This is the initial gradient that will be backpropagated through the network
        let mut output_grad = activations
            .last()
            .unwrap()
            .iter()
            .zip(target.iter())
            .map(|(output, target)| output - target)
            .collect::<Vec<f64>>();

        // Backward pass through each layer
        self.layers
            .iter_mut()
            .enumerate()
            .rev()
            .for_each(|(i, layer)| {
                let input = if i == 0 { inputs } else { &activations[i - 1] };
                output_grad = layer.backward(input, &output_grad, learning_rate);
            });
    }

    pub fn train(&mut self, dataset: &[(Vec<f64>, Vec<f64>)]) {
        for _ in 0..self.config.epochs {
            for (inputs, target) in dataset.iter() {
                self.forward(inputs);
                self.backward(inputs, target, self.config.learning_rate);
            }
        }
    }

}
