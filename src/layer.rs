use rand::RngCore;

use crate::activation::{get_activation, Activation};
use crate::config::ConfigLayer;
use crate::neuron::Neuron;

pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub activation: Box<dyn Activation>,
}

impl Layer {
    pub fn new(rng: &mut dyn RngCore, config: &ConfigLayer) -> Self {
        Self {
            activation: get_activation(&config.activation),
            neurons: (0..config.output_size)
                .map(|_| Neuron::new(rng, config.input_size))
                .collect(),
        }
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|w| self.activation.function(w.forward(inputs)))
            .collect()
    }

    pub fn backward(
        &mut self,
        inputs: &[f64],
        output_grad: &[f64],
        learning_rate: f64,
    ) -> Vec<f64> {
        let mut input_grad = vec![0.0; inputs.len()];

        self.neurons
            .iter_mut()
            .zip(output_grad.iter())
            .for_each(|(neuron, grad)| {
                // Calculate the gradient of the loss with respect to the neuron's output
                let neuron_output = neuron.forward(inputs);
                let activation_grad = self.activation.derivative(neuron_output) * grad;

                // Backpropagate through the neuron
                let neuron_input_grad = neuron.backward(inputs, activation_grad, learning_rate);
                neuron_input_grad.iter().enumerate().for_each(|(i, &val)| {
                    input_grad[i] += val;
                });
            });

        input_grad
    }
}
