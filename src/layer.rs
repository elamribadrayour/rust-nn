use std::collections::HashMap;

use itertools::izip;

use crate::activation::{get_activation, Activation};
use crate::config::ConfigLayer;
use crate::initialization::Initialization;
use crate::neuron::Neuron;

pub struct Layer {
    pub name: String,
    pub neurons: Vec<Neuron>,
    pub activation: Box<dyn Activation>,
}

impl Layer {
    pub fn new(config: &ConfigLayer, initialization: &mut Box<dyn Initialization>) -> Self {
        Self {
            name: config.name.clone(),
            activation: get_activation(&config.activation, HashMap::new()),
            neurons: (0..config.output_size)
                .map(|_| Neuron::new(config.input_size, initialization))
                .collect(),
        }
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|w| {
                *self
                    .activation
                    .function(&[w.forward(inputs)])
                    .first()
                    .unwrap()
            })
            .collect()
    }

    pub fn backward(&mut self, lr: f64, inputs: &[f64], output_grads: &[f64]) -> Vec<f64> {
        let mut input_grad = vec![0.0; inputs.len()];

        let errors = self.activation.derivative(inputs);
        for (neuron, output_grad, error) in
            izip!(self.neurons.iter_mut(), output_grads.iter(), errors.iter())
        {
            let error = error * output_grad;
            let neuron_input_grad = neuron.backward(inputs, error, lr);
            for i in 0..neuron_input_grad.len() {
                input_grad[i] += neuron_input_grad[i];
            }
        }
        input_grad
    }
}
