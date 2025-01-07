use std::collections::HashMap;

use crate::config::Config;
use crate::initialization::{get_initialization, Initialization};
use crate::layer::Layer;
use crate::logger::Logger;
use crate::loss::{get_loss, Loss};

pub struct Network {
    pub config: Config,
    pub logger: Logger,
    pub layers: Vec<Layer>,
    pub loss: Box<dyn Loss>,
    pub initialization: Box<dyn Initialization>,
}

impl Network {
    pub fn new(config: Config) -> Self {
        let mut initialization = get_initialization(&config.initialization);
        let layers = config
            .layers
            .iter()
            .map(|c| Layer::new(c, &mut initialization))
            .collect();
        Self {
            initialization,
            logger: Logger::new(),
            config: config.clone(),
            loss: get_loss(config.loss.as_str(), HashMap::new()),
            layers,
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

    pub fn backward(&mut self, lr: f64, inputs: &[f64], targets: &[f64]) {
        // https://en.wikipedia.org/wiki/Backpropagation

        let activations = self.activations(inputs);

        let output = activations.last().unwrap();
        let mut output_grad = self.loss.gradient(output, targets);

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let input = if i == 0 { inputs } else { &activations[i - 1] };
            output_grad = layer.backward(lr, input, &output_grad);
        }
    }

    pub fn train(&mut self, dataset: &[(Vec<f64>, Vec<f64>)]) {
        for epoch in 0..self.config.epochs {
            for (id, (inputs, targets)) in dataset.iter().enumerate() {
                let outputs = self.forward(inputs);
                let loss = self.loss.function(&outputs, targets);
                self.backward(self.config.lr, inputs, targets);
                self.logger.loss(epoch, id, loss);
            }
        }
    }
}
