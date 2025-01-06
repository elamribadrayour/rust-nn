use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use rust_nn::config::{Config, ConfigLayer};
use rust_nn::network::Network;

#[test]
fn test_network_new() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let config = Config {
        epochs: 10,
        learning_rate: 0.1,
        layers: vec![
            ConfigLayer {
                input_size: 3,
                output_size: 2,
                activation: "sigmoid".to_string(),
            },
            ConfigLayer {
                input_size: 2,
                output_size: 1,
                activation: "sigmoid".to_string(),
            },
        ],
    };
    let network = Network::new(&mut rng, config);

    assert_eq!(network.layers.len(), 2);
    assert_eq!(network.layers[0].neurons.len(), 2);
    assert_eq!(network.layers[1].neurons.len(), 1);
    network.layers.iter().for_each(|layer| {
        layer.neurons.iter().for_each(|neuron| {
            assert!(neuron.bias >= 0.0 && neuron.bias < 1.0);
            neuron.weights.iter().for_each(|&w| {
                assert!(w >= 0.0 && w < 1.0);
            });
        });
    });
}

#[test]
fn test_network_forward() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let config = Config {
        epochs: 10,
        learning_rate: 0.1,
        layers: vec![
            ConfigLayer {
                input_size: 3,
                output_size: 2,
                activation: "sigmoid".to_string(),
            },
            ConfigLayer {
                input_size: 2,
                output_size: 2,
                activation: "sigmoid".to_string(),
            },
        ],
    };
    let inputs = vec![0.5, 0.5, 0.5];
    let network = Network::new(&mut rng, config);
    let outputs = network.forward(&inputs);

    assert_eq!(outputs.len(), 2);
    outputs.iter().for_each(|&output| {
        assert!(output >= 0.0 && output < 1.0); // Since Sigmoid activation function output is in range [0.0, 1.0)
    });
}

#[test]
fn test_network_activations() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let config = Config {
        epochs: 10,
        learning_rate: 0.1,
        layers: vec![
            ConfigLayer {
                input_size: 3,
                output_size: 5,
                activation: "sigmoid".to_string(),
            },
            ConfigLayer {
                input_size: 5,
                output_size: 4,
                activation: "sigmoid".to_string(),
            },
            ConfigLayer {
                input_size: 4,
                output_size: 2,
                activation: "sigmoid".to_string(),
            },
        ],
    };
    let inputs = vec![0.5, 0.5, 0.5];
    let network = Network::new(&mut rng, config.clone());
    let activations = network.activations(&inputs);

    assert_eq!(activations.len(), 3);
    activations
        .iter()
        .zip(config.layers.iter())
        .for_each(|(output, layer)| {
            assert_eq!(output.len(), layer.output_size);
            output.iter().for_each(|&activation| {
                assert!(activation >= 0.0 && activation <= 1.0); // Since Sigmoid activation function output is in range [0.0, 1.0]
            });
        });
}

#[test]
fn test_network_backward() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let config = Config {
        epochs: 10,
        learning_rate: 0.1,
        layers: vec![
            ConfigLayer {
                input_size: 3,
                output_size: 2,
                activation: "sigmoid".to_string(),
            },
            ConfigLayer {
                input_size: 2,
                output_size: 1,
                activation: "sigmoid".to_string(),
            },
        ],
    };
    let mut network = Network::new(&mut rng, config);
    let inputs = vec![0.5, 0.5, 0.5];
    let target_outputs = vec![0.0]; // Example target output for a single output neuron

    let learning_rate = 0.1; // Define the learning rate
    network.backward(&inputs, &target_outputs, learning_rate);

    network.layers.iter().for_each(|layer| {
        layer.neurons.iter().for_each(|neuron| {
            neuron.weights.iter().for_each(|&w| {
                assert!(w >= 0.0 && w < 1.0);
            });
            assert!(neuron.bias >= 0.0 && neuron.bias < 1.0);
        });
    });
}

#[test]
fn test_network_train_xor() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let config = Config {
        epochs: 1000,
        learning_rate: 0.1,
        layers: vec![
            ConfigLayer {
                input_size: 2,
                output_size: 2,
                activation: "relu".to_string(),
            },
            ConfigLayer {
                input_size: 2,
                output_size: 1,
                activation: "relu".to_string(),
            },
        ],
    };
    let mut network = Network::new(&mut rng, config);

    let xor_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let xor_outputs = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    let dataset: Vec<(Vec<f64>, Vec<f64>)> = xor_inputs.iter().cloned().zip(xor_outputs.iter().cloned()).collect();
    network.train(&dataset);

    for (inputs, target) in xor_inputs.iter().zip(xor_outputs.iter()) {
        let target = target[0];
        let output = network.forward(inputs);
        let prediction = if output[0] > 0.5 { 1.0 } else { 0.0 };
        assert_eq!(prediction, target);
    }
}
