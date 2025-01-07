use rust_nn::config::{Config, ConfigInitialization, ConfigLayer};
use rust_nn::network::Network;

#[test]
fn test_network_new() {
    let config = Config {
        lr: 0.1,
        epochs: 10,
        loss: "mse".to_string(),
        initialization: ConfigInitialization {
            method: "zero-centered".to_string(),
            seed: Some(42),
        },
        layers: vec![
            ConfigLayer {
                input_size: 3,
                output_size: 2,
                name: "layer-1".to_string(),
                activation: "sigmoid".to_string(),
            },
            ConfigLayer {
                input_size: 2,
                output_size: 1,
                name: "layer-2".to_string(),
                activation: "sigmoid".to_string(),
            },
        ],
    };
    let network = Network::new(config);

    assert_eq!(network.layers.len(), 2);
    assert_eq!(network.layers[0].neurons.len(), 2);
    assert_eq!(network.layers[1].neurons.len(), 1);
}

#[test]
fn test_network_forward() {
    let config = Config {
        lr: 0.01,
        epochs: 10,
        loss: "mse".to_string(),
        initialization: ConfigInitialization {
            method: "zero-centered".to_string(),
            seed: Some(42),
        },
        layers: vec![
            ConfigLayer {
                input_size: 3,
                output_size: 2,
                name: "layer-1".to_string(),
                activation: "sigmoid".to_string(),
            },
            ConfigLayer {
                input_size: 2,
                output_size: 2,
                name: "layer-2".to_string(),
                activation: "sigmoid".to_string(),
            },
        ],
    };
    let inputs = vec![0.5, 0.5, 0.5];
    let network = Network::new(config);
    let outputs = network.forward(&inputs);

    assert_eq!(outputs.len(), 2);
}

#[test]
fn test_network_activations() {
    let config = Config {
        epochs: 10,
        lr: 0.1,
        loss: "mse".to_string(),
        initialization: ConfigInitialization {
            method: "zero-centered".to_string(),
            seed: Some(42),
        },
        layers: vec![
            ConfigLayer {
                input_size: 3,
                output_size: 5,
                name: "layer-1".to_string(),
                activation: "sigmoid".to_string(),
            },
            ConfigLayer {
                input_size: 5,
                output_size: 4,
                name: "layer-2".to_string(),
                activation: "sigmoid".to_string(),
            },
            ConfigLayer {
                input_size: 4,
                output_size: 2,
                name: "layer-3".to_string(),
                activation: "sigmoid".to_string(),
            },
        ],
    };
    let inputs = vec![0.5, 0.5, 0.5];
    let network = Network::new(config.clone());
    let activations = network.activations(&inputs);

    assert_eq!(activations.len(), 3);
}

#[test]
fn test_network_backward() {
    let config = Config {
        lr: 0.1,
        epochs: 10,
        loss: "mse".to_string(),
        initialization: ConfigInitialization {
            method: "zero-centered".to_string(),
            seed: Some(42),
        },
        layers: vec![
            ConfigLayer {
                input_size: 3,
                output_size: 2,
                name: "layer-1".to_string(),
                activation: "sigmoid".to_string(),
            },
            ConfigLayer {
                input_size: 2,
                output_size: 1,
                name: "layer-2".to_string(),
                activation: "sigmoid".to_string(),
            },
        ],
    };

    let target = vec![0.0];
    let inputs = vec![0.5, 0.5, 0.5];
    let mut network = Network::new(config.clone());

    network.backward(config.lr, &inputs, &target);
}

#[test]
fn test_network_train_xor() {
    let dataset = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    let config = Config {
        lr: 0.5,
        epochs: 20000,
        loss: "mse".to_string(),
        initialization: ConfigInitialization {
            method: "zero-centered".to_string(),
            seed: Some(42),
        },
        layers: vec![
            ConfigLayer {
                name: "hidden".to_string(),
                input_size: 2,
                output_size: 8,
                activation: "relu".to_string(),
            },
            ConfigLayer {
                name: "hidden2".to_string(),
                input_size: 8,
                output_size: 4,
                activation: "relu".to_string(),
            },
            ConfigLayer {
                name: "output".to_string(),
                input_size: 4,
                output_size: 1,
                activation: "sigmoid".to_string(),
            },
        ],
    };

    let mut network = Network::new(config);
    network.train(&dataset);

    // Test predictions
    for (inputs, expected) in dataset {
        let output = network.forward(&inputs);
        assert!(
            (output[0] - expected[0]).abs() < 1e-3,
            "Failed XOR test: input {:?}, expected {}, got {}",
            inputs,
            expected[0],
            output[0]
        );
    }
}

#[test]
fn test_network_and() {
    let dataset = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![0.0]),
        (vec![1.0, 0.0], vec![0.0]),
        (vec![1.0, 1.0], vec![1.0]),
    ];

    let config = Config {
        lr: 0.3,
        epochs: 20000,
        loss: "mse".to_string(),
        initialization: ConfigInitialization {
            method: "zero-centered".to_string(),
            seed: Some(42),
        },
        layers: vec![
            ConfigLayer {
                name: "hidden1".to_string(),
                input_size: 2,
                output_size: 4,
                activation: "relu".to_string(),
            },
            ConfigLayer {
                name: "hidden2".to_string(),
                input_size: 4,
                output_size: 4,
                activation: "relu".to_string(),
            },
            ConfigLayer {
                name: "output".to_string(),
                input_size: 4,
                output_size: 1,
                activation: "sigmoid".to_string(),
            },
        ],
    };

    let mut network = Network::new(config);
    network.train(&dataset);

    for (inputs, expected) in dataset {
        let output = network.forward(&inputs);
        assert!(
            (output[0] - expected[0]).abs() < 1e-3,
            "Failed AND test: input {:?}, expected {}, got {}",
            inputs,
            expected[0],
            output[0]
        );
    }
}

#[test]
fn test_network_binary_addition() {
    let dataset = vec![
        (vec![0.0, 0.0], vec![0.0, 0.0]), // 0+0 = 00
        (vec![0.0, 1.0], vec![1.0, 0.0]), // 0+1 = 01
        (vec![1.0, 0.0], vec![1.0, 0.0]), // 1+0 = 01
        (vec![1.0, 1.0], vec![0.0, 1.0]), // 1+1 = 10
    ];

    let config = Config {
        lr: 0.5,
        epochs: 10000,
        loss: "mse".to_string(),
        initialization: ConfigInitialization {
            method: "zero-centered".to_string(),
            seed: Some(42),
        },
        layers: vec![
            ConfigLayer {
                name: "hidden1".to_string(),
                input_size: 2,
                output_size: 12,
                activation: "tanh".to_string(),
            },
            ConfigLayer {
                name: "hidden2".to_string(),
                input_size: 12,
                output_size: 8,
                activation: "tanh".to_string(),
            },
            ConfigLayer {
                name: "hidden3".to_string(),
                input_size: 8,
                output_size: 6,
                activation: "tanh".to_string(),
            },
            ConfigLayer {
                name: "output".to_string(),
                input_size: 6,
                output_size: 2,
                activation: "sigmoid".to_string(),
            },
        ],
    };

    let mut network = Network::new(config);
    network.train(&dataset);

    for (inputs, expected) in dataset {
        let output = network.forward(&inputs);
        assert!(
            (output[0] - expected[0]).abs() < 1e-3 && (output[1] - expected[1]).abs() < 1e-3,
            "Failed binary addition test: input {:?}, expected {:?}, got {:?}",
            inputs,
            expected,
            output
        );
    }
}

#[test]
fn test_network_pattern_recognition() {
    // Input: 4 binary digits representing a pattern
    // Output: 1 if pattern has more 1s than 0s, 0 otherwise
    let dataset = vec![
        (vec![0.0, 0.0, 0.0, 0.0], vec![0.0]), // 0 ones
        (vec![1.0, 0.0, 0.0, 0.0], vec![0.0]), // 1 one
        (vec![1.0, 1.0, 0.0, 0.0], vec![0.0]), // 2 ones
        (vec![1.0, 1.0, 1.0, 0.0], vec![1.0]), // 3 ones
        (vec![1.0, 1.0, 1.0, 1.0], vec![1.0]), // 4 ones
    ];

    let config = Config {
        lr: 0.3,
        epochs: 15000,
        loss: "mse".to_string(),
        initialization: ConfigInitialization {
            method: "zero-centered".to_string(),
            seed: Some(42),
        },
        layers: vec![
            ConfigLayer {
                name: "hidden".to_string(),
                input_size: 4,
                output_size: 6,
                activation: "relu".to_string(),
            },
            ConfigLayer {
                name: "output".to_string(),
                input_size: 6,
                output_size: 1,
                activation: "sigmoid".to_string(),
            },
        ],
    };

    let mut network = Network::new(config);
    network.train(&dataset);

    for (inputs, expected) in dataset {
        let output = network.forward(&inputs);
        assert!(
            (output[0] - expected[0]).abs() < 1e-3,
            "Failed pattern recognition test: input {:?}, expected {}, got {}",
            inputs,
            expected[0],
            output[0]
        );
    }
}
