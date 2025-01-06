# rust-nn

`rust-nn` is a neural network library written in Rust. It provides a simple and efficient way to build and train neural networks.

## Features

- Define custom neural network architectures
- Support for different activation functions (e.g., ReLU, Sigmoid)
- Train networks using backpropagation
- Serialize and deserialize network configurations

## Installation

To use `rust-nn` in your project, add the following to your `Cargo.toml`:

```bash
[dependencies]
rust-nn = "0.1.0"
```

## Usage

Here's a basic example of how to create and train a neural network using `rust-nn`:

```rust

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use rust_nn::network::Network;
use rust_nn::config::{Config, ConfigLayer};

fn main() {
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
```

## License

This project is licensed under the WTFPL license [LICENSE](LICENSE).

## Author

- **Name:** El Amri Badr Ayour
- **Email:** [badrayour.elamri@protonmail.com](mailto:badrayour.elamri@protonmail.com)
- **GitHub:** [elamribadrayour](https://github.com/elamribadrayour)
