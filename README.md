# rust-nn

`rust-nn` is a neural network library written in Rust. It provides a simple and efficient way to build and train neural networks.

## Features

- Define custom neural network architectures
- Support for different activation functions (e.g., ReLU, Sigmoid, Tanh)
- Train networks using backpropagation
- Serialize and deserialize network configurations
- Log training metrics such as loss and accuracy

## Installation

To use `rust-nn`, you need to clone the repository and build the project:

### Clone the Repository

```
git clone https://github.com/elamribadrayour/rust-nn.git
cd rust-nn
```

### Build the Project

Build the project using Cargo:

```bash
cargo build --release
```

### Run Tests

To ensure everything is working correctly, run the tests:

```bash
cargo test
```

## Usage

Here's a basic example of how to create and train a neural network using `rust-nn`:

```rust
use rust_nn::network::Network;
use rust_nn::config::{Config, ConfigLayer};

fn main() {
    let config = Config {
        epochs: 1000,
        lr: 0.1,
        loss: "mse".to_string(),
        initialization: ConfigInitialization {
            method: "zero-centered".to_string(),
            seed: Some(42),
        },
        layers: vec![
            ConfigLayer {
                input_size: 2,
                output_size: 2,
                activation: "relu".to_string(),
            },
            ConfigLayer {
                input_size: 2,
                output_size: 1,
                activation: "sigmoid".to_string(),
            },
        ],
    };

    let mut network = Network::new(config);

    let dataset = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];

    network.train(&dataset);

    for (inputs, target) in dataset.iter() {
        let target = target[0];
        let output = network.forward(inputs);
        let prediction = if output[0] > 0.5 { 1.0 } else { 0.0 };
        assert_eq!(prediction, target);
    }
}
```

## License

This project is licensed under the WTFPL license. See the [LICENSE](LICENSE) file for details.

## Author

- **Email:** [badrayour.elamri@protonmail.com](mailto:badrayour.elamri@protonmail.com)
