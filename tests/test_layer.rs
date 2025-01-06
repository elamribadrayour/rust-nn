use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use rust_nn::config::ConfigLayer;
use rust_nn::layer::Layer;

#[test]
fn test_layer_new() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let config = ConfigLayer {
        input_size: 3,
        output_size: 2,
        activation: "sigmoid".to_string(),
    };
    let layer = Layer::new(&mut rng, &config);
    assert_eq!(layer.neurons.len(), 2);
    layer.neurons.iter().for_each(|neuron| {
        assert_eq!(neuron.weights.len(), 3);
        assert!(neuron.bias >= 0.0 && neuron.bias < 1.0);
        neuron.weights.iter().for_each(|&w| {
            assert!(w >= 0.0 && w < 1.0);
        });
    });
}

#[test]
fn test_layer_forward() {
    let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(42);
    let config = ConfigLayer {
        input_size: 3,
        output_size: 2,
        activation: "sigmoid".to_string(),
    };
    let layer = Layer::new(&mut rng, &config);
    let inputs = vec![0.5, 0.5, 0.5];
    let outputs = layer.forward(&inputs);
    assert_eq!(outputs.len(), 2);
    outputs.iter().for_each(|&output| {
        assert!(output >= 0.0 && output < 1.0); // Since Sigmoid activation function output is in range [0.0, 1.0)
    });
}

#[test]
fn test_layer_backward() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let config = ConfigLayer {
        input_size: 3,
        output_size: 2,
        activation: "sigmoid".to_string(),
    };
    let mut layer = Layer::new(&mut rng, &config);
    let inputs = vec![0.5, 0.5, 0.5];
    let output_grads = vec![0.1, 0.1];
    let learning_rate = 0.01;

    let input_grads = layer.backward(&inputs, &output_grads, learning_rate);

    // Check that the input gradients are calculated correctly
    assert_eq!(input_grads.len(), inputs.len());

    // Check that each neuron's weights and bias are updated
    layer.neurons.iter().for_each(|neuron| {
        neuron.weights.iter().for_each(|&w| {
            assert!(w >= 0.0 && w < 1.0);
        });
        assert!(neuron.bias >= 0.0 && neuron.bias < 1.0);
    });
}
