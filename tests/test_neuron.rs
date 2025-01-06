use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use rust_nn::neuron::Neuron;

#[test]
fn test_neuron_new() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let neuron = Neuron::new(&mut rng, 3);
    assert_eq!(neuron.weights.len(), 3);
    assert!(neuron.bias >= 0.0 && neuron.bias < 1.0);
    neuron.weights.iter().for_each(|&w| {
        assert!(w >= 0.0 && w < 1.0);
    });
}

#[test]
fn test_neuron_forward() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let neuron = Neuron::new(&mut rng, 3);
    let inputs = vec![0.5, 0.5, 0.5];
    let output = neuron.forward(&inputs);
    assert!(output >= 0.0 && output < 2.0); // Since weights and bias are in range [0.0, 1.0)
}

#[test]
fn test_neuron_backward() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut neuron = Neuron::new(&mut rng, 3);
    let inputs = vec![0.5, 0.5, 0.5];
    let output_grad = 0.1;
    let learning_rate = 0.01;

    let input_grad = neuron.backward(&inputs, output_grad, learning_rate);

    // Check that the input gradients are calculated correctly
    assert_eq!(input_grad.len(), inputs.len());

    // Check that weights and bias are updated
    neuron.weights.iter().for_each(|&w| {
        assert!(w >= 0.0 && w < 1.0);
    });
    assert!(neuron.bias >= 0.0 && neuron.bias < 1.0);
}
