use rust_nn::config::{ConfigInitialization, ConfigLayer};
use rust_nn::initialization::get_initialization;
use rust_nn::layer::Layer;

#[test]
fn test_layer_new() {
    let config = ConfigLayer {
        input_size: 3,
        output_size: 2,
        name: "test".to_string(),
        activation: "sigmoid".to_string(),
    };
    let initialization = ConfigInitialization {
        method: "uniform-distribution".to_string(),
        seed: Some(42),
    };
    let mut initialization = get_initialization(&initialization);
    let layer = Layer::new(&config, &mut initialization);
    let weights = [
        0.900550815344968,
        -0.1449671942869606,
        0.25472104239468063,
        -0.7000822594193501,
        -0.3839188808041807,
        0.6077455343512534,
    ];

    let mut i = 0;
    layer.neurons.iter().for_each(|neuron| {
        assert_eq!(neuron.weights.len(), 3);
        // assert!(neuron.bias >= 0.0 && neuron.bias < 1.0);
        neuron.weights.iter().for_each(|&w| {
            assert!((w - weights[i]).abs() < 1e-7);
            i += 1;
        });
    });
}

#[test]
fn test_layer_forward() {
    let config = ConfigLayer {
        input_size: 3,
        output_size: 2,
        name: "test".to_string(),
        activation: "sigmoid".to_string(),
    };
    let initialization = ConfigInitialization {
        method: "uniform-distribution".to_string(),
        seed: Some(42),
    };
    let mut initialization = get_initialization(&initialization);
    let layer = Layer::new(&config, &mut initialization);
    let inputs = vec![0.5, 0.5, 0.5];
    let outputs = layer.forward(&inputs);
    assert_eq!(outputs.len(), 2);
    outputs.iter().for_each(|&output| {
        assert!(output >= 0.0 && output < 1.0); // Since Sigmoid activation function output is in range [0.0, 1.0)
    });
}

#[test]
fn test_layer_backward() {
    let initialization = ConfigInitialization {
        method: "uniform-distribution".to_string(),
        seed: Some(42),
    };
    let config = ConfigLayer {
        input_size: 3,
        output_size: 2,
        name: "test".to_string(),
        activation: "sigmoid".to_string(),
    };
    let mut initialization = get_initialization(&initialization);
    let mut layer = Layer::new(&config, &mut initialization);
    let inputs = vec![0.5, 0.5, 0.5];
    let output_grads = vec![0.1, 0.1];
    let lr = 0.01;

    let input_grads = layer.backward(lr, &inputs, &output_grads);

    // Check that the input gradients are calculated correctly
    assert_eq!(input_grads.len(), inputs.len());

    let weights = [
        0.9004333134888671,
        -0.14508469614306138,
        0.2546035405385798,
        -0.7001997612754509,
        -0.3840363826602815,
        0.6076280324951526,
    ];
    let mut i = 0;
    layer.neurons.iter().for_each(|neuron| {
        neuron.weights.iter().for_each(|&w| {
            assert!((w - weights[i]).abs() < 1e-7);
            i += 1;
        });
    });
}
