use rust_nn::config::ConfigInitialization;
use rust_nn::initialization::get_initialization;
use rust_nn::neuron::Neuron;

#[test]
fn test_neuron_new() {
    let initialization = ConfigInitialization {
        method: "uniform-distribution".to_string(),
        seed: Some(42),
    };
    let mut initialization = get_initialization(&initialization);
    let neuron = Neuron::new(3, &mut initialization);
    assert_eq!(neuron.weights.len(), 3);
    assert!(neuron.bias >= 0.0 && neuron.bias < 1.0);
}

#[test]
fn test_neuron_forward() {
    let initialization = ConfigInitialization {
        method: "uniform-distribution".to_string(),
        seed: Some(42),
    };
    let mut initialization = get_initialization(&initialization);
    let neuron = Neuron::new(3, &mut initialization);
    let inputs = vec![0.5, 0.5, 0.5];
    neuron.forward(&inputs);
}

#[test]
fn test_neuron_backward() {
    let initialization = ConfigInitialization {
        method: "uniform-distribution".to_string(),
        seed: Some(42),
    };
    let lr = 0.01;
    let output_grad = 0.1;
    let inputs = vec![0.5, 0.5, 0.5];
    let mut initialization = get_initialization(&initialization);

    let mut neuron = Neuron::new(3, &mut initialization);
    let input_grad = neuron.backward(&inputs, output_grad, lr);

    assert_eq!(input_grad.len(), inputs.len());
}
