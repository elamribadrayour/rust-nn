use std::collections::HashMap;

use rust_nn::activation::*;

#[test]
fn test_activation_relu_function() {
    let relu = ReLU {};
    assert_eq!(relu.function(-1.0), 0.0);
    assert_eq!(relu.function(0.0), 0.0);
    assert_eq!(relu.function(1.0), 1.0);
}

#[test]
fn test_activation_relu_derivative() {
    let relu = ReLU {};
    assert_eq!(relu.derivative(-1.0), 0.0);
    assert_eq!(relu.derivative(0.0), 0.0);
    assert_eq!(relu.derivative(1.0), 1.0);
}

#[test]
fn test_activation_linear_function() {
    let linear = Linear {};
    assert_eq!(linear.function(-1.0), -1.0);
    assert_eq!(linear.function(0.0), 0.0);
    assert_eq!(linear.function(1.0), 1.0);
}

#[test]
fn test_activation_linear_derivative() {
    let linear = Linear {};
    assert_eq!(linear.derivative(-1.0), 1.0);
    assert_eq!(linear.derivative(0.0), 1.0);
    assert_eq!(linear.derivative(1.0), 1.0);
}

#[test]
fn test_activation_sigmoid_function() {
    let sigmoid = Sigmoid {};
    assert!((sigmoid.function(0.0) - 0.5).abs() < 1e-7);
    assert!((sigmoid.function(2.0) - 0.880797).abs() < 1e-6);
}

#[test]
fn test_activation_sigmoid_derivative() {
    let sigmoid = Sigmoid {};
    let x = sigmoid.function(0.0);
    assert!((sigmoid.derivative(x) - x * (1.0 - x)).abs() < 1e-7);
}

#[test]
fn test_activation_heaviside_function() {
    let heaviside = Heaviside {};
    assert_eq!(heaviside.function(-1.0), 0.0);
    assert_eq!(heaviside.function(0.0), 0.0);
    assert_eq!(heaviside.function(1.0), 1.0);
}

#[test]
fn test_activation_heaviside_derivative() {
    let heaviside = Heaviside {};
    assert_eq!(heaviside.derivative(-1.0), 0.0);
    assert_eq!(heaviside.derivative(0.0), 0.0);
    assert_eq!(heaviside.derivative(1.0), 0.0);
}

#[test]
fn test_activation_gaussian_function() {
    let mut params = HashMap::new();
    params.insert("mu".to_string(), 0.0);
    params.insert("sigma".to_string(), 1.0);
    let gaussian = Gaussian::new(params);
    assert!((gaussian.function(0.0) - 1.0).abs() < 1e-7);
}

#[test]
fn test_activation_gaussian_derivative() {
    let mut params = HashMap::new();
    params.insert("mu".to_string(), 0.0);
    params.insert("sigma".to_string(), 1.0);
    let gaussian = Gaussian::new(params);
    assert!((gaussian.derivative(0.0) - 0.0).abs() < 1e-7);
}

#[test]
fn test_activation_multiquadratics_function() {
    let mut params = HashMap::new();
    params.insert("mu".to_string(), 0.0);
    params.insert("a".to_string(), 1.0);
    let multiquadratics = Multiquadratics::new(params);
    assert!((multiquadratics.function(0.0) - 1.0).abs() < 1e-7);
}

#[test]
fn test_activation_multiquadratics_derivative() {
    let mut params = HashMap::new();
    params.insert("mu".to_string(), 0.0);
    params.insert("a".to_string(), 1.0);
    let multiquadratics = Multiquadratics::new(params);
    assert!((multiquadratics.derivative(0.0) - 0.0).abs() < 1e-7);
}
