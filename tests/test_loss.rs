use rust_nn::loss::get_loss;
use std::{collections::HashMap, f64::EPSILON};

#[test]
fn test_loss_mse_function() {
    let loss = get_loss("mse", HashMap::new());
    let y_true = vec![1.0, 0.0, 1.0];
    let y_pred = vec![0.9, 0.1, 0.8];
    let result = loss.function(&y_true, &y_pred);
    assert!((result - 0.019999999999999993).abs() < 1e-6);
}

#[test]
fn test_loss_mse_gradient() {
    let loss = get_loss("mse", HashMap::new());
    let y_true = vec![1.0, 0.0, 1.0];
    let y_pred = vec![0.9, 0.1, 0.8];
    let gradient = loss.gradient(&y_true, &y_pred);
    let expected_gradient = vec![
        0.06666666666666665,
        -0.06666666666666667,
        0.1333333333333333,
    ];
    for (g, e) in gradient.iter().zip(expected_gradient.iter()) {
        assert!((g - e).abs() < 1e-6);
    }
}

#[test]
fn test_loss_crossentropy_function() {
    let loss = get_loss("crossentropy", HashMap::new());
    let y_true = vec![1.0, 0.0, 1.0];
    let y_pred = vec![0.9, 0.1, 0.8];
    let result = loss.function(&y_true, &y_pred);
    assert!((result - 3.6043653389117156).abs() < 1e-6);
}

#[test]
fn test_loss_crossentropy_gradient() {
    let loss = get_loss("crossentropy", HashMap::new());
    let y_true = vec![1.0, 0.0, 1.0];
    let y_pred = vec![0.9, 0.1, 0.8];
    let gradient = loss.gradient(&y_true, &y_pred);
    let expected_gradient = vec![0.09999999999999998, -0.1, 0.19999999999999996];
    for (g, e) in gradient.iter().zip(expected_gradient.iter()) {
        assert!((g - e).abs() < 1e-6);
    }
}

#[test]
fn test_loss_binary_crossentropy_function() {
    let loss = get_loss("binary-crossentropy", HashMap::new());
    let y_true = vec![1.0, 0.0];
    let y_pred = vec![1.0, 1.0];
    let result = loss.function(&y_true, &y_pred);
    assert!((result - 36.04365338911715).abs() < EPSILON);
}

#[test]
fn test_loss_binary_crossentropy_gradient() {
    let loss = get_loss("binary-crossentropy", HashMap::new());
    let y_true = vec![1.0, 0.0, 1.0];
    let y_pred = vec![0.9, 0.1, 0.8];
    let gradient = loss.gradient(&y_true, &y_pred);
    let expected_gradient = vec![0.09999999999999998, -0.1, 0.19999999999999996];
    for (g, e) in gradient.iter().zip(expected_gradient.iter()) {
        assert!((g - e).abs() < EPSILON);
    }
}
