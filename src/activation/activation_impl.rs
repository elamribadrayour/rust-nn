pub trait Activation {
    fn function(&self, x: &[f64]) -> Vec<f64>;
    fn derivative(&self, x: &[f64]) -> Vec<f64>;
}
