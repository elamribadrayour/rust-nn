pub trait Activation {
    fn function(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
}
