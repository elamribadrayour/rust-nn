pub mod activation_impl;
pub mod relu;
pub mod sigmoid;
pub mod utils;

pub use crate::activation::activation_impl::Activation;
pub use crate::activation::relu::ReLU;
pub use crate::activation::sigmoid::Sigmoid;
pub use crate::activation::utils::get_activation;
