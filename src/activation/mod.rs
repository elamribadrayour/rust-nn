pub mod activation_impl;
pub mod gaussian;
pub mod heaviside;
pub mod linear;
pub mod multiquadratics;
pub mod relu;
pub mod sigmoid;
pub mod utils;

pub use crate::activation::activation_impl::Activation;
pub use crate::activation::gaussian::Gaussian;
pub use crate::activation::heaviside::Heaviside;
pub use crate::activation::linear::Linear;
pub use crate::activation::multiquadratics::Multiquadratics;
pub use crate::activation::relu::ReLU;
pub use crate::activation::sigmoid::Sigmoid;
pub use crate::activation::utils::get_activation;
