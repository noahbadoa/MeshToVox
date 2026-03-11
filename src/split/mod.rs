mod descriptors;
mod uniforms;
pub mod save;
mod data;
mod textures;
mod razter;

pub use uniforms::{Uniforms, Bounds, BoundingBox};
pub use descriptors::DescriptorState;
pub use razter::{RazterState, razter_fragment};

use crate::gpu::VulkanSingleton;
use crate::utils::Icords;

