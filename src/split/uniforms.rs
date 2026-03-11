#[derive(Debug)]
#[repr(C)]
#[allow(dead_code)]
pub struct Uniforms{
    pub shift : [f32; 3],
    pub scale : f32,

    pub voxel_depth : u32,
    pub voxel_capacity : u32,
    pub voxel_scale : u32,
    other : u32,
}

impl Uniforms{
    pub fn new<T : Bounds>(uints : &[T], voxel_depth : u32, voxel_scale : u32, voxel_capacity : u32) -> Self{
        let bounding_box = get_scale(uints);
        let scale = [0, 1, 2].map(|idx|{
            bounding_box.max[idx] - bounding_box.min[idx]
        });
        
        let scale = 1.0 / scale[0].max(scale[1]).max(scale[2]);

        Self { shift : bounding_box.min, scale, voxel_capacity, voxel_depth, voxel_scale, other : 0}
    }
}

#[derive(Debug)]
pub struct BoundingBox{
    pub min : [f32; 3],
    pub max : [f32; 3]
}

pub trait Bounds {
    fn bounds(&self) -> BoundingBox;
}

fn get_scale<T : Bounds>(uints : &[T]) -> BoundingBox{
    let mut bounds = BoundingBox{min : [f32::MAX; 3], max : [f32::MIN; 3]};

    for uint in uints{
        let bounding_box = uint.bounds();
        bounds.min = [0, 1, 2].map(|idx|{bounds.min[idx].min(bounding_box.min[idx])});
        bounds.max = [0, 1, 2].map(|idx|{bounds.max[idx].max(bounding_box.max[idx])});
    }

    return bounds;
}
