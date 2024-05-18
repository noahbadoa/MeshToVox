pub type Fcords = nalgebra::Vector3<f32>;
use nalgebra as na;
use nalgebra_glm as glm;
use crate::{io::{ImageOrColor, Mesh}, octree::{Octree, OctreeCord}, utils::Icords};
use crate::utils::{closest_point_triangle, get_barycentric_coordinates};

pub trait VoxelStorage{
    fn store(&mut self, position : na::Vector3<i32>, val : image::Rgb<u8>);
}

impl VoxelStorage for Octree{
    fn store(&mut self, position : na::Vector3<i32>, val : image::Rgb<u8>) {
        let node = OctreeCord{cords : Icords::from_na(position), depth : self.depth};

        // bc floating point error some erroneous voxels
        if node.cords.min() < 1 || node.cords.max() >= ((1 << (self.depth + 1)) - 1){return;}

        self.insert(&node, val);
    }
}

fn dst(a : Fcords, b : Fcords) -> f32{
    let dif = a- b;
    (dif.x * dif.x) + (dif.y * dif.y) + (dif.z * dif.z)
}

fn voxelize_tri<const WIRE_FRAME : bool>(store : &mut impl VoxelStorage, shading : &Shading, tri_pos : [na::Vector3<f32>; 3]){
    if WIRE_FRAME{
        voxelize_line(store, shading, tri_pos[0], tri_pos[1]);
        voxelize_line(store, shading, tri_pos[1], tri_pos[2]);
        voxelize_line(store, shading, tri_pos[0], tri_pos[2]);
    }else{
        //don't know if min distance sorting acctualy improves perision
        const PARS : [(usize, usize); 3]= [(1, 2), (0, 2), (0, 1)];
        let dsts = PARS.map(|(a, b)|{dst(tri_pos[a], tri_pos[b])});
        let mut min_idx = 0;
        let mut min_dst = dsts[0];
        for idx in 1..3{
            if dsts[idx] > min_dst{min_dst = dsts[idx]; min_idx = idx;}
        }

        let (a, b) = PARS[min_idx];
        let dst1 = dsts[min_idx].sqrt();
        let num_steps = (dst1.ceil() as i32).max(1);
        let dir : na::Vector3<f32> = (tri_pos[b] - tri_pos[a]) / (num_steps as f32);
        
        for i in 0..(num_steps + 1){
            let start : na::Vector3<f32> = tri_pos[a] + (dir * i as f32);

            voxelize_line(store, shading, start, tri_pos[min_idx]);
        };
    }
}

fn voxelize_line(store : &mut impl VoxelStorage, shading : &Shading, p1 : na::Vector3<f32>, p2 : na::Vector3<f32>){

    let end = p2.try_cast::<i32>().unwrap();
    let ray_pos : na::Vector3<f32> = p1;

    if p1 == p2 {return;}
    let ray_dir : na::Vector3<f32> = glm::normalize(&(p2 - p1));
    if ray_dir.x.is_nan() || ray_dir.x.is_infinite(){return;}

    let temp_pos : na::Vector3<f32> = glm::floor(&ray_pos);
    let map_pos : na::Vector3<i32> = na::Vector3::new(temp_pos.x as i32, temp_pos.y as i32, temp_pos.z as i32);

    let len = glm::length(&ray_dir);
    let delta_dist : na::Vector3<f32> = glm::abs(&(glm::Vec3::new(len, len, len).component_div(&ray_dir)));

    let temp_dir : na::Vector3<f32> = glm::sign(&ray_dir);
    let ray_step : na::Vector3<i32> = na::Vector3::new(temp_dir.x as i32, temp_dir.y as i32, temp_dir.z as i32);
    
    let mut side_dist : na::Vector3<f32> = (glm::sign(&ray_dir)).component_mul(&(map_pos.cast() - ray_pos));
    side_dist = (side_dist + (glm::sign(&ray_dir) * 0.5f32).add_scalar(0.5f32)).component_mul(&delta_dist);
    
    voxelize_loop(store, shading, side_dist, delta_dist, ray_step, map_pos, end);


}

fn voxelize_loop(store : &mut impl VoxelStorage, shading : &Shading, mut side_dist : na::Vector3<f32>, delta_dist : na::Vector3<f32>, ray_step : na::Vector3<i32>, mut map_pos : na::Vector3<i32>, end : na::Vector3<i32>){

    loop{
        let color = shading.get_color(map_pos);
        store.store(map_pos, color);
        if map_pos == end{break;}
        
        if side_dist.x < side_dist.y {
            if side_dist.x < side_dist.z {
                side_dist.x += delta_dist.x;
                map_pos.x += ray_step.x;
            }
            else {
                side_dist.z += delta_dist.z;
                map_pos.z += ray_step.z;
            }
        }
        else {
            if side_dist.y < side_dist.z {
                side_dist.y += delta_dist.y;
                map_pos.y += ray_step.y;
            }
            else {
                side_dist.z += delta_dist.z;
                map_pos.z += ray_step.z;
            }
        }
    }
}


#[derive(Debug)]
struct TexturedShading<'a>{
    pub image : &'a image::RgbImage,
    pub tri_cords : [na::Vector3<f32>; 3],
    pub text_cords : [na::Vector2<f32>; 3],
}

#[derive(Debug)]
enum Shading<'a>{
    Texture(TexturedShading<'a>), 
    Color([u8; 3])
}

impl Shading<'_>{
    pub fn get_color(&self, map_pos : na::Vector3<i32>) -> image::Rgb<u8>{
        #[inline]
        pub fn wrap_around(x : f32) -> f32{
            let x = x % 1.0;
            if x < 0.{1. + x}
            else{x}
        }

        match self{
            Shading::Texture(texture) =>{

                let point = closest_point_triangle(map_pos.cast(), texture.tri_cords[0], 
                texture.tri_cords[1], texture.tri_cords[2]);

                let weights = get_barycentric_coordinates(point, texture.tri_cords[0], 
                    texture.tri_cords[1], texture.tri_cords[2]);
                
                let mut texture_cords = (texture.text_cords[0] * weights.x) + 
                    (texture.text_cords[1] * weights.y) + (texture.text_cords[2] * weights.z);

                texture_cords.x = wrap_around(texture_cords.x);
                texture_cords.y = wrap_around(texture_cords.y);

                let (x, y) = texture.image.dimensions();
                let x = (((x - 1) as f32) * texture_cords.x) as u32;
                let y = (((y - 1) as f32) * texture_cords.y) as u32;
                
                let color = texture.image.get_pixel(x, y);
                image::Rgb([color.0[0], color.0[1], color.0[2]])
            }

            Shading::Color(color) => {image::Rgb(*color)}
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum VoxelizationMode{
    Triangles,
    Lines,
    Points
}

pub fn voxelize_point(store : &mut impl VoxelStorage, point : Fcords){
    let point = point.map(|f|{f.round() as i32});
    store.store(point, image::Rgb([32, 32, 32]))
}

pub fn voxelize(mesh : &Mesh, size : u32, mode : VoxelizationMode) -> Octree{
    let num_tris = mesh.triangle.len();

    //leave one voxel gap around model to allow for inside/outside checking
    let max_size = size - 1;
    let depth = 31 - (size + 1).leading_zeros(); 

    let differnce : Fcords = mesh.bounds.max - mesh.bounds.min;
    let max_differnce = differnce.x.max(differnce.y).max(differnce.z);

    let scale : na::Vector3::<f64> = na::Vector3::new(max_size as f64, max_size as f64, max_size as f64) / max_differnce as f64;
    let mut tree = Octree::new(depth);

    for tri in 0..num_tris{
        let trii_pos = mesh.triangle[tri];
        let tri_pos = trii_pos.map(|vert|{

            ((vert - mesh.bounds.min).cast::<f64>().component_mul(&scale).add_scalar(1.0)).cast::<f32>()
        });

        let mat_id = mesh.extras[tri][0].material_idx;
        let materail = &mesh.materials[mat_id as usize];

        let shading = match materail {
            ImageOrColor::Image(img) =>{
                let refer = &mesh.extras[tri];
                let uv_position = refer.clone().map(|x|{x.uv.unwrap()});
                let texture = TexturedShading{image : img, tri_cords : tri_pos, text_cords: uv_position};
                
                Shading::Texture(texture)
            }
            ImageOrColor::Color(color) => {Shading::Color(*color)}
        };

        match mode{
            VoxelizationMode::Triangles => {
                voxelize_tri::<false>(&mut tree, &shading, tri_pos);
            }
            VoxelizationMode::Lines => {
                voxelize_tri::<true>(&mut tree, &shading, tri_pos);
            }
            VoxelizationMode::Points => {
                for point in tri_pos{
                    voxelize_point(&mut tree, point);
                }
            }
        }
    }

    tree
}
