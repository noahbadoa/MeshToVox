use crate::{io::save_as_magica_voxel, octree::{Octree, OctreeCord}, split::Icords};

fn frag_to_cord(frag : u64, voxel_depth : u32) -> (Icords, image::Rgb<u8>){
    let mask = !(u64::MAX << voxel_depth);
    let x = frag & mask;
    let y = (frag >> voxel_depth) & mask;
    let z = (frag >> (voxel_depth * 2)) & mask;
    let cord = Icords{x : x as i32, y : y as i32, z : z as i32};

    let color_offset = voxel_depth * 3;
    let color_mask = u8::MAX as u64;
    let color = [(frag >> color_offset) & color_mask, (frag >> (color_offset + 8)) & mask, (frag >> (color_offset + 8 * 2)) & mask];
    let color = color.map(|x|{x as u8});

    (cord, image::Rgb(color))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaveType{
    Gltf,
    GltfPruned,
    MagicaVoxel
}

pub fn save<S: AsRef<std::path::Path> + ?Sized>(voxel_fragments : &[u64], save_path : &S, kind : SaveType, voxel_scale : u32, voxel_depth : u32){
    match kind{
        SaveType::Gltf => {
            vis_fragments(voxel_fragments, voxel_scale, voxel_depth, save_path);
        }

        SaveType::GltfPruned => {
            vis_outer_fragments(voxel_fragments, voxel_scale, voxel_depth, save_path);
        }

        SaveType::MagicaVoxel => {
            vis_magica(voxel_fragments, voxel_depth, save_path);
        }
    }
}

fn vis_fragments<S: AsRef<std::path::Path> + ?Sized>(voxel_fragments : &[u64], voxel_scale : u32, voxel_depth : u32, save_path : &S){
    let unique_fragments = voxel_fragments.into_iter().map(|fragment|{
        frag_to_cord(*fragment, voxel_depth)
    }).collect::<std::collections::HashMap<_, _>>();


    let mut vertex = Vec::new();
    for (cords, color) in unique_fragments{
        for i in 0..6{
            let node = crate::space_filling::MeshNode{cords : cords, dim : i / 2, positive : (i % 2) == 0};
            let triangles = node.to_triangles(1 as i32);
            let verts : [Icords; 6] = unsafe{core::mem::transmute(triangles)};


            let verts = verts.map(|vert|{
                let position = (vert.add(-1).to_na().cast::<f64>() / voxel_scale as f64).cast::<f32>();
                let position = (position * 2.0).add_scalar(-1.0);

                crate::io::Vertex{position, color}
            });
            
            vertex.extend(verts);
        }
    }

    crate::io::save_gltf(&vertex, save_path, None, true).unwrap();
}

fn vis_outer_fragments<S: AsRef<std::path::Path> + ?Sized>(voxel_fragments : &[u64], voxel_scale : u32, voxel_depth : u32, save_path : &S) {
    let max_scale_for_depth = 1 << voxel_depth;
    let octree_depth = if max_scale_for_depth < (voxel_scale + 2) {voxel_depth + 1} else {voxel_depth};

    let mut octree = Octree::new(octree_depth);
    for fragment in voxel_fragments{
        let (cord, color) = frag_to_cord(*fragment, voxel_depth);
        let cord = Icords { x: cord.x + 1, y: cord.y + 1, z: cord.z + 1 };
        octree.insert(OctreeCord{cords : cord, depth : octree_depth - 1}, color);
    }

    let vertices = octree.fill_space(voxel_scale as i32);
    
    crate::io::save_gltf(&vertices, save_path, None, true).unwrap();
}

fn vis_magica<S: AsRef<std::path::Path> + ?Sized>(voxel_fragments : &[u64], voxel_depth : u32, save_path : &S){
    let iterator = voxel_fragments.iter().map(|fragment|{
        let (cord, color) = frag_to_cord(*fragment, voxel_depth);
        (cord.to_array().map(|x|{x as u32}), color)
    });
    
    save_as_magica_voxel(iterator, save_path, 1 << voxel_depth).unwrap();
}
