use std::collections::HashSet;
use crate::io::Vertex;
use crate::octree::{ALL_OCTREE_SIDES, OCT_PERMS, Octree, OctreeCord};
use crate::octree::octree_header;
use crate::utils::Icords;

pub type Fcords = nalgebra::Vector3<f32>;
type CordMap = HashSet<OctreeCord>;

#[derive(Debug, Clone)]
pub struct MeshNode{
    pub cords : Icords,
    pub dim : u8,
    pub positive : bool
}

impl MeshNode{
    const fn to_square(&self, size : i32) -> [Icords; 2]{
        let mut base: Icords = self.cords;

        if self.positive{
            base = base.index_set(self.dim, base.index(self.dim) + size);
        }

        let mut opposite = base;
        if self.dim != 0 {opposite.x += size}
        if self.dim != 1 {opposite.y += size}
        if self.dim != 2 {opposite.z += size}

        [base, opposite]
    }

    pub const fn to_triangles(&self, size : i32) -> [[Icords; 3]; 2]{
        let [base, opposite] = self.to_square(size);

        let mut corner1 = base;
        if self.dim != 0 {corner1.x += size}
        else if self.dim != 1 {corner1.y += size}
        // else if self.dim != 2 {corner1.z += size}

        let mut corner2 = base;
        if self.dim != 2 {corner2.z += size}
        else if self.dim != 1 {corner2.y += size}
        // else if self.dim != 0 {corner2.x += size}
        
        let tri1 = [base, corner1, opposite];
        let tri2 = [base, corner2, opposite];

        [tri1, tri2]
    }
}

#[derive(Debug)]
struct FillSpaceData<'a, 'b>{
    pub next : &'a mut CordMap,
    pub filled : &'b Octree,
    pub empty_tree : &'b mut Octree,
    pub start : OctreeCord,
    pub side : u8,
}

#[derive(Debug)]
struct FilledIterStruct{
    pub filled_offset : u32,
    pub empty_offset : u32,

    pub cords : OctreeCord,
}

impl Octree{
    fn recursive_collect_primary(adjacent : &FilledIterStruct, info : &mut FillSpaceData){
        let filled_header: u32 = info.filled.data[adjacent.filled_offset as usize];
        let axis = info.side % 3;
        let positive = (info.side / 3) == 0;

        let empty_header: u32 = info.empty_tree.data[adjacent.empty_offset as usize];
        let mut cords = adjacent.cords.cords.to_array();
        
        let octant_size = 1 << (info.filled.depth - (info.start.depth + 1));
        let negative_shift = 1 << (info.filled.depth - (adjacent.cords.depth + 1));
        let negative_works = (adjacent.cords.cords.to_array()[axis as usize] % negative_shift) == 0;

        let shift = if positive {octant_size} else if negative_works {-negative_shift} else {-1};
        let out = cords[axis as usize] as i32 + shift;
        if out < 0 || out >= (1 << info.empty_tree.depth) {return;}
        cords[axis as usize] = out;

        let cords = Icords::from_array(cords);
        let octant : u32 = info.filled.get_oct_inverted(cords, adjacent.cords.depth) as u32;
        if octree_header::get_final(empty_header, octant) {return;}

        let filled_exist = octree_header::get_exists(filled_header, octant);
        let empty_exist = octree_header::get_exists(empty_header, octant);
        
        if !filled_exist{
            octree_header::set_exists(&mut info.empty_tree.data[adjacent.empty_offset as usize], octant);
            octree_header::set_final(&mut info.empty_tree.data[adjacent.empty_offset as usize], octant);
            let output = OctreeCord{cords, depth : adjacent.cords.depth};
            let output = output.align(info.empty_tree.depth);

            info.next.insert(output);
            return;
        }else if (adjacent.cords.depth + 1) == info.empty_tree.depth{
            return;
        }

        let empty_offset = if empty_exist{
            info.empty_tree.data[(adjacent.empty_offset + 1 + octant) as usize]
        }else{
            let header_index = info.empty_tree.create_new_oct(0) as u32;
            octree_header::set_exists(&mut info.empty_tree.data[adjacent.empty_offset as usize], octant);
            info.empty_tree.data[(adjacent.empty_offset + 1 + octant) as usize] = header_index;

            header_index
        };

        let aligned_next = info.start.depth == adjacent.cords.depth;

        let mut next = FilledIterStruct{
            filled_offset : info.filled.data[(adjacent.filled_offset + 1 + octant) as usize],
            empty_offset,
            cords : OctreeCord{cords : adjacent.cords.cords, depth : adjacent.cords.depth + 1}
        };

        if aligned_next{
            let mut array = next.cords.cords.to_array();
            array[axis as usize] += if positive {octant_size} else {0};
            next.cords.cords = Icords::from_array(array);

            Self::recursive_collect_secondary(&next, info);
        }else{
            Self::recursive_collect_primary(&next, info)
        }
    }

    fn recursive_collect_secondary(adjacent : &FilledIterStruct, info : &mut FillSpaceData){
        let axis = info.side % 3;

        for oct in ALL_OCTREE_SIDES[axis as usize]{
            Self::recursive_collect_secondary_oct(adjacent, info, oct)
        }
    }

    fn recursive_collect_secondary_oct(adjacent : &FilledIterStruct, info : &mut FillSpaceData, oct : u8){
        let filled_header: u32 = info.filled.data[adjacent.filled_offset as usize];
        let axis = info.side % 3;
        let positive = (info.side / 3) == 0;

        let empty_header: u32 = info.empty_tree.data[adjacent.empty_offset as usize];

        let mut octant = oct as u32;
        if !positive {octant |= 1 << axis};

        if octree_header::get_final(filled_header, octant) {return;}
        if octree_header::get_final(empty_header, octant) {return;}

        let octant_size = 1 << (info.empty_tree.depth - (adjacent.cords.depth + 1));
        
        let mut position = [0usize, 1, 2].map(|index|{
            adjacent.cords.cords[index] + OCT_PERMS[oct as usize][index] * octant_size
        });
        
        let cords = Icords::from_array(position);
        if !positive {position[axis as usize] -= octant_size;}
        let effective_cords = Icords::from_array(position);

        if !(OctreeCord{cords : effective_cords, depth : adjacent.cords.depth}.in_bounds(info.filled.depth)) {return;}

        let filled_exist = octree_header::get_exists(filled_header, octant);
        let empty_exist = octree_header::get_exists(empty_header, octant);

        if !filled_exist{
            octree_header::set_exists(&mut info.empty_tree.data[adjacent.empty_offset as usize], octant);
            octree_header::set_final(&mut info.empty_tree.data[adjacent.empty_offset as usize], octant);
            let output = OctreeCord{cords : effective_cords, depth : adjacent.cords.depth};
            info.next.insert(output);
            return;
        }else if (adjacent.cords.depth + 1) == info.empty_tree.depth{
            return;
        }

        let empty_offset = if empty_exist{
            info.empty_tree.data[(adjacent.empty_offset + 1 + octant) as usize]
        }else{
            let header_index = info.empty_tree.create_new_oct(0) as u32;
            octree_header::set_exists(&mut info.empty_tree.data[adjacent.empty_offset as usize], octant);
            info.empty_tree.data[(adjacent.empty_offset + 1 + octant) as usize] = header_index;

            header_index
        };

        let next_octant = OctreeCord{cords, depth : adjacent.cords.depth + 1};
        let new_adjcent = FilledIterStruct{
            cords: next_octant, 
            filled_offset : info.filled.data[(adjacent.filled_offset + 1 + octant) as usize],
            empty_offset
        };

        Self::recursive_collect_secondary(&new_adjcent, info);
    }

    pub fn fill_space(&self, max_size : i32) -> Vec<Vertex>{
        let mut empty_tree = Octree::new(self.depth);
        let mut current : CordMap = HashSet::new();
        let mut next : CordMap = HashSet::new();


        let start = Icords::new(0, 0, 0);
        let start_depth = self.max_empty_depth(start);
        let start = OctreeCord{cords: start, depth : start_depth}.align(self.depth);
        empty_tree.insert(start, image::Rgb([0; 3]));
        current.insert(start);

        'outer : loop{
            for cord in &current{
                for i in 0..6{
                    let iter_info = FilledIterStruct{filled_offset : 0, empty_offset : 0, cords : OctreeCord { cords: cord.cords, depth: 0 }};
                    let mut thing: FillSpaceData = FillSpaceData{empty_tree : &mut empty_tree, next : &mut next, filled : self, side : i as u8, start : *cord};

                    Self::recursive_collect_primary(&iter_info, &mut thing);
                }
            }

            core::mem::swap(&mut current, &mut next);
            next.clear();
            if current.len() == 0 {break 'outer;}
        }

        Self::vis_empty(self, &empty_tree, max_size)
        // Self::vis_tree(&empty_tree)
    }

    fn vis_empty(full : &Self, empty : &Self, max_size : i32) -> Vec<Vertex>{
        let nodes = Self::empty_to_mesh(full, &empty);
        let mut output: Vec<Vertex> = Vec::new();

        for (node, color) in nodes{
            let triangles = node.to_triangles(1);

            let mapping = |x : Icords|{
                let position : Fcords = x.add(-1).to_na().cast::<f32>() / max_size as f32;
                let position = (position * 2.0).add_scalar(-1.0);
                Vertex{position, color}
            };

            let a : [Vertex; 3] = triangles[0].map(mapping);
            let b : [Vertex; 3] = triangles[1].map(mapping);

            output.extend(a);
            output.extend(b);
        }

        output
    }

    fn max_empty_depth(&self, cord : Icords) -> u32{
        let mut offset = 0;
        for depth in 0..(self.depth + 1){
            let oct = self.get_oct_inverted(cord, depth);
            let header = self.data[offset as usize];

            if !octree_header::get_exists(header, oct as u32){
                return depth;
            }

            offset = self.data[(offset + 1 + oct as u32) as usize];
        }

        unsafe{std::hint::unreachable_unchecked();}
    }

    fn empty_to_mesh(filled : &Self, empty : &Self) -> Vec<(MeshNode, image::Rgb<u8>)>{
        let mut mesh : Vec<(MeshNode, image::Rgb<u8>)> = Vec::new();
        let nodes = filled.collect_nodes();

        for (cord, value) in &nodes{
            let color = octree_header::to_color(*value);
            
            for i in 0..6{
                let mut adjacent = cord.cords.to_na();
                let dim: usize = (i / 2) as usize;
                let positive: bool = (i % 2) == 0;

                adjacent[dim] += if positive {1} else {-1};
                if adjacent[dim] < 0 {continue;}
                
                let cords = Icords::from_na(adjacent);
                let node = OctreeCord { cords, depth: filled.depth - 1};

                if empty.contains_point(&node).all(){
                    let mesh_node = MeshNode{cords : cord.cords, dim : dim as u8, positive};
                    mesh.push((mesh_node, color)); 
                }
            }
        }

        mesh
    }

    #[allow(unused)]
    pub fn vis_tree(&self, max_size : i32) -> Vec<Vertex>{
        let mut output: Vec<Vertex> = Vec::new();
        let nodes: Vec<(OctreeCord, u32)> = self.collect_nodes();
        for (node, color) in nodes{
            let (color, _) : (image::Rgb<u8>, u8) = unsafe{std::mem::transmute(color)};
    
            for side in 0..6{
                let positive = (side / 3) == 0;
                let axis = side % 3;
                let mesh = MeshNode{
                    cords : node.cords,
                    positive,
                    dim : axis
                };
                
                let triangles = mesh.to_triangles(1 << (self.depth - (node.depth + 1)));
                let mapping = |x : Icords|{
                    let position : Fcords = x.add(-1).to_na().cast::<f32>() / max_size as f32;
                    let position = (position * 2.0).add_scalar(-1.0);
                    Vertex{position, color}
                };
        
                let a : [Vertex; 3] = triangles[0].map(mapping);
                let b : [Vertex; 3] = triangles[1].map(mapping);
        
                output.extend(a);
                output.extend(b);
            }
        }
        
        output
    }
}

#[test]
fn space_filling_test(){
    use rand::RngExt;

    let mut rng = rand::rng();
    let depth = 8;

    let dim = 1 << depth;
    let max_size = dim - 2;
    let mut tree = Octree::new(depth);

    // sphere
    for x in 0..dim{
        for y in 0..dim{
            for z in 0..dim{
                let thingy = [x, y, z];
                let cords = Icords::from_array(thingy.map(|x|{x as i32}));
                let thing = thingy.map(|x|{
                    (x as f32 / dim as f32) - 0.5
                });

                let dst = (thing[0] * thing[0] + thing[1] * thing[1] + thing[2] * thing[2]).sqrt();
                if (dst - 0.4).abs() > (1.0 / dim as f32) {continue;}

                assert!(thingy[0].min(thingy[1]).min(thingy[2]) > 0);
                assert!(thingy[0].max(thingy[1]).max(thingy[2]) < (dim - 1));

                let color = rng.random::<[u8; 3]>();
                tree.insert(OctreeCord { cords, depth : depth - 1 }, image::Rgb(color));
            }
        }
    }

    let out = tree.fill_space(max_size);
    let test_path: &str = env!("CARGO_MANIFEST_DIR");

    let save_path = format!("{test_path}/test/test.gltf");
    _ = std::fs::remove_dir_all(std::path::Path::new(save_path.as_str()).parent().unwrap());
    crate::io::save_gltf(&out, save_path.as_str(), None, true).unwrap();
}

