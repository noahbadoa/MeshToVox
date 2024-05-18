use std::collections::HashSet;
use crate::io::Vertex;
use crate::octree::{Octree, OctreeCord, ALL_OCTREE_SIDES};
use crate::octree::octree_header;
use crate::utils::{vec_cast, Icords};
use crate::voxelizer::Fcords;
type CordMap = HashSet<OctreeCord>;

#[derive(Debug, Clone)]
pub struct MeshNode{
    pub cords : Icords,
    pub dim : u8,
    pub positive : bool,
    pub depth : u8,
}


pub const fn bit_toggle(cords : Icords, depth : u32, oct : u32) -> Icords{
    pub const fn thing(dim : i32, depth : u32, set : bool) -> i32{
        if set{
            dim | (1 << depth)
        }else{
            dim & (!(1 << depth))
        }
    }
    
    Icords::new(
        thing(cords.x, depth, ((oct >> 0) & 1) != 0),
        thing(cords.y, depth, ((oct >> 1) & 1) != 0),
        thing(cords.z, depth, ((oct >> 2) & 1) != 0),
    )
}

impl MeshNode{
    
    pub const fn to_square(&self, octree_depth : u8) -> [Icords; 2]{
        let size = 1 << (octree_depth - self.depth);
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

    pub const fn to_triangles(&self, octree_depth : u8) -> [[Icords; 3]; 2]{
        let size = 1 << (octree_depth - self.depth);
        let [base, opposite] = self.to_square(octree_depth);

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
    pub empty_tree : &'b mut Octree,
}

#[derive(Debug)]
struct FilledIterStruct{
    pub filled_offset : u32,
    pub empty_offset : u32,

    pub cords : OctreeCord,
    pub side : u8,
}

impl Octree{
    pub fn fill_space(&self, max_size : u32) -> Vec<Vertex>{
        let mut empty_tree = Octree::new(self.depth);
        let mut current : CordMap = HashSet::new();
        let mut next : CordMap = HashSet::new();

        let start = Icords::new(0, 0, 0);
        let depth = self.insert_max_start(&mut empty_tree, start);
        let start = OctreeCord{cords: start, depth};
        current.insert(start);

        'outer : loop{
            for cord in &current{
                for i in 0..6{
                    let adjcent = self.min_adjcent_depth(&mut empty_tree, &mut next, cord, i);
                    if adjcent.is_none(){continue;}

                    let adjcent: FilledIterStruct = adjcent.unwrap();
                    let mut thing: FillSpaceData = FillSpaceData{empty_tree : &mut empty_tree, next : &mut next};
                    self.recursive_collect(&adjcent, &mut thing);
                }
            }

            core::mem::swap(&mut current, &mut next);
            next.clear();
            if current.len() == 0{break 'outer;}
        }

        let nodes = Self::empty_to_mesh(self, &empty_tree);

        let triangles : Vec<_> = nodes.iter().map(|(node, color)|{
            let color = *color;
            let triangles = node.to_triangles(self.depth as u8);

            let mapping = |x : Icords|{
                let position : Fcords = x.add(-1).to_na().cast::<f32>() / max_size as f32;
                let position = (position * 2.0).add_scalar(-1.0);
                Vertex{position, color}
            };

            let a : [Vertex; 3] = triangles[0].map(mapping);
            let b : [Vertex; 3] = triangles[1].map(mapping);

            [a, b]
        }).collect();
        let triangles : Vec<Vertex> = vec_cast(triangles);

        triangles
    }

    fn insert_max_start(&self, empty_tree : &mut Self, start : Icords) -> u32{
        let mut empty_pointer : u32 = 0;
        let mut filled_pointer : u32 = 0;

        for d in 0..(self.depth + 1){
            let filled_header = self.data[filled_pointer as usize];
            let empty_header = &mut empty_tree.data[empty_pointer as usize];
            let oct = self.get_oct_inverted(start, d) as u32;

            //if octree_header::get_final(filled_header, oct as u32){panic!();}

            if !octree_header::get_exists(filled_header, oct as u32){
                octree_header::set_final(empty_header, oct as u32);
                octree_header::set_exists(empty_header, oct as u32);

                return d;
            }

            if !octree_header::get_exists(*empty_header, oct as u32){
                octree_header::set_exists(empty_header, oct as u32);

                let next = empty_tree.create_empty_oct(d);
                empty_tree.data[(empty_pointer + 1 + oct) as usize] = next as u32;
            }

            filled_pointer = self.data[(filled_pointer + 1 + oct) as usize];
            empty_pointer = empty_tree.data[(empty_pointer + 1 + oct) as usize];
        }
        //panic!();
        unsafe{core::hint::unreachable_unchecked()}
    }

    fn min_adjcent_depth(&self, empty : &mut Self, next : &mut CordMap, cord : &OctreeCord, side : u8) -> Option<FilledIterStruct>{
        let max_size = 1 << (self.depth + 1);
        let min_octant_size = 1 << (self.depth - cord.depth);

        let mut base = cord.cords.to_na();
        let dim = side % 3;
        base[dim as usize] += if side < 3{min_octant_size}else{-1};

        if (base[dim as usize] >= max_size) || (base[dim as usize] < 0) {return None;}
        
        let adjcent = Icords::from_na(base);
        
        let mut empty_offset : u32 = 0;
        let mut filled_offset : u32 = 0;

        for d in 0..(cord.depth + 1){
            let adjacent_oct = self.get_oct_inverted(adjcent, d);

            let empty_header = empty.data[empty_offset as usize];
            let filled_header = self.data[filled_offset as usize];

            if octree_header::get_final(filled_header | empty_header, adjacent_oct as u32){return None}

            if !octree_header::get_exists(filled_header, adjacent_oct as u32){
                let cord = OctreeCord{cords : Icords::from_na(base), depth : d};
                next.insert(cord);

                octree_header::set_exists(&mut empty.data[empty_offset as usize], adjacent_oct as u32);
                octree_header::set_final(&mut empty.data[empty_offset as usize], adjacent_oct as u32);

                return None;
            }

            if !octree_header::get_exists(empty_header, adjacent_oct as u32){
                let next = empty.create_empty_oct(d);
                octree_header::set_exists(&mut empty.data[empty_offset as usize], adjacent_oct as u32);
                empty.data[(empty_offset + 1 + adjacent_oct as u32) as usize] = next as u32;
            }

            empty_offset = empty.data[(empty_offset + 1 + adjacent_oct as u32) as usize];
            filled_offset = self.data[(filled_offset + 1 + adjacent_oct as u32) as usize];
        }

        let base = OctreeCord{cords : adjcent, depth : (cord.depth + 1)};
        let new_cord = FilledIterStruct{cords : base, empty_offset, filled_offset, side};

        return Some(new_cord);
    }

    fn recursive_collect(&self, adjcent : &FilledIterStruct, info : &mut FillSpaceData){
        let empty_header = info.empty_tree.data[adjcent.empty_offset as usize];
        let filled_header = self.data[adjcent.filled_offset as usize];

        for oct in ALL_OCTREE_SIDES[adjcent.side as usize]{
            let oct = oct as u32;
            if octree_header::get_final(filled_header, oct){continue;}
            if octree_header::get_final(empty_header, oct){continue;}

            let pos = bit_toggle(adjcent.cords.cords, self.depth - adjcent.cords.depth, oct);
            let octant = OctreeCord{cords : pos, depth : adjcent.cords.depth};

            if !octree_header::get_exists(filled_header, oct){
                octree_header::set_exists(&mut info.empty_tree.data[adjcent.empty_offset as usize], oct);
                octree_header::set_final(&mut info.empty_tree.data[adjcent.empty_offset as usize], oct);

                let out = octant.simplify(self.depth);
                info.next.insert(out);
                continue;
            }

            if !octree_header::get_exists(empty_header, oct){
                octree_header::set_exists(&mut info.empty_tree.data[adjcent.empty_offset as usize], oct);
                let next = info.empty_tree.create_empty_oct(adjcent.cords.depth);
                info.empty_tree.data[(adjcent.empty_offset + 1 + oct) as usize] = next as u32;
            }

            let filled_offset = self.data[(adjcent.filled_offset + 1 + oct) as usize];
            let empty_offset = info.empty_tree.data[(adjcent.empty_offset + 1 + oct) as usize];

            let next_octant = OctreeCord{cords : pos, depth : adjcent.cords.depth + 1};
            let new_adjcent = FilledIterStruct{cords: next_octant, filled_offset, empty_offset, side : adjcent.side};
            self.recursive_collect(&new_adjcent, info);
        }
    }

    fn empty_to_mesh(filled : &Self, empty : &Self) -> Vec<(MeshNode, image::Rgb<u8>)>{
        let mut mesh : Vec<(MeshNode, image::Rgb<u8>)> = Vec::new();

        let nodes = filled.collect_nodes();
        let max_size = 1 << (filled.depth + 1);

        for (cord, value) in &nodes{
            let color = octree_header::to_color(*value);
            
            for i in 0..6{
                let mut adjcent = cord.cords.to_na();
                let dim: usize = (i / 2) as usize;
                let positive: bool = (i % 2) == 0;

                adjcent[dim] += if positive {1}else{-1};
                if adjcent[dim] >= max_size || adjcent[dim] < 0{continue;}
                let cords = Icords::from_na(adjcent);
                let node = OctreeCord { cords, depth: filled.depth};

                if empty.contains_point(&node){
                    let mesh_node = MeshNode{cords : cord.cords, dim : dim as u8, positive, depth : filled.depth as u8};
                    mesh.push((mesh_node, color)); 
                }
            }
        }

        mesh
    }

    fn create_new_empty_oct(&mut self) -> usize{
        let old_len = self.data.len();
        let mut header = 0;
        octree_header::set_header_tag(&mut header);
        self.data.push(header);
    
        old_len
    }

    fn create_empty_oct(&mut self, depth : u32) -> usize{
        if self.depth == depth {self.create_new_empty_oct()}
        else{self.create_new_oct(0)}
    }
}
