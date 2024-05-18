use std::io::{Write, Read};
use crate::utils::{any_as_u8_slice, slice_cast, slice_cast_mut, vec_cast};
use super::utils::{any_as_mut_u8_slice, Icords};

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct OctreeCord{
    pub cords : Icords,
    pub depth : u32,
}

impl OctreeCord{
    pub const fn zero(depth : u32) -> Self{
        Self{cords : Icords::new(0, 0, 0), depth}
    }

    pub const fn simplify(&self, max_depth : u32) -> Self{
        let cords = Icords::new(
            self.cords.x & !((1 << (max_depth - self.depth)) - 1),
            self.cords.y & !((1 << (max_depth - self.depth)) - 1),
            self.cords.z & !((1 << (max_depth - self.depth)) - 1)
        );

        Self{cords, depth : self.depth}
    }

    pub const fn is_simple(&self, max_depth : u32) -> bool{
        const fn vaildate(dim : i32, depth : u32, max_depth : u32) -> bool{
            dim.trailing_zeros() >= (max_depth - depth)
        }

        let a = vaildate(self.cords.x, self.depth, max_depth);
        let b = vaildate(self.cords.y, self.depth, max_depth);
        let c = vaildate(self.cords.z, self.depth, max_depth);

        a && b && c
    }
}

//replace with macro
#[derive(Debug, Clone)]
pub struct UnCheckedVec<T : Copy>(Vec<T>);
impl<T : Copy> UnCheckedVec<T>{
    pub fn new() -> Self{Self(Vec::new())} 
    pub fn with_capacity(capacity: usize) -> Self{Self(Vec::with_capacity(capacity))}


    pub fn push(&mut self, value : T){self.0.push(value)}
    pub fn len(&self) -> usize{self.0.len()}


    pub fn as_slice(&self) -> &[T] {self.0.as_slice()}
    pub fn as_mut_slice(&mut self) -> &mut [T] {self.0.as_mut_slice()}

    pub fn reserve(&mut self, additional : usize){self.0.reserve(additional)}
    pub unsafe fn set_len(&mut self, len : usize){self.0.set_len(len)}

    pub fn get_mut(&mut self, index : usize) -> &mut T{unsafe{self.0.get_unchecked_mut(index)}}

    pub fn vec_cast<B: std::marker::Copy>(self) -> UnCheckedVec<B>{UnCheckedVec::<B>(vec_cast::<T, B>(self.0))}
}

impl<T : Copy> std::ops::Index<usize> for UnCheckedVec<T>{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        unsafe{self.0.get_unchecked(index)}
    }
}
impl<T : Copy> std::ops::IndexMut<usize> for UnCheckedVec<T>{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe{self.0.get_unchecked_mut(index)}
    }
}

pub type OcteeStorage = UnCheckedVec<u32>;
#[derive(Debug, Clone)]
pub struct Octree{
    pub data : OcteeStorage,
    pub depth : u32,
}

pub const fn get_oct(cords : Icords, depth : u32) -> i32{
    let x = (cords.x >> depth) & 1;
    let y = (cords.y >> depth) & 1;
    let z = (cords.z >> depth) & 1;

    x | (y << 1) | (z << 2)
}

impl Octree{

    pub const fn get_oct_inverted(&self, cords : Icords, i : u32) -> i32{
        let depth = self.depth - i;
        get_oct(cords, depth)
    }
}

pub mod octree_header{
    pub const EXISTS_OFFSET : u32 = 0;
    pub const FINAL_OFFSET : u32 = 8;
    pub const EMPTY_OFFSET : u32 = 16;
    pub const TAG_OFFSET : u32 = 24;

    pub const COLOR_TAG : u8 = 118;
    pub const HEADER_TAG : u8 = 68;

    pub const fn from_color(color : image::Rgb<u8>) -> u32{
        unsafe{std::mem::transmute((color, COLOR_TAG))}
    }

    pub fn to_color(offset : u32) -> image::Rgb<u8>{
        let (color, _tag) : (image::Rgb<u8>, u8) = unsafe{std::mem::transmute(offset)};

        color
    }

    pub const fn get_empty(header : u32, idx : u32) -> bool{
        ((header >> (EMPTY_OFFSET + idx)) & 1) != 0
    }

    pub fn set_empty(header : &mut u32, idx : u32){
        *header |= 1 << (EMPTY_OFFSET + idx);
    }

    pub fn set_header_tag(header : &mut u32){
        *header |= (HEADER_TAG as u32) << TAG_OFFSET;
    }

    pub const fn is_header(header : u32) -> bool{
        (header >> TAG_OFFSET) == HEADER_TAG as u32
    }

    pub const fn get_exists(header : u32, idx : u32) -> bool{
        ((header >> (idx + EXISTS_OFFSET)) & 1) != 0
    }

    pub fn set_exists(header : &mut u32, idx : u32){
        *header |= 1 << (idx + EXISTS_OFFSET);
    }
    
    pub const fn get_final(header : u32, idx : u32) -> bool{
        ((header >> (idx + FINAL_OFFSET)) & 1) != 0
    }

    pub fn set_final(header : &mut u32, idx : u32){
        *header |= 1 << (idx + FINAL_OFFSET)
    }
}

#[derive(Debug, Clone)]
pub struct IterStruct{
    pub offset : u32,
    pub cords : OctreeCord,
}

const fn gen_oct_perumations() -> [Icords; 8]{
    let mut cube : [Icords; 8] = [Icords::new(0, 0, 0); 8];
    let mut counter : i32 = 0;

    while counter < 8{
        cube[counter as usize] = Icords::new(
            counter & 1,
            (counter >> 1) & 1,
            (counter >> 2) & 1,
        );

        counter += 1;
    }

    cube
}

const fn gen_octree_sides(positive : bool) -> [[u8; 4]; 3]{
    const fn generate_mask(dim : u8, positive : bool) -> [u8; 4]{
        let mut counter = 0;
        let mut output_counter = 0;
        let mut output: [u8; 4] = unsafe{std::mem::MaybeUninit::uninit().assume_init()};
    
        loop{
            if counter == 8{break;}
            if ((counter >> dim) & 1) == (positive as u8){
                output[output_counter] = counter;

                output_counter += 1;
            }
    
            counter += 1;
        }
    
        output
    }

    let mut counter = 0;
    let mut output: [[u8; 4]; 3]= unsafe{std::mem::MaybeUninit::uninit().assume_init()};

    loop{
        if counter == 3{break;}

        output[counter as usize] = generate_mask(counter, positive);

        counter += 1;
    }

    output
}

pub const ALL_OCTREE_SIDES : [[u8; 4]; 6] = unsafe {std::mem::transmute((gen_octree_sides(false), gen_octree_sides(true)))};
pub const OCT_PERMS : [Icords; 8] = gen_oct_perumations();

impl Octree{
    pub fn new(depth : u32) -> Self{
        let mut output = Self{depth, data : OcteeStorage::new()};
        output.create_new_oct(0);


        output
    }

    pub fn with_capacity(depth : u32, capacity : usize) -> Self{
        let mut output = Self{depth, data : OcteeStorage::with_capacity(capacity)};
        output.create_new_oct(0);

        output
    }

    pub fn save_as_octree(&self, fname : &str){
        let mut file = std::fs::File::create(fname).unwrap();
        let meta_data = (self.depth, self.data.len());

        let meta_slice = any_as_u8_slice(&meta_data);
        let main_data = slice_cast(self.data.as_slice());

        file.write_all(meta_slice).unwrap();
        file.write_all(main_data).unwrap();
    }

    pub fn load_octree(fname : &str) -> Self{
        let mut file = std::fs::File::open(fname).unwrap();
        let mut metadata : (u32, usize) = (0, 0);

        file.read_exact(any_as_mut_u8_slice(&mut metadata)).unwrap();

        let mut data : OcteeStorage = OcteeStorage::with_capacity(metadata.1);
        unsafe{data.set_len(metadata.1)}

        file.read_exact(slice_cast_mut(data.as_mut_slice())).unwrap();

        Self{depth : metadata.0, data}
    }

    pub fn create_new_oct(&mut self, mut header : u32) -> usize{
        self.data.reserve(9);
        let old_len = self.data.len();
        octree_header::set_header_tag(&mut header);

        unsafe{
            self.data.set_len(old_len + 9);
            self.data[old_len] = header;
            for i in 0..8{
                self.data[old_len + 1 + i] = 69420420;
            }
        }
        old_len
    }

    pub fn contains_point(&self, node : &OctreeCord) -> bool{
        let mut currnet_pointer : u32 = 0;
        let mut current_oct;
        let mut current_header;

        for d in 0..(node.depth + 1){
            current_header = self.data[currnet_pointer as usize];
            current_oct = self.get_oct_inverted(node.cords, d) as u32;

            if !octree_header::get_exists(current_header, current_oct as u32){return false}
            if octree_header::get_final(current_header, current_oct as u32){return true}

            currnet_pointer = self.data[(currnet_pointer + 1 + current_oct) as usize];
        }
        false
    }
    
    pub fn contains_exact(&self, node : &OctreeCord) -> bool{
        let mut currnet_pointer : u32 = 0;
        let mut current_oct;
        let mut current_header;

        for d in 0..node.depth{
            current_header = self.data[currnet_pointer as usize];
            current_oct = self.get_oct_inverted(node.cords, d) as u32;

            if !octree_header::get_exists(current_header, current_oct as u32){return false}
            if octree_header::get_final(current_header, current_oct as u32){return false}

            currnet_pointer = self.data[(currnet_pointer + 1 + current_oct) as usize];
        }

        current_header = self.data[currnet_pointer as usize];
        current_oct = self.get_oct_inverted(node.cords, node.depth) as u32;

        if octree_header::get_final(current_header, current_oct as u32){return true}
        else{false}
    }

    pub fn insert(&mut self, node : &OctreeCord, value : image::Rgb<u8>) -> Option<u32>{
        if node.depth > self.depth{return None;}

        let mut currnet_pointer : u32 = 0;
        let mut current_oct = self.get_oct_inverted(node.cords, 0) as u32;
        let mut current_node = currnet_pointer + 1 + current_oct as u32;
        let mut inserted = true;
        
        for d in 0..node.depth{
            let current_header = self.data[currnet_pointer as usize];
            let next_oct = self.get_oct_inverted(node.cords, d + 1) as u32;
            
            currnet_pointer = if octree_header::get_exists(current_header, current_oct as u32) && inserted{
                if octree_header::get_final(current_header, current_oct as u32){return None}
            
                self.data[current_node as usize]
            }else{
                let mut next_header = 0;
                octree_header::set_exists(&mut next_header, next_oct as u32);
                let next_pointer = self.create_new_oct(next_header) as u32;


                octree_header::set_exists(&mut self.data[currnet_pointer as usize], current_oct as u32);
                self.data[current_node as usize] = next_pointer;
                inserted = false;

                next_pointer
            };

            current_node = currnet_pointer + 1 + next_oct as u32;
            current_oct = next_oct;
        }

        let next_node = currnet_pointer + 1 + current_oct as u32;
        let current_header = self.data.get_mut(currnet_pointer as usize);

        if octree_header::get_exists(*current_header, current_oct as u32) && inserted{return None}

        octree_header::set_exists(current_header, current_oct as u32);
        octree_header::set_final(current_header, current_oct as u32);

        self.data[next_node as usize] = octree_header::from_color(value);

        Some(next_node)
    }
    
    //replace with non recursive implementation
    fn collect_recursive(&self, nodes : &mut Vec<(OctreeCord, u32)>, iter_level : IterStruct){

        let header = self.data[iter_level.offset as usize];

        for i in 0..8{    
            if !octree_header::get_exists(header, i){continue;}

            let scale = 1 << (self.depth - iter_level.cords.depth);
            let cords = OCT_PERMS[i as usize].mul(scale);
            let new_position = iter_level.cords.cords.addv(&cords);
            let offset = self.data[(iter_level.offset + 1 + i) as usize];
            
            if octree_header::get_final(header, i){
                let cords = OctreeCord{cords : new_position, depth : iter_level.cords.depth};
                nodes.push((cords, offset));
            }else{
                let cords = OctreeCord{cords : new_position, depth : iter_level.cords.depth + 1};
                let new_iter : IterStruct = IterStruct{cords, offset};
                self.collect_recursive(nodes, new_iter);
            }
        }
    }

    pub fn collect_nodes(&self) -> Vec<(OctreeCord, u32)>{
        let length = self.data.len() / 9;
        let mut collected : Vec<(OctreeCord, u32)> = Vec::with_capacity(length);
        let cords  = OctreeCord{cords : Icords::new(0, 0, 0), depth : 0};
        let first_iter = IterStruct{cords, offset : 0};

        self.collect_recursive(&mut collected, first_iter);


        collected
    }
}
