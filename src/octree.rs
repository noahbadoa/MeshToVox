use super::utils::Icords;

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct OctreeCord{
    pub cords : Icords,
    pub depth : u32,
}

impl OctreeCord{
    pub const fn align(&self, depth : u32) -> Self{
        let alignment = depth - (self.depth + 1);
        let alignment_mask = !((1 << alignment) - 1);
        
        let cords = Icords{x : self.cords.x & alignment_mask, y : self.cords.y & alignment_mask, z : self.cords.z & alignment_mask};

        Self { cords, depth : self.depth }
    }

    pub fn in_bounds(&self, depth : u32) -> bool{
        let min = self.cords.x.min(self.cords.y).min(self.cords.z);
        let max = self.cords.x.max(self.cords.y).max(self.cords.z);

        (min >= 0) && (max < (1 << depth))
    }
}

#[derive(Debug, Clone)]
pub struct UnCheckedVec<T : Copy>(Vec<T>);
impl<T : Copy> UnCheckedVec<T>{
    pub fn new() -> Self{Self(Vec::new())} 
    pub fn len(&self) -> usize{self.0.len()}

    pub fn reserve(&mut self, additional : usize) {self.0.reserve(additional)}
    pub unsafe fn set_len(&mut self, len : usize) {unsafe{self.0.set_len(len)}}

    pub fn get_mut(&mut self, index : usize) -> &mut T{unsafe{self.0.get_unchecked_mut(index)}}

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

type OcteeStorage = UnCheckedVec<u32>;
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
        let depth = self.depth - (i + 1);
        get_oct(cords, depth)
    }
}

pub mod octree_header{
    pub const EXISTS_OFFSET : u32 = 0;
    pub const FINAL_OFFSET : u32 = 8;

    pub const fn from_color(color : image::Rgb<u8>) -> u32{
        unsafe{std::mem::transmute((color, 0u8))}
    }
    pub fn to_color(offset : u32) -> image::Rgb<u8>{
        let (color, _tag) : (image::Rgb<u8>, u8) = unsafe{std::mem::transmute(offset)};

        color
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

const fn generate_mask(dim : u8, positive : bool) -> [u8; 4]{
    let mut counter = 0;
    let mut output_counter = 0;
    let mut output = [0; 4];

    while counter < 8{
        if ((counter >> dim) & 1) == (positive as u8){
            output[output_counter] = counter;

            output_counter += 1;
        }

        counter += 1;
    }

    output
}

const fn gen_octree_sides(positive : bool) -> [[u8; 4]; 3]{
    let mut counter = 0;
    let mut output = [[0; 4]; 3];

    while counter < 3{
        output[counter as usize] = generate_mask(counter, positive);
        counter += 1;
    }

    output
}



pub const ALL_OCTREE_SIDES : [[u8; 4]; 6] = unsafe {std::mem::transmute((gen_octree_sides(false), gen_octree_sides(true)))};
pub const OCT_PERMS : [Icords; 8] = gen_oct_perumations();

#[derive(Debug, PartialEq, Eq)]
pub enum Contains{
    All,
    Some,
    None
}

#[allow(dead_code)]
impl Contains{
    pub const fn all(&self) -> bool{
        if let Self::All = self {true} else {false}
    }
    pub const fn some(&self) -> bool{
        if let Self::Some = self {true} else {false}
    }
    pub const fn none(&self) -> bool{
        if let Self::None = self {true} else {false}
    }
    pub const fn some_or_all(&self) -> bool{
        self.all() | self.some()
    }
}

impl Octree{
    pub fn contains_point(&self, node : &OctreeCord) -> Contains{
        let mut currnet_pointer : u32 = 0;
        let mut current_oct;
        let mut current_header;

        for d in 0..(node.depth + 1){
            current_header = self.data[currnet_pointer as usize];
            current_oct = self.get_oct_inverted(node.cords, d) as u32;

            if !octree_header::get_exists(current_header, current_oct as u32) {return Contains::None}
            if octree_header::get_final(current_header, current_oct as u32) {return Contains::All}

            currnet_pointer = self.data[(currnet_pointer + 1 + current_oct) as usize];
        }

        Contains::Some
    }


    pub fn new(depth : u32) -> Self{
        let mut output = Self{depth, data : OcteeStorage::new()};
        output.create_new_oct(0);

        output
    }

    pub fn create_new_oct(&mut self, header : u32) -> usize{
        self.data.reserve(9);
        let old_len = self.data.len();

        unsafe{
            self.data.set_len(old_len + 9);
            self.data[old_len] = header;
        }

        #[cfg(debug_assertions)]
        for i in 0..8{
            self.data[old_len + 1 + i] = u32::MAX;
        }

        old_len
    }


    
    pub fn insert(&mut self, node : OctreeCord, value : image::Rgb<u8>) -> bool{
        let mut currnet_pointer : u32 = 0;
        
        for d in 0..node.depth{
            let current_header = self.data[currnet_pointer as usize];
            let oct = self.get_oct_inverted(node.cords, d) as u32;
            let current_node = currnet_pointer + 1 + oct as u32;
            
            currnet_pointer = if octree_header::get_exists(current_header, oct as u32){
                if octree_header::get_final(current_header, oct as u32) {return false}
            
                self.data[current_node as usize]
            }else{
                let next_pointer = self.create_new_oct(0) as u32;

                octree_header::set_exists(&mut self.data[currnet_pointer as usize], oct as u32);
                self.data[current_node as usize] = next_pointer;

                next_pointer
            };
        }

        let oct = self.get_oct_inverted(node.cords, node.depth) as u32;
        let next_node = currnet_pointer + 1 + oct as u32;
        let current_header = self.data.get_mut(currnet_pointer as usize);

        if octree_header::get_exists(*current_header, oct as u32) {return false}

        octree_header::set_exists(current_header, oct as u32);
        octree_header::set_final(current_header, oct as u32);

        self.data[next_node as usize] = octree_header::from_color(value);

        true
    }
    
    fn collect_recursive(&self, nodes : &mut Vec<(OctreeCord, u32)>, iter_level : IterStruct){

        let header = self.data[iter_level.offset as usize];

        for i in 0..8{    
            if !octree_header::get_exists(header, i) {continue;}

            let scale = 1 << (self.depth - (iter_level.cords.depth + 1));
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

#[test]
fn octree_test(){
    use rand::RngExt;

    let mut rng = rand::rng();
    let num_iter = 99_999;
    let depth = 6;

    let dim = 1 << depth;
    let mut map: std::collections::HashMap<[i32; 3], [u8; 3]> = std::collections::HashMap::new();
    let mut tree = Octree::new(depth);

    for _ in 0..num_iter{
        let cord = rng.random::<[u32; 3]>().map(|x|{(x % dim) as i32});
        let color = rng.random::<[u8; 3]>();

        if map.contains_key(&cord) {continue;}

        let a = map.insert(cord, color).is_none();
        let b = tree.insert(OctreeCord { cords: Icords::from_array(cord), depth : depth - 1 }, image::Rgb(color));

        assert_eq!(a, b);
    }

    let nodes = tree.collect_nodes();
    assert_eq!(nodes.len(), map.len());
    for (cord, value) in nodes{
        let (color, _) = unsafe{std::mem::transmute::<u32, (image::Rgb<u8>, u8)>(value)};
        assert_eq!(cord.depth, tree.depth - 1);
        assert_eq!(color.0, *map.get(&cord.cords.to_array()).unwrap());
    }


    for (key, _) in& map{
        let exists = tree.contains_point(&OctreeCord { cords: Icords::from_array(*key), depth : depth - 1 }).all();
        assert!(exists);
    }

    for _ in 0..num_iter{
        let cord = rng.random::<[u32; 3]>().map(|x|{(x % dim) as i32});
        if map.contains_key(&cord) { continue; }
        
        assert!(!tree.contains_point(&OctreeCord { cords: Icords::from_array(cord), depth : depth - 1 }).all());
    }
}

