use core::mem::size_of;

pub fn slice_cast_mut<'a, A, B>(a : &'a mut [A]) -> &'a mut [B]{
    use core::slice::from_raw_parts_mut;

    let num_bytes = a.len() * size_of::<A>();
    let new_numel = num_bytes / size_of::<B>();
    let out_ptr = a.as_mut_ptr().cast::<B>();

    if (num_bytes % size_of::<B>()) != 0 {panic!();}
    if !out_ptr.is_aligned() {panic!();}

    unsafe{from_raw_parts_mut(out_ptr, new_numel)}
}


pub fn slice_cast<'a, A, B>(a : &'a [A]) -> &'a [B]{
    use core::slice::from_raw_parts;

    let num_bytes = a.len() * size_of::<A>();
    let new_numel = num_bytes / size_of::<B>();
    let out_ptr = a.as_ptr().cast::<B>();

    if (num_bytes % size_of::<B>()) != 0 {panic!();}
    if !out_ptr.is_aligned() {panic!();}

    unsafe{from_raw_parts(out_ptr, new_numel)}
}

// for these be very carful you don't pass in a double refernce by mistake
pub const fn any_as_u8_slice<'a, T: Sized>(p: &'a T) -> &'a [u8] {
    use core::slice::from_raw_parts;
    unsafe{
        from_raw_parts(
            (p as *const T) as *const u8,
            size_of::<T>(),
        )
    }
}

pub fn any_as_mut_u8_slice<'a, T: Sized>(p: &'a mut T) -> &'a mut [u8] {
    use core::slice::from_raw_parts_mut;
    unsafe{
        from_raw_parts_mut(
            (p as *mut T) as *mut u8,
            size_of::<T>(),
        )
    }
}

#[derive(Debug, Clone, Copy, Hash, std::cmp::PartialEq, std::cmp::Eq)]
pub struct Icords{
    pub x : i32,
    pub y : i32,
    pub z : i32,
}

impl std::ops::Index<usize> for Icords{
    type Output = i32;
    fn index(&self, index: usize) -> &Self::Output {
        let ptr : *const i32 = unsafe{core::mem::transmute(self)};
        unsafe{&*ptr.offset(index as isize)}
    }
}

impl Icords{
    pub const fn index(&self, index : u8) -> i32{
        let ptr : *const i32 = unsafe{core::mem::transmute(self)};
        unsafe{*ptr.offset(index as isize)}
    }

    pub const fn index_set(&self, index : u8, val : i32) -> Self{
        if index == 0{Icords::new(val, self.y, self.z)}
        else if index == 1{Icords::new(self.x, val, self.z)}
        else if index == 2{Icords::new(self.x, self.y, val)}
        else{panic!()}
    }

    pub fn max(&self) -> i32{
        return self.x.max(self.y).max(self.z);
    }

    pub fn min(&self) -> i32{
        return self.x.min(self.y).min(self.z);
    }

    pub const fn to_na(&self) -> nalgebra::Vector3<i32>{
        nalgebra::Vector3::new(self.x, self.y, self.z)
    }

    pub const fn to_array(&self) -> [i32; 3]{
        [self.x, self.y, self.z]
    }

    pub const fn from_na(x : nalgebra::Vector3<i32>) -> Self{
        unsafe{std::mem::transmute(x)}
    }
    
    pub const fn from_array(x : [i32; 3]) -> Self{
        unsafe{std::mem::transmute(x)}
    }
    pub const fn new(x : i32, y : i32, z : i32) -> Self{
        Self{x, y, z}
    }

    pub const fn div(&self, div : i32) -> Self{
        Self{x : self.x / div, y : self.y / div, z : self.z / div}
    }

    pub const fn mul(&self, mul : i32) -> Self{
        Self{x : self.x * mul, y : self.y * mul, z : self.z * mul}
    }

    pub const fn add(&self, add : i32) -> Self{
        Self{x : self.x + add, y : self.y + add, z : self.z + add}
    }

    pub const fn addv(&self, other : &Self) -> Self{
        Self{x : self.x + other.x, y : self.y + other.y, z : self.z + other.z}
    }
}

pub struct Timer{
    pub start : std::time::Instant,
    pub message : &'static str,
}


impl Timer{
    pub fn new(message : &'static str) -> Self{
        Self{start : std::time::Instant::now(), message}
    }
}

impl Drop for Timer{
    fn drop(&mut self) {
        let end = std::time::Instant::now();
        let duration = end - self.start;
        println!("{} : {:?}", self.message, duration);
    }
}