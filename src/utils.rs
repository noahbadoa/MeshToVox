use core::mem::size_of;

pub fn slice_cast_mut<A, B>(a : &mut [A]) -> &mut [B]{
    use core::slice::from_raw_parts_mut;

    let num_bytes = a.len() * size_of::<A>();
    if (num_bytes % size_of::<B>()) != 0{panic!();}
    let new_numel = num_bytes / size_of::<B>();
    
    unsafe{from_raw_parts_mut(a.as_mut_ptr().cast::<B>(), new_numel)}
}

pub const fn slice_cast<A, B>(a : &[A]) -> &[B]{
    use core::slice::from_raw_parts;

    let num_bytes = a.len() * size_of::<A>();
    if (num_bytes % size_of::<B>()) != 0{panic!();}
    let new_numel = num_bytes / size_of::<B>();

    unsafe{from_raw_parts(a.as_ptr().cast::<B>(), new_numel)}
}

pub fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    use core::slice::from_raw_parts;
    unsafe{
        from_raw_parts(
            (p as *const T) as *const u8,
            size_of::<T>(),
        )
    }
}

pub fn any_as_mut_u8_slice<T: Sized>(p: &mut T) -> &mut [u8] {
    use core::slice::from_raw_parts_mut;
    unsafe{
        from_raw_parts_mut(
            (p as *mut T) as *mut u8,
            size_of::<T>(),
        )
    }
}

pub fn vec_cast<A, B>(x : Vec<A>) -> Vec<B>{
    let a_size = core::mem::size_of::<A>();
    let b_size = core::mem::size_of::<B>();
    let num_bytes = x.len() * a_size;
    assert!(num_bytes % b_size == 0);

    unsafe{
        let y : Vec<B> = Vec::from_raw_parts(x.as_ptr().cast_mut().cast::<B>(), num_bytes / b_size, x.capacity() / b_size);
        core::mem::forget(x);
        y
    }
}

#[derive(Debug, Clone, Copy, Hash, std::cmp::PartialEq, std::cmp::Eq)]
pub struct Icords{
    pub x : i32,
    pub y : i32,
    pub z : i32,
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

    pub const fn from_na(x : nalgebra::Vector3<i32>) -> Self{
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


type Vec3fa = nalgebra::Vector3<f32>;
//https://github.com/embree/embree/blob/master/tutorials/common/math/closest_point.h
pub fn closest_point_triangle(p : Vec3fa, a : Vec3fa, b : Vec3fa, c : Vec3fa) -> Vec3fa{
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = nalgebra_glm::dot(&ab, &ap);
    let d2 = nalgebra_glm::dot(&ac, &ap);
    if d1 <= 0.0 && d2 <= 0.0{return a};

    let bp = p - b;
    let d3 = nalgebra_glm::dot(&ab, &bp);
    let d4 = nalgebra_glm::dot(&ac, &bp);
    if d3 >= 0.0 && d4 <= d3{return b};

    let cp = p - c;
    let d5 = nalgebra_glm::dot(&ab, &cp);
    let d6 = nalgebra_glm::dot(&ac, &cp);
    if d6 >= 0.0 && d5 <= d6 {return c};

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0{
        let v = d1 / (d1 - d3);
        return a + v * ab;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0{
        let v = d2 / (d2 - d6);
        return a + v * ac;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0{
        let v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + v * (c - b);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return a + v * ab + w * ac;
}

//https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
pub fn get_barycentric_coordinates(p : Vec3fa, a : Vec3fa, b : Vec3fa, c : Vec3fa) -> Vec3fa{
    pub fn get_normal(a : Vec3fa, b : Vec3fa, c : Vec3fa) -> Vec3fa{
        let normed = nalgebra_glm::cross(&(b - a), &(c - a));
        nalgebra_glm::normalize(&normed)
    }
    let normal = get_normal(a, b, c);

    let area_abc = nalgebra_glm::dot(&normal, &nalgebra_glm::cross(&(b - a), &(c - a)));
    let area_pbc = nalgebra_glm::dot(&normal, &nalgebra_glm::cross(&(b - p), &(c - p)));
    let area_pca = nalgebra_glm::dot(&normal, &nalgebra_glm::cross(&(c - p), &(a - p)));

    let x = area_pbc / area_abc;
    let y = area_pca / area_abc;
    let bary = Vec3fa::new(x, y, 1.0 - (x + y));

    bary
}

pub struct Timer{
    pub start : std::time::Instant,
    pub message : &'static str,
}

pub static mut PERF : bool= false;

impl Timer{
    pub fn new(message : &'static str) -> Self{
        Self{start : std::time::Instant::now(), message}
    }
}

impl Drop for Timer{
    fn drop(&mut self) {
        if unsafe{PERF}{
            let end = std::time::Instant::now();
            let duration = end - self.start;
            println!("{} : {:?}", self.message, duration);
        }
    }
}