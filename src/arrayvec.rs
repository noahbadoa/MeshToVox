pub struct ArrayVec<T, const N : usize>{
    pub data : [core::mem::MaybeUninit<T>; N],
    pub length : u32,
}

impl<T : std::fmt::Debug, const N : usize> std::fmt::Debug for ArrayVec<T, N>{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ArrayVec<{:?}> {{{:?}}}", N, self.as_slice())
    }
}

impl<T : PartialEq, const N : usize> PartialEq for ArrayVec<T, N>{
    fn eq(&self, other: &Self) -> bool {
        self.as_slice().eq(other.as_slice())
    }

    fn ne(&self, other: &Self) -> bool {
        self.as_slice().ne(other.as_slice())
    }
}

impl<T : Eq, const N : usize> Eq for ArrayVec<T, N>{}

impl<T, const N : usize> Default for ArrayVec<T, N>{
    fn default() -> Self {
        let data: [core::mem::MaybeUninit<T>; N] = unsafe{core::mem::MaybeUninit::uninit().assume_init()};
        
        Self { data , length: 0 }
    }
}

impl<T, const N : usize> Iterator for ArrayVec<T, N>{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.length == 0 {return None;}
        self.length -= 1;

        Some(unsafe{self.data[self.length as usize].assume_init_read()})
    }
}

impl<T : Clone, const N : usize> Clone for ArrayVec<T, N>{
    fn clone(&self) -> Self {
        let mut data: [core::mem::MaybeUninit<T>; N] = unsafe{core::mem::MaybeUninit::uninit().assume_init()};
        let ptr = data.as_ptr().cast::<T>();
        for i in 0..self.length{
            data[i as usize] = core::mem::MaybeUninit::new(unsafe{(&*ptr.offset(i as isize)).clone()});
        }

        Self { data: data, length: self.length }
    }
}
impl<T : Copy, const N : usize> Copy for ArrayVec<T, N>{}

impl<T, const N : usize> ArrayVec<T, N>{
    pub const fn new() -> Self{
        Self { data: unsafe{core::mem::MaybeUninit::uninit().assume_init()}, length: 0 }
    }

    pub const fn get(&self, idx : usize) -> Option<&T>{
        if self.length as usize <= idx {return None;}
        let t_ptr = self.data.as_ptr().cast::<T>();
        unsafe {Some(&*t_ptr.offset(idx as isize))}
    }

    pub fn extend(&mut self, iter : impl Iterator<Item = T>){
        for val in iter{
            self.push(val);
        }
    }

    pub fn as_slice(&self) -> &[T]{
        let slice : &[T] = unsafe{core::mem::transmute(self.data.as_slice())};
        &slice[0..self.length as usize]
    }

    pub fn as_mut_slice(&mut self) -> &mut [T]{
        let slice : &mut [T] = unsafe{core::mem::transmute(self.data.as_mut_slice())};
        &mut slice[0..self.length as usize]
    }

    pub const fn push(&mut self, val : T){
        let ptr = self.data[self.length as usize].as_mut_ptr();
        unsafe {core::ptr::write(ptr, val);}
        self.length += 1;
    }

    pub const fn pop(&mut self) -> Option<T>{
        if self.length == 0 {return None;}
        let data = unsafe {self.data[(self.length - 1) as usize].assume_init_read()};
        self.length -= 1;

        Some(data)
    }

    pub fn mapping<G>(&self, mut func : impl FnMut(&T) -> G) -> ArrayVec::<G, N>{
        let mut out = ArrayVec::<G, N>::new();
        for val in self.as_slice(){
            out.push(func(val));
        }

        out
    }

    pub const fn from_array<const M : usize>(array : [T; M]) -> Self{
        const{if M > N {panic!()}};
        
        let mut data: [core::mem::MaybeUninit<T>; N] = unsafe{core::mem::MaybeUninit::uninit().assume_init()};
        unsafe {core::ptr::copy_nonoverlapping(array.as_ptr(), data.as_mut_ptr().cast::<T>(), M);}
        core::mem::forget(array);

        Self { data, length: M as u32 }
    }
}

impl<T: core::marker::Copy, const N : usize> ArrayVec<T, N>{
    pub fn full(val : T, length : u32) -> Self{
        let mut data: [core::mem::MaybeUninit<T>; N] = unsafe{core::mem::MaybeUninit::uninit().assume_init()};
        let slice = unsafe{core::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<T>(), length as usize)};
        slice.fill(val);

        Self { data , length}
    }

    pub fn from_slice(slice : &[T]) -> Self{
        assert!(slice.len() <= N);
        let mut data: [core::mem::MaybeUninit<T>; N] = unsafe{core::mem::MaybeUninit::uninit().assume_init()};
        unsafe {core::ptr::copy_nonoverlapping(slice.as_ptr(), data.as_mut_ptr().cast::<T>(), slice.len());}

        Self { data, length: slice.len() as u32 }
    }

    pub fn shift_out(&mut self, shift : u32) -> T{
        let last = unsafe{self.data[shift as usize].assume_init()};
        self.as_mut_slice().rotate_left(shift as usize + 1);
        last
    }
}

use core::ops::{Index, IndexMut};

impl<T: core::marker::Copy, const N : usize> Index<u32> for ArrayVec<T, N>{
    type Output = T;

    fn index(&self, index: u32) -> &Self::Output {
        &self.as_slice()[index as usize]    
    }
}

impl<T: core::marker::Copy, const N : usize> IndexMut<u32> for ArrayVec<T, N>{
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        &mut self.as_mut_slice()[index as usize]
    }
}


impl<T, const N : usize> Index<usize> for ArrayVec<T, N>{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]    
    }
}

impl<T, const N : usize> IndexMut<usize> for ArrayVec<T, N>{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

impl<M, const N : usize> core::iter::FromIterator::<M> for ArrayVec<M, N>{
    fn from_iter<T: IntoIterator<Item = M>>(iter: T) -> Self {
        let mut out = Self::new();
        for val in iter{
            out.push(val);
        }

        out
    }
}
