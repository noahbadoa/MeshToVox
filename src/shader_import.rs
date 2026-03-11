// https://users.rust-lang.org/t/can-i-conveniently-compile-bytes-into-a-rust-program-with-a-specific-alignment/24049/2
#[repr(C)] // guarantee 'bytes' comes after '_align'
struct AlignedAs<Align, Bytes: ?Sized> {
    pub _align: [Align; 0],
    pub bytes: Bytes,
}

macro_rules! include_bytes_align_as {
    ($align_ty:ty, $path:literal) => {
        {  // const block expression to encapsulate the static
            // this assignment is made possible by CoerceUnsized
            const ALIGNED: &AlignedAs::<$align_ty, [u8]> = &AlignedAs {
                _align: [],
                bytes: *include_bytes!($path),
            };

            let size = std::mem::size_of::<$align_ty>();
            if (ALIGNED.bytes.len() % size) != 0{
                panic!("file size not multiple of type");
            }

            let length = ALIGNED.bytes.len() / size;
            unsafe{std::slice::from_raw_parts(ALIGNED.bytes.as_ptr().cast::<$align_ty>(), length)}
        }
    };
}


#[allow(unused)]
#[derive(Debug)]
struct FileInfo{
    pub range : std::ops::Range<u64>,
    pub id : u64,
}

#[derive(Debug)]
pub struct MegaFile<'a>{
    shader_info : &'a [FileInfo],
    data : &'a [u8],
}

const fn bytes_to_value<T>(slice : &[u8]) -> T{
    let mut out = std::mem::MaybeUninit::uninit();
    unsafe{
        std::ptr::copy_nonoverlapping(slice.as_ptr().cast::<T>(), out.as_mut_ptr(), 1);

        out.assume_init()
    }
}

const fn bytes_in_slice<T>(slice : &[T]) -> usize{
    std::mem::size_of::<T>() * slice.len()
}

const fn slice_t_to_u8_slice<'a, T>(val : &'a [T]) -> &'a [u8]{
    unsafe{
        std::slice::from_raw_parts(val.as_ptr().cast::<u8>(), bytes_in_slice(val))
    }
}


impl<'a> MegaFile<'a>{
    pub const fn from_slice(slice : &'a [u8]) -> Self{
        let num_shaders = bytes_to_value::<u64>(slice);
        let shader_info = unsafe{std::slice::from_raw_parts(slice.as_ptr().add(8).cast::<FileInfo>(), num_shaders as usize)};

        let data_start = 8 + std::mem::size_of::<FileInfo>() * num_shaders as usize;
        let data = unsafe{std::slice::from_raw_parts(slice.as_ptr().add(data_start), slice.len() - data_start)};

        Self { shader_info, data}
    }

    pub fn get_shader_source(&self, file_id : u64) -> Option<&[u8]>{
        let info = self.shader_info.iter().find(|x|{x.id == file_id});
        if info.is_none() {return None;}
        let info = info.unwrap();

        Some(&self.data[(info.range.start as usize)..(info.range.end as usize)])
    }
}

const IMPORT_SOURCE : &[u64] = include_bytes_align_as!(u64, "../megashader.bin");
pub const ALL_SHADERS : MegaFile = MegaFile::from_slice(slice_t_to_u8_slice(IMPORT_SOURCE));

pub mod shader_options{
    #[repr(u8)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum VertexDataVariants {
        NoColorInfo,
        PerVertexColorInfo,
        UvBasedColorInfo,    
    }
    #[repr(u8)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum Target {
        Release,
        Debug
    }
    
    #[repr(u8)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum PrimitiveType{
        TriangleList
    }
    
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ShaderOptions{
        pub vertex : VertexDataVariants,
        pub primitives : PrimitiveType,
        pub target : Target,
    }
    
    impl ShaderOptions{
        pub fn to_file_id(&self) -> u32{
            let num : u32 = unsafe{std::mem::transmute((*self, 0u8))};
            num
        }
    }
}


