use std::io::{Read, Write};


#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexDataVariants {
    NoColorInfo,
    PerVertexColorInfo,
    UvBasedColorInfo,    
}
impl VertexDataVariants{pub const ALL :  [Self; 3] = [Self::NoColorInfo, Self::PerVertexColorInfo, Self::UvBasedColorInfo];}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Target {
    Release,
    Debug
}
impl Target{pub const ALL :  [Self; 2] = [Self::Release, Self::Debug];}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveType{
    TriangleList
}
impl PrimitiveType{pub const ALL :  [Self; 1] = [Self::TriangleList];}

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

pub fn compile_variant(options : ShaderOptions) -> std::io::Result<String>{
    let input_path = format!("{}/src/shaders/shade.slang", env!("CARGO_MANIFEST_DIR"));
    let output_path = format!("{}/{}/{}", env!("CARGO_MANIFEST_DIR"), TMP_SHADER_DIR, options.to_file_id());

    _ = std::fs::create_dir_all(format!("{}/{}", env!("CARGO_MANIFEST_DIR"), TMP_SHADER_DIR));
    let mut args: Vec<&str> = Vec::new();
    
    args.push(&input_path);
    args.extend_from_slice(&["-lang", "slang", "-std", "2026", "-warnings-as-errors", "all", "-fp-mode", "precise", "-target", "spirv"]);

    match options.vertex {
        VertexDataVariants::NoColorInfo => {},
        VertexDataVariants::PerVertexColorInfo => {args.push("-DPerVertexColor");}
        VertexDataVariants::UvBasedColorInfo => {args.push("-DPerVertexUvs");}
    };

    match options.primitives{
        PrimitiveType::TriangleList => {},
    };

    let slice = match options.target {
        Target::Release => {&["-O0", "-g3"]}
        Target::Debug => {&["-O3", "-g0"]}
    };
    args.extend_from_slice(slice);

    args.extend_from_slice(&["-o", output_path.as_str()]);

    let mut cmd = std::process::Command::new("slangc");
    cmd.args(args.as_slice());
    let done = cmd.output()?;

    if !done.status.success(){
        panic!("{}", String::from_utf8(done.stderr).unwrap());
    }

    Ok(output_path)
}

#[allow(unused)]
pub struct FileInfo{
    pub range : std::ops::Range<u64>,
    pub id : u64,
}

pub struct MegaFile<'a>{
    pub num_shaders : u64,
    pub shader_info : &'a [FileInfo],
    pub data : &'a [u8],
}



const TMP_SHADER_DIR : &str = "target/shaders";
const OUTPUT_FILE : &str = "megashader.bin";

const fn ref_to_u8_slice<'a, T>(val : &'a T) -> &'a [u8]{
    unsafe{
        let ptr : *const T = val;
        std::slice::from_raw_parts(ptr.cast::<u8>(), std::mem::size_of::<T>())
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

fn build_slang() -> std::io::Result<()>{
    let mut meta_data = Vec::new();
    let mut data = Vec::new();

    for vertex in VertexDataVariants::ALL{
        for target in Target::ALL{
            for primitives in PrimitiveType::ALL{
                let options = ShaderOptions{vertex, target, primitives};
                let out = compile_variant(options)?;

                let mut file = std::fs::File::open(out)?;
                let offset = data.len() as u64;
                file.read_to_end(&mut data)?;
                let range = offset..data.len() as u64;

                let info = FileInfo{id : options.to_file_id() as u64, range};
                meta_data.push(info);
                drop(file);
            }
        }
    }

    let length = data.len();
    let extra = length - ((length / 8) * 8);
    data.extend((0..extra).into_iter().map(|_|{0}));

    let base_dir = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), TMP_SHADER_DIR);
    std::fs::remove_dir_all(base_dir.as_str())?;
    
    let mut out_file = std::fs::File::create(format!("{}/{}", env!("CARGO_MANIFEST_DIR"), OUTPUT_FILE))?;
    out_file.write(ref_to_u8_slice(&(meta_data.len() as u64)))?;
    out_file.write(slice_t_to_u8_slice(meta_data.as_slice()))?;
    out_file.write(data.as_slice())?;

    Ok(())
}


// rust slang complier bindings exist https://github.com/FloatyMonkey/slang-rs
// but they are dynamicly linked and don't provide a way to conditionally use them at runtime
// so just use slang cli for now
// also won't work if complied in directory that cotains spaces because using cli 
fn main(){
    // println!("cargo::rerun-if-changed=src/shaders");
    // let _ = build_slang();
}
