pub mod octree;
pub mod voxelizer;
pub mod io;
pub mod utils;
pub mod space_filling;

use crate::voxelizer::{voxelize, VoxelizationMode};

enum InputType{GlbGltf}
impl InputType{
    pub fn from_file(file : &str) -> Option<Self>{
        let exten = get_extension(file).unwrap();
        if exten == "glb" || exten == "gltf"{return Some(Self::GlbGltf);}
        None
    }
}
enum OutputType{
    Gltf,
    MagicaVoxel,
}
impl OutputType{
    pub fn from_file(file : &str) -> Option<Self>{
        let exten = get_extension(file).unwrap();
        if exten == "gltf"{return Some(Self::Gltf);}
        if exten == "vox"{return Some(Self::MagicaVoxel);}
        None
    }
}

fn voxelize_mesh(args : &Args) -> Result<(), io::Error>{
    let input_type = InputType::from_file(&args.f);
    if input_type.is_none(){return Err(io::Error::UnsupportedFileType)}

    let output_type = OutputType::from_file(&args.o);
    if output_type.is_none(){return Err(io::Error::UnsupportedFileType)}

    let _1 = Timer::new("loading gltf");
    let mesh  = match input_type.unwrap(){
        InputType::GlbGltf => {io::load_gltf(&args.f)?}
    };
    drop(_1);

    let _2 = Timer::new("voxelization");
    let data = voxelize(&mesh, args.dim, VoxelizationMode::Triangles);
    drop(_2);

    match output_type.unwrap(){
        OutputType::Gltf => {
            data.save_as_gltf::<String>(&args.o, mesh.view, args.sparse, args.dim,  true)?;
        }
        OutputType::MagicaVoxel => {data.save_as_magica_voxel(&args.o, args.dim)?;}
    }

    Ok(())
}

pub fn get_extension(path : &str) -> Option<&str>{
    Some(std::path::Path::new(path).extension()?.to_str()?)
}

use clap::Parser;
use utils::Timer;
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    f : String,

    #[arg(short, long)]
    o : String,

    #[arg(short, long, default_value_t = 1022)]
    dim : u32,

    #[arg(short, long, action = clap::ArgAction::Set, default_value_t = true)]
    sparse : bool,

    #[arg(short, long, action = clap::ArgAction::Set, default_value_t = false)]
    timer : bool,
}

fn screenshot(input_path : &str, output_path : &str) -> std::process::Output{
    std::process::Command::new("gltf-viewer")
    .args([input_path, "-w", "1024", "-h", "1024", "-s", output_path]).output().unwrap()
}

fn gather(dir : &str) -> Vec<String>{
    std::fs::read_dir(dir).unwrap().map(|x: Result<std::fs::DirEntry, std::io::Error>|{
        let m = x.unwrap();
        m.file_name().to_str().unwrap().to_owned()
    }).collect()
}

fn generate_test_images(){
    let cwd = std::env::current_dir().unwrap();
    let cwd = cwd.as_os_str().to_str().unwrap();

    let output_dir = "data/voxelized";
    let input_dir = "data/original";
    let image_dir = "data/images";

    _ = std::fs::create_dir(output_dir);
    let files = gather(input_dir);

    for file in &files{
        let file_name = std::path::Path::new(file).file_stem().unwrap().to_str().unwrap();
        let file_path = format!("{input_dir}/{file}");
        let mut main_path : Option<String> = None;

        if std::fs::metadata(std::path::Path::new(file_path.as_str())).unwrap().is_dir(){
            for dir in std::fs::read_dir(file_path.as_str()).unwrap(){
                let name = dir.unwrap().file_name();
                let name = name.as_os_str().to_str().unwrap();
                let exten = get_extension(name);
                if exten.is_none(){continue;}
                if exten.unwrap() == "gltf"{
                    main_path = Some(format!("{input_dir}/{file}/{name}"));
                    break;
                }
            }
        }
        
        let f = if main_path.is_some(){main_path.unwrap()}else{file_path};
        let o = format!("{output_dir}/{file_name}/scene.gltf");
        let vox_name = format!("{image_dir}/{file_name}_vox.png");
        let oringal_name = format!("{image_dir}/{file_name}.png");

        let args = Args{f : f.clone(), o : o.clone(), dim : 10, sparse : true, timer : false};
        voxelize_mesh(&args).unwrap();  

        screenshot(f.as_str(), oringal_name.as_str());
        screenshot(o.as_str(), vox_name.as_str());
    }

}

fn main() {
    let time = Timer::new("total");

    let args = Args::parse();
    unsafe{crate::utils::PERF = args.timer};
    voxelize_mesh(&args).unwrap();
}
