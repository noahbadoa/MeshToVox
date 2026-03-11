mod octree;
#[allow(unused)]
mod utils;
#[allow(unused)]
mod arrayvec;
mod gpu;
mod split;
mod io;

mod space_filling;
mod shader_import;


use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    f : String,

    #[arg(short, long)]
    o : String,

    #[arg(short, long, default_value_t = 1024)]
    dim : u32,

    #[arg(short, long, action = clap::ArgAction::Set, default_value_t = true)]
    sparse : bool
}


fn from_args(args: &Args){
    use crate::io::obj::OwnedObjData;
    use crate::gpu::VulkanSingleton;
    use crate::io::obj::MeshView;
    use crate::split::{RazterState, razter_fragment};

    let imported = OwnedObjData::load(args.f.as_str());

    let data: Box<[MeshView<'_>]> = (0..imported.length()).into_iter().map(|index|{
        imported.view(index)
    }).collect::<Box<_>>();

    let vulkan = VulkanSingleton::new(cfg!(debug_assertions), &[ash::vk::PhysicalDeviceType::CPU, ash::vk::PhysicalDeviceType::INTEGRATED_GPU, ash::vk::PhysicalDeviceType::DISCRETE_GPU]);

    let state = RazterState::new(vulkan, args.dim, data.len() as u32);
    let fragment = razter_fragment(&data, &state);

    
    let save_path = std::path::Path::new(args.o.as_str());

    let kind = match save_path.extension().unwrap().to_str().unwrap() {
        "gltf" => {
            if args.sparse{
                crate::split::save::SaveType::GltfPruned
            }else{
                crate::split::save::SaveType::Gltf
            }
        },

        "vox" => crate::split::save::SaveType::MagicaVoxel,

        extension => panic!("unknow extension {extension}")
    };


    crate::split::save::save(fragment.as_slice(), save_path, kind, state.scale, state.depth);

    fragment.drop(&state.vulkan);
    state.drop();
}

fn main() {
    let args: Args = Args::parse();
    from_args(&args);
}


