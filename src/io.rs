use crate::utils::{slice_cast, vec_cast, Icords, Timer};
use crate::octree::octree_header;
use image::buffer::ConvertBuffer;

use crate::{octree::Octree, voxelizer::Fcords};

#[allow(dead_code)]
mod gltf2{

    //https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_bufferview_target 5.11.5. bufferView.target
    pub mod  buffer_view_target{
        pub const ARRAY_BUFFER : u32 = 34962;
        pub const ELEMENT_ARRAY_BUFFER : u32= 34963;
    }

    //https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_mesh_primitive_mode 5.24.4. mesh.primitive.mode
    pub mod primitive_mode{
        pub const POINTS : u32 = 0;
        pub const LINES : u32 = 1;
        pub const LINE_LOOP : u32 = 2;
        pub const LINE_STRIP : u32 = 3;
        pub const TRIANGLES : u32 = 4;
        pub const TRIANGLE_STRIP : u32 = 5;
        pub const TRIANGLE_FAN : u32 = 6;
    }
    
    pub trait IndexComponentType{
        const INDEX_COMPONENT_TYPE : i32;
    }

    //https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html 3.6.2.2. Accessor Data Types
    impl IndexComponentType for i8{const INDEX_COMPONENT_TYPE : i32 = 5120;}
    impl IndexComponentType for u8{const INDEX_COMPONENT_TYPE : i32 = 5121;}
    impl IndexComponentType for i16{const INDEX_COMPONENT_TYPE : i32 = 5122;}
    impl IndexComponentType for u16{const INDEX_COMPONENT_TYPE : i32 = 5123;}
    impl IndexComponentType for i32{const INDEX_COMPONENT_TYPE : i32 = 5125;}
    impl IndexComponentType for u32{const INDEX_COMPONENT_TYPE : i32 = 5126;}

    pub trait AccessorComponentType{
        const ACCESSOR_COMPONENT_TYPE : i32;
    }

    //https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html 5.1.3. accessor.componentType
    impl AccessorComponentType for i8{const ACCESSOR_COMPONENT_TYPE : i32 = 5120;}
    impl AccessorComponentType for u8{const ACCESSOR_COMPONENT_TYPE : i32 = 5121;}
    impl AccessorComponentType for i16{const ACCESSOR_COMPONENT_TYPE : i32 = 5122;}
    impl AccessorComponentType for u16{const ACCESSOR_COMPONENT_TYPE : i32 = 5123;}
    impl AccessorComponentType for i32{const ACCESSOR_COMPONENT_TYPE : i32 = 5125;}
    impl AccessorComponentType for f32{const ACCESSOR_COMPONENT_TYPE : i32 = 5126;}
}
use gltf2::{AccessorComponentType, buffer_view_target};


#[derive(Debug, Clone)]
pub struct Vertex{
    pub position : Fcords,
    pub color : image::Rgb<u8>,
}

#[derive(Debug, Clone)]
pub struct FloatVertex{
    pub position : Fcords,
    pub color : image::Rgb<f32>,
}

impl From<Vertex> for FloatVertex{
    fn from(value: Vertex) -> Self {
        let color = value.color.0.map(|c|{c as f32 / 255.0});
        Self{position : value.position, color : image::Rgb(color)}
    }
}

#[derive(Debug, Clone)]
pub struct Extra{
    pub normal : Option<nalgebra::Vector3<f32>>,
    pub uv : Option<nalgebra::Vector2<f32>>,

    pub material_idx : u32,
}

#[derive(Debug, Clone)]
pub enum ImageOrColor{
    Image(image::RgbImage),
    Color([u8; 3])
}

fn parse_material(mat : &gltf::Material, image_data: &Vec<gltf::image::Data>, source_dir : &str) -> Result<ImageOrColor, gltf::Error>{
    fn fallback(mat : &gltf::Material) -> ImageOrColor{
        let base_color = mat.emissive_factor();
        let base_color = [(base_color[0] * 255.0) as u8, (base_color[1] * 255.0) as u8, (base_color[2] * 255.0) as u8];
        ImageOrColor::Color(base_color)
    }

    let shadding = if let Some(metal) = mat.pbr_metallic_roughness().base_color_texture(){
        ImageOrColor::Image(parse_gltf_image(&image_data, metal.texture(), source_dir)?)
    }
    else if let Some(emissive) = mat.emissive_texture(){
        ImageOrColor::Image(parse_gltf_image(&image_data, emissive.texture(), source_dir)?)
    }
    else if let Some(spectral) = mat.pbr_specular_glossiness(){
        if let Some(image) = spectral.diffuse_texture(){
            ImageOrColor::Image(parse_gltf_image(&image_data, image.texture(), source_dir)?)
        }else{fallback(mat)}
    }else{fallback(mat)};

    Ok(shadding)
}

//todo add test for images
fn convert_gltf_image(data : &gltf::image::Data) -> image::RgbImage{
    pub trait BitWidth{fn to_u8(&self) -> u8;}
    impl BitWidth for u8{fn to_u8(&self) -> u8 {*self}}

    type Int16 = u16;
    impl BitWidth for Int16{fn to_u8(&self) -> u8 {(*self / 255) as u8}}
    impl BitWidth for f32{fn to_u8(&self) -> u8 {(self * 255.0) as u8}}

    fn r_convert<T : BitWidth>(data : &gltf::image::Data) -> image::RgbImage{
        let mut new_image : image::RgbImage = image::RgbImage::new(data.width, data.height);
        let num_pixel = data.width * data.height;
        let data_slice : &[T] = slice_cast(data.pixels.as_slice());

        for pixel in 0..num_pixel{
            let red = data_slice[pixel as usize].to_u8();
            *new_image.get_pixel_mut(pixel % data.width, pixel / data.width) = image::Rgb([red, 0, 0]);
        }

        new_image
    }

    fn rg_convert<T : BitWidth>(data : &gltf::image::Data) -> image::RgbImage{
        let mut new_image : image::RgbImage = image::RgbImage::new(data.width, data.height);
        let num_pixel = data.width * data.height;
        let data_slice : &[T] = slice_cast(data.pixels.as_slice());

        for pixel in 0..num_pixel{
            let red = data_slice[(pixel * 2) as usize].to_u8();
            let greeen = data_slice[(pixel * 2) as usize + 1].to_u8();
            *new_image.get_pixel_mut(pixel % data.width, pixel / data.width) = image::Rgb([red, greeen, 0]);
        }

        new_image
    }

    match data.format{
        gltf::image::Format::R8 => {r_convert::<u8>(data)}
        gltf::image::Format::R16 => {r_convert::<Int16>(data)}

        gltf::image::Format::R8G8 => {rg_convert::<u8>(data)}
        gltf::image::Format::R16G16 => {rg_convert::<Int16>(data)}


        gltf::image::Format::R32G32B32FLOAT => {
            let image_buffer: (Vec<f32>, u32, u32) = (vec_cast(data.pixels.clone()), data.width, data.height);
            let image : image::ImageBuffer<image::Rgb<f32>, Vec<f32>> = unsafe{std::mem::transmute(image_buffer)};
            image.convert()
        }

        gltf::image::Format::R16G16B16 => {
            let image_buffer: (Vec<Int16>, u32, u32) = (vec_cast(data.pixels.clone()), data.width, data.height);
            let image : image::ImageBuffer<image::Rgb<Int16>, Vec<Int16>> = unsafe{std::mem::transmute(image_buffer)};
            image.convert()
        }

        gltf::image::Format::R8G8B8 => {
            let image_buffer: (Vec<u8>, u32, u32) = (data.pixels.clone(), data.width, data.height);
            unsafe{std::mem::transmute(image_buffer)}
        }

        gltf::image::Format::R32G32B32A32FLOAT => {
            let image_buffer: (Vec<f32>, u32, u32) = (vec_cast(data.pixels.clone()), data.width, data.height);
            let image : image::ImageBuffer<image::Rgba<f32>, Vec<f32>> = unsafe{std::mem::transmute(image_buffer)};
            image.convert()
        }

        gltf::image::Format::R16G16B16A16 => {
            let image_buffer: (Vec<Int16>, u32, u32) = (vec_cast(data.pixels.clone()), data.width, data.height);
            let image : image::ImageBuffer<image::Rgba<Int16>, Vec<Int16>> = unsafe{std::mem::transmute(image_buffer)};
            image.convert()
        }

        gltf::image::Format::R8G8B8A8 => {
            let image_buffer: (Vec<u8>, u32, u32) = (vec_cast(data.pixels.clone()), data.width, data.height);
            let image : image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = unsafe{std::mem::transmute(image_buffer)};
            image.convert()
        }
    }
}

fn parse_gltf_image(image_data : &Vec<gltf::image::Data>, texture : gltf::Texture, source_dir : &str) -> Result<image::RgbImage, gltf::Error>{
    let source = texture.source().source();

    let data = match source {
        gltf::image::Source::Uri {uri, mime_type : _} =>{
            let path = format!("{source_dir}/{uri}");
            let image = image::open(path.as_str()).unwrap();
            image.into_rgb8()
        }

        gltf::image::Source::View { view: _, mime_type : _image_type } => {
            let index = texture.index();
            convert_gltf_image(&image_data[index])
        }
    };

    Ok(data)
}

#[derive(Debug, Clone)]
pub struct BoundingBox{
    pub min : Fcords,
    pub max : Fcords,
}

impl BoundingBox{
    pub const fn max() -> Self{
        Self{min : Fcords::new(f32::MAX, f32::MAX, f32::MAX), max : Fcords::new(f32::MIN, f32::MIN, f32::MIN)}
    }

    pub fn swap(&mut self, pos : Fcords){
        self.min = Fcords::new(self.min.x.min(pos.x), self.min.y.min(pos.y), self.min.z.min(pos.z));
        self.max = Fcords::new(self.max.x.max(pos.x), self.max.y.max(pos.y), self.max.z.max(pos.z));
    }
}

#[derive(Debug, Clone)]
pub struct Mesh{
    pub triangle : Vec<[nalgebra::Vector3<f32>; 3]>,
    pub extras : Vec<[Extra; 3]>,
    pub materials : Vec<ImageOrColor>,

    pub bounds : BoundingBox,
    pub view : View
}

#[derive(Debug)]
pub enum Error{
    Gltf(gltf::Error),
    UnsupportedFileType,
    NonTriangleGeometry,
    NoVertexPosition,
    NoVertexIndices,
}

impl From<gltf::Error> for Error{
    fn from(value: gltf::Error) -> Self {
        Self::Gltf(value)
    }
}

impl From<std::io::Error> for Error{
    fn from(value: std::io::Error) -> Self {
        Self::Gltf(gltf::Error::Io(value))
    }
}

#[derive(Debug, Clone)]
pub struct PerspectiveCamera{
    pub yfov: f32,
    pub znear: f32,

    pub zfar: Option<f32>,
    pub aspect_ratio: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct OrthographiCamera{
    pub xmag: f32,
    pub ymag: f32,
    pub zfar: f32,
    pub znear: f32,
}

impl PerspectiveCamera{
    pub fn new(value : &gltf::camera::Perspective<'_>) -> Self{
        Self{yfov : value.yfov(), znear : value.znear(), zfar : value.zfar(), aspect_ratio : value.aspect_ratio()}
    }

    pub fn to_json(&self) -> json::JsonValue{
        json::object!{
            "type" : "perspective",
            perspective : {
                yfov : self.yfov,
                znear : self.znear,
                zfar : self.zfar,
                aspect_ratio : self.aspect_ratio,
            }
        }
    }
}
impl OrthographiCamera{
    pub fn new(value : &gltf::camera::Orthographic<'_>) -> Self{
        Self{xmag : value.xmag(), ymag : value.ymag(), zfar : value.zfar(), znear : value.znear()}
    }

    pub fn to_json(&self) -> json::JsonValue{
        json::object!{
            "type": "orthographic",
            orthographic : {
                xmag : self.xmag,
                ymag : self.ymag,
                zfar : self.zfar,
                znear : self.znear,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Camera {
    PerspectiveCamera(PerspectiveCamera),
    OrthographiCamera(OrthographiCamera)
}
impl Camera{
    pub fn new(cam : &gltf::camera::Projection<'_>) -> Self{
        match cam {
            gltf::camera::Projection::Orthographic(ort) => {Camera::OrthographiCamera(OrthographiCamera::new(ort))}
            gltf::camera::Projection::Perspective(per) => {Camera::PerspectiveCamera(PerspectiveCamera::new(per))}
        }
    }
    pub fn to_json(&self) -> json::JsonValue{
        match self {
            Self::PerspectiveCamera(ort) => {ort.to_json()}
            Self::OrthographiCamera(per) => {per.to_json()}
        }
    }
}


#[derive(Debug, Clone)]
pub struct View{
    pub cam : Option<Camera>,
    pub mvp : [[f32; 4]; 4]
}

pub fn mpv_to_json(mvp : &[[f32; 4]; 4]) -> json::JsonValue{
    let mut output : Vec<json::JsonValue> = Vec::with_capacity(16);
    let mvp : &[f32; 16] = unsafe{std::mem::transmute(mvp)};
    for i in 0..16{
        output.push(mvp[i].into());
    }
    json::JsonValue::Array(output)
}

pub fn load_gltf(path : &str) -> Result<Mesh, Error>{
    let (document, buffers, images) = gltf::import(path)?;

    let folder = std::path::Path::new(path).parent().unwrap();
    let folder = folder.as_os_str().to_str().unwrap();

    let mut camera = None;
    for cam in document.cameras(){
        if cam.index() != 0{continue;}
        camera = Some(Camera::new(&cam.projection()));
        break;
    }
    let mvp = document.scenes().next().unwrap().nodes().next().unwrap().transform().matrix();
    let view = View{cam : camera, mvp};
    
    let mut triangle : Vec<nalgebra::Vector3<f32>> = Vec::new();
    let mut extras : Vec<Extra> = Vec::new();

    let material_info = document.materials();
    let mut materials : Vec<ImageOrColor> = Vec::new();
    for info in material_info{
        let mat = parse_material(&info, &images, folder)?;
        materials.push(mat)
    };

    //ie default material
    materials.push(ImageOrColor::Color([0, 0, 0]));

    let mut bounds: BoundingBox = BoundingBox::max();

    for mesh in document.meshes() {
        for primitive in mesh.primitives(){
            let mode = primitive.mode();
            if let gltf::mesh::Mode::Triangles = mode{}else{return Err(Error::NonTriangleGeometry);}

            let bound = primitive.bounding_box();
            bounds.swap(unsafe{core::mem::transmute(bound.min)});
            bounds.swap(unsafe{core::mem::transmute(bound.max)});
    
            if let gltf::mesh::Mode::Triangles = primitive.mode(){}else{todo!();}
            let idx = primitive.material().index();
            let idx = if idx.is_none(){materials.len()}else{idx.unwrap()};
    
            let data = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let indicies = data.read_indices();
            if indicies.is_none(){return Err(Error::NoVertexIndices);}
            let indicies : Vec<u32> = indicies.unwrap().into_u32().map(|x|{x}).collect();
            
            let vert_cords = data.read_positions();
            if vert_cords.is_none(){return Err(Error::NoVertexPosition);}
            let vert_cords : Vec<_> = vert_cords.unwrap().map(|x|{nalgebra::Vector3::new(x[0], x[1], x[2])}).collect();
    
            let normals = data.read_normals();
            let normals : Option<Vec<nalgebra::Vector3<f32>>> = if let Some(normals) = normals{
                Some(normals.map(|normal|{nalgebra::Vector3::new(normal[0], normal[1], normal[2])}).collect())
            }else{None};
    
            let uvs = data.read_tex_coords(0);
            let uvs : Option<Vec<nalgebra::Vector2<f32>>> = if let Some(uvs) = uvs{
                let uvs = uvs.into_f32();
                Some(uvs.map(|uv|{nalgebra::Vector2::new(uv[0], uv[1])}).collect())
            }else{None};
    
    
            for index in indicies{
                triangle.push(vert_cords[index as usize]);
                
                let normal = if let Some(ref normals) = normals{Some(normals[index as usize])}else{None};
                let uv = if let Some(ref uvs) = uvs{Some(uvs[index as usize])}else{None};
                let extra = Extra{material_idx : idx as u32, normal, uv};
    
                extras.push(extra)
            }
        }
    }

    Ok(Mesh{materials, triangle : vec_cast(triangle), extras : vec_cast(extras), bounds, view})
}

mod magica{
    //give r and g 3 bytes and b two bytes
    pub const fn encode(color : image::Rgb<u8>) -> u8{
        let color = color.0;
        (color[0] >> 5) | ((color[1] >> 5) << 3) | ((color[2] >> 6) << 6)
    }

    pub const fn decode(byte : u8) -> image::Rgb<u8>{
        let mask3 = (1 << 3) - 1;
        let mask2 = (1 << 2) - 1;

        let r = (byte & mask3) << 5;
        let g = ((byte >> 3) & mask3) << 5;
        let b = ((byte >> 6) & mask2) << 6;

        image::Rgb([r, g, b])
    }

    #[cfg(test)]
    pub const fn _gather(){
        let mut counter = 0;
        loop {
            if encode(decode(counter)) != counter{panic!()}
            if counter == u8::MAX{break;}
            counter += 1;
        }
    }

    #[cfg(test)]
    pub const _ : () = _gather();
}


impl Octree{
    //add model aware encodeing ie pallet should minize loss of colors used in input model not all possible rgb colors
    pub fn save_as_magica_voxel<S: AsRef<str> + ?Sized>(&self, file_path : &S, size : u32) -> std::io::Result<()> {
        let mut vox = vox_writer::VoxWriter::create(size as i32, size as i32, size as i32);
        

        let nodes = self.collect_nodes();
        for index in 0..u8::MAX{
            let color = magica::decode(index);

            vox.add_color(color.0[0], color.0[1], color.0[1], 0, index);
        }

        for (cord, color) in nodes{
            let color = octree_header::to_color(color);
            let color_idx = magica::encode(color);
            vox.add_voxel(cord.cords.x - 1, cord.cords.y - 1, cord.cords.z -1, color_idx as i32);
        }

        let string = file_path.as_ref().to_string();

        vox.save_to_file(string)
    }

    pub fn save_as_gltf<S: AsRef<str> + ?Sized>(&self, gltf_path : &S, view : View, sparse : bool, size : u32, float : bool) -> std::io::Result<()>{
        let max_size = size - 1;

        let mesh = if sparse{
            let _t = Timer::new("hollowing");
            self.fill_space(max_size)
        }else{
            let nodes = self.collect_nodes();
            let mut tris : Vec<Vertex> = Vec::with_capacity(nodes.len() * 36);
            for (node, color) in &nodes{
                let color = octree_header::to_color(*color);
                for i in 0..6{

                    let node = crate::space_filling::MeshNode{cords : node.cords, dim : i / 2, positive : (i % 2) == 0, depth : node.depth as u8};
                    let triangles = node.to_triangles(self.depth as u8);
                    let verts  : [Icords; 6]= unsafe{core::mem::transmute(triangles)};

                    let verts = verts.map(|vert|{
                        let position = (vert.add(-1).to_na().cast::<f64>() / max_size as f64).cast::<f32>();
                        let position = (position * 2.0).add_scalar(-1.0);
                        Vertex{position, color}
                    });

                    for vert in verts{
                        tris.push(vert);
                    }
                }
            }

            tris
        };

        let _t = Timer::new("saveing gltf");
        save_gltf(&mesh, gltf_path, view, float)
    }
}

fn get_bb(vertices : &[Vertex]) -> BoundingBox{
    let mut bounds = BoundingBox::max();
    for vert in vertices{
        let pos = vert.position;
        bounds.swap(pos);
    }
    bounds
}

fn save_gltf<S: AsRef<str> + ?Sized>(vertices : &[Vertex], gltf_path : &S, view : View, float : bool) -> std::io::Result<()> {
    let bb = get_bb(vertices);
    let num_bytes = vertices.len() * if float{core::mem::size_of::<FloatVertex>()}else{core::mem::size_of::<Vertex>()};

    let buffer = json::object!{
        uri : "model.bin",
        byteLength : num_bytes,
    };

    let byteStride = if float{core::mem::size_of::<FloatVertex>()}else{core::mem::size_of::<Vertex>()};
    let vertex_view = json::object!{
        buffer : 0,
        byteOffset : 0,
        byteLength : num_bytes,

        byteStride : byteStride,

        //target : gltf2::buffer_view_target::ELEMENT_ARRAY_BUFFER,
    };

    let byteOffset = if float{core::mem::offset_of!(FloatVertex, position)}else{core::mem::offset_of!(Vertex, position)};
    let position_accessor = json::object!{
        bufferView : 0,
        byteOffset : byteOffset,
        componentType : f32::ACCESSOR_COMPONENT_TYPE,
        count : vertices.len(),
        type : "VEC3",

        max : [bb.max.x, bb.max.y, bb.max.z],
        min : [bb.min.x, bb.min.y, bb.min.z],
    };

    let byteOffset = if float{core::mem::offset_of!(FloatVertex, color)}else{core::mem::offset_of!(Vertex, color)};
    let componentType = if float{f32::ACCESSOR_COMPONENT_TYPE}else{u8::ACCESSOR_COMPONENT_TYPE};
    let normalized = if float{false}else{true};
    
    let color_accessor = json::object!{
        bufferView : 0,
        byteOffset : byteOffset,
        componentType : componentType,
        normalized : normalized,
        count : vertices.len(),
        type : "VEC3",
    };

    let material = json::object! {
        doubleSided : true,
    };

    let mesh = json::object!{
        primitives : [{
            attributes : {
                POSITION : 0,
                COLOR_0 : 1,
            },

            material : 0
        }],
    };

    let gltf = json::object!{
        materials : [material],
        scenes : [ {nodes : [ 0 ]} ],
        nodes : [ {
            mesh : 0,
            matrix : mpv_to_json(&view.mvp),
        }],

        meshes : [mesh],
        buffers : [buffer],
        bufferViews : [vertex_view],
        accessors : [position_accessor, color_accessor],
        asset : {version : "2.0" }
    };

    let folder = std::path::Path::new(gltf_path.as_ref()).parent().unwrap();
    let folder = folder.as_os_str().to_str().unwrap();

    std::fs::create_dir(folder)?;
    let bin_path = format!("{}/model.bin", folder);

    std::fs::write(gltf_path.as_ref(), gltf.dump())?;
    if float{
        let vertices : Vec<_>= vertices.iter().map(|vert|{FloatVertex::from(vert.clone())}).collect();
        std::fs::write(bin_path, slice_cast(&vertices))?;
    }else{  
        std::fs::write(bin_path, slice_cast(vertices))?;
    }
    

    Ok(())
}
