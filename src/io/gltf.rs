use crate::utils::slice_cast;

pub type Fcords = nalgebra::Vector3<f32>;

#[allow(dead_code)]
mod gltf2{

    //https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_bufferview_target 5.11.5. bufferView.target
    pub mod  buffer_view_target{
        pub const ARRAY_BUFFER : u32 = 34962;
        pub const ELEMENT_ARRAY_BUFFER : u32 = 34963;
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
use gltf2::AccessorComponentType;


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
pub struct View{
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


fn get_bb(vertices : &[Vertex]) -> BoundingBox{
    let mut bounds = BoundingBox::max();
    for vert in vertices{
        let pos = vert.position;
        bounds.swap(pos);
    }
    bounds
}

#[allow(non_snake_case)]
pub fn save_gltf<S: AsRef<std::path::Path> + ?Sized>(vertices : &[Vertex], gltf_path : &S, view : Option<View>, float : bool) -> std::io::Result<()> {
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

    let node = if let Some(view) = view{
        json::object!{
            mesh : 0,
            matrix : mpv_to_json(&view.mvp),
        }
    }else{
        json::object!{
            mesh : 0
        }
    };

    let gltf = json::object!{
        materials : [material],
        scenes : [ {nodes : [ 0 ]} ],
        nodes : [ node ],

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
