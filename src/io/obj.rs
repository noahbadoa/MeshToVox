use crate::split::{BoundingBox, Bounds};
use crate::utils::slice_cast;

pub struct OwnedObjData{
    models : Box<[tobj::Mesh]>,
    mats : Vec<MaterailView>,
}

impl OwnedObjData{
    pub fn load<P : AsRef<std::path::Path> + std::fmt::Debug>(path : P) -> Self{
        let opition = tobj::LoadOptions{
            ignore_points : true,
            ignore_lines : true,
            single_index : true,
            triangulate : true
        };


        let (models, materials) = tobj::load_obj(path.as_ref(), &opition).expect("Failed to OBJ load file");

        let materials: Vec<tobj::Material> = materials.unwrap();
        let parent = path.as_ref().parent().unwrap().to_path_buf();
    
        let mut mats: Vec<MaterailView> = materials.into_iter().map(|x|{
            if let Some(diffuse_texture) = x.diffuse_texture{
                let mut parent = parent.clone();
                parent.push(diffuse_texture);
                
                let mut image = image::open(parent).unwrap().to_rgba8();
                image::imageops::flip_vertical_in_place(&mut image);
    
                MaterailView::Textured { image }
            }else if let Some(diffuse) = x.diffuse{
                MaterailView::Diffuse { color: diffuse }
            }else{
                MaterailView::Empty
            }
        }).collect::<Vec<_>>();
        mats.push(MaterailView::Empty);
        let models: Box<[tobj::Mesh]> = models.into_iter().map(|x|{
            let out = x.mesh;
            
            out
        }).collect::<Box<_>>();


        Self { models, mats }
    }

    pub fn length(&self) -> usize{
        self.models.len()
    } 

    pub fn view<'a>(&'a self, index : usize) -> MeshView<'a>{
        let model = &self.models[index];

        let texture_cords = if model.texcoords.len() != 0{
            Some(slice_cast(model.texcoords.as_slice()))
        }else{
            None
        };

        let color = if model.vertex_color.len() != 0{
            Some(slice_cast(model.vertex_color.as_slice()))
        }else{
            None
        };

        let material_id = model.material_id.unwrap_or(self.mats.len() - 1);
        
        MeshView{
            vertices : slice_cast(model.positions.as_slice()),
            indices : slice_cast(model.indices.as_slice()),
            texture_cords,
            color,
            materail : &self.mats[material_id],
        }
    }
}



#[derive(Debug)]
pub enum MaterailView{
    Textured{image : image::RgbaImage},
    Diffuse{color : [f32; 3]},
    Empty
}

#[derive(Debug)]
pub struct MeshView<'a>{
    pub vertices : &'a [[f32; 3]],
    pub indices :  &'a [[u32; 3]],
    pub texture_cords : Option<&'a [[f32; 2]]>,
    pub color : Option<&'a [[f32; 3]]>,
    pub materail : &'a MaterailView,
}

impl<'a> Bounds for MeshView<'a>{
    fn bounds(&self) -> BoundingBox {
        let mut out = BoundingBox{max : [f32::MIN; 3], min : [f32::MAX; 3]};

        for vertex in self.vertices{
            for i in 0..3{
                out.max[i] = out.max[i].max(vertex[i]);
                out.min[i] = out.min[i].min(vertex[i]);
            }
        }

        out
    }
}
