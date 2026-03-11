use crate::{arrayvec::ArrayVec, gpu::VulkanSingleton};

fn new_shader_module(code : &[u32], vulkan : &VulkanSingleton) -> Result<ash::vk::ShaderModule, ash::vk::Result>{
    let shader_info = ash::vk::ShaderModuleCreateInfo::default().code(code);
    unsafe{vulkan.device.create_shader_module(&shader_info, None)}
}

use ash::vk::{ShaderStageFlags, PipelineShaderStageCreateInfo, PipelineViewportStateCreateInfo};

pub struct Pipeline{
    pub pipeline : ash::vk::Pipeline,
    pub layout : ash::vk::PipelineLayout,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct VertexAtribInfo{
    pub stride : u32,
    pub format : ash::vk::Format,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PipelineState{
    pub topology : ash::vk::PrimitiveTopology,
    pub position : VertexAtribInfo,
    pub texture_cordinate : Option<VertexAtribInfo>,
    pub color : Option<VertexAtribInfo>,
}

use ash::vk::{VertexInputBindingDescription, VertexInputAttributeDescription};
impl PipelineState{
    pub fn to_vertex_info(&self) -> (ArrayVec<VertexInputBindingDescription, 2>, ArrayVec<VertexInputAttributeDescription, 2>){
        let mut binding: ArrayVec<VertexInputBindingDescription, 2> = ArrayVec::new();
        let mut atribute: ArrayVec<VertexInputAttributeDescription, 2>  = ArrayVec::new();

        let mut add_atrib = |atrib : &VertexAtribInfo, binding_loc : u32| {
            binding.push(ash::vk::VertexInputBindingDescription{binding : binding_loc, stride : atrib.stride, input_rate : ash::vk::VertexInputRate::VERTEX});
            atribute.push(ash::vk::VertexInputAttributeDescription{binding : binding_loc, location : binding_loc, offset : 0, format : atrib.format});
        };

        add_atrib(&self.position, 0);
        if let Some(texture_cordinate) = &self.texture_cordinate {add_atrib(texture_cordinate, 1)}
        else if let Some(color) = &self.color {add_atrib(color, 1)};

        (binding, atribute)
    }
}

impl Pipeline{
    pub fn drop(&mut self, vulkan : &VulkanSingleton){
        unsafe{
            vulkan.device.destroy_pipeline(self.pipeline, None);
            vulkan.device.destroy_pipeline_layout(self.layout, None);
        }
    }

    fn init_pipeline_layout(vulkan : &VulkanSingleton, set_layouts : &[ash::vk::DescriptorSetLayout], push_constants : bool) -> ash::vk::PipelineLayout{
        let mut layout_info = ash::vk::PipelineLayoutCreateInfo{
            set_layout_count : set_layouts.len() as u32,
            p_set_layouts : set_layouts.as_ptr(),

            ..Default::default()
        };
        if push_constants{
            layout_info = layout_info.push_constant_ranges(&[ash::vk::PushConstantRange{stage_flags : ash::vk::ShaderStageFlags::FRAGMENT, offset : 0, size : 12}]);
        }

        unsafe{vulkan.device.create_pipeline_layout(&layout_info, None)}.expect("layout")
    }

    fn init_pipeline(vulkan : &VulkanSingleton, cache : ash::vk::PipelineCache, render_pass : ash::vk::RenderPass, shader : ash::vk::ShaderModule, layout: ash::vk::PipelineLayout, info : PipelineState, dim : u32) -> ash::vk::Pipeline{
        use std::ptr::addr_of;

        let vertex = PipelineShaderStageCreateInfo{module: shader, p_name: c"vertex".as_ptr(), stage: ShaderStageFlags::VERTEX, ..Default::default()};
        let geometry = PipelineShaderStageCreateInfo{module: shader, p_name: c"geometry".as_ptr(), stage: ShaderStageFlags::GEOMETRY, ..Default::default()};
        let fragment = PipelineShaderStageCreateInfo{module: shader, p_name: c"fragment".as_ptr(), stage: ShaderStageFlags::FRAGMENT, ..Default::default()};
        let stages = &[vertex, geometry, fragment];

        let (vertex_describ, vertex_info) = info.to_vertex_info(); 
        let vertex_data = ash::vk::PipelineVertexInputStateCreateInfo::default().vertex_attribute_descriptions(vertex_info.as_slice()).vertex_binding_descriptions(vertex_describ.as_slice());
        
        let assembly_data = ash::vk::PipelineInputAssemblyStateCreateInfo{
            topology : info.topology,
            primitive_restart_enable : ash::vk::FALSE,
            ..Default::default()
        };
        let tesselation_data = ash::vk::PipelineTessellationStateCreateInfo::default();

        let view = [ash::vk::Viewport{x : 0.0, y : 0.0, width : dim as f32, height : dim as f32, min_depth : 0.0, max_depth : 1.0}];
        let scissors = [ash::vk::Rect2D{offset : ash::vk::Offset2D{x : 0, y : 0}, extent : ash::vk::Extent2D{width : dim, height : dim}}];
        let viewport = PipelineViewportStateCreateInfo::default().viewports(&view).scissors(&scissors);


        let extra = ash::vk::PipelineRasterizationConservativeStateCreateInfoEXT{
            extra_primitive_overestimation_size : 0.5f32.sqrt(),
            conservative_rasterization_mode : ash::vk::ConservativeRasterizationModeEXT::OVERESTIMATE,
            flags : ash::vk::PipelineRasterizationConservativeStateCreateFlagsEXT::empty(),
            ..Default::default()
        };

        let raster_state = ash::vk::PipelineRasterizationStateCreateInfo{
            depth_clamp_enable : ash::vk::FALSE,
            rasterizer_discard_enable : ash::vk::FALSE,
            depth_bias_enable : ash::vk::FALSE,

            polygon_mode : ash::vk::PolygonMode::FILL,
            cull_mode : ash::vk::CullModeFlags::NONE,

            p_next : addr_of!(extra).cast(),
            line_width : 1.0,

            ..Default::default()
        };

        let multi_sample = ash::vk::PipelineMultisampleStateCreateInfo{
            rasterization_samples : ash::vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let blend = [ash::vk::PipelineColorBlendAttachmentState{
            blend_enable : ash::vk::FALSE,
            ..Default::default()
        }];
        let color = ash::vk::PipelineColorBlendStateCreateInfo{
            logic_op_enable : ash::vk::FALSE,
            
            ..Default::default()
        }.attachments(&blend);

        let graphics_pipeline = ash::vk::GraphicsPipelineCreateInfo{
            stage_count : stages.len() as u32,
            p_stages : stages.as_ptr(),

            p_vertex_input_state : addr_of!(vertex_data),
            p_input_assembly_state : addr_of!(assembly_data),
            p_tessellation_state : addr_of!(tesselation_data),

            p_viewport_state : addr_of!(viewport),
            p_rasterization_state : addr_of!(raster_state),
            p_multisample_state : addr_of!(multi_sample),
            p_color_blend_state : addr_of!(color),

            layout,
            subpass : 0,
            base_pipeline_handle : ash::vk::Pipeline::null(),

            render_pass,

            ..Default::default()
        };

        unsafe {vulkan.device.create_graphics_pipelines(cache, &[graphics_pipeline], None)}.expect("wokred")[0]
    }

    pub fn empty_render_pass(vulkan : &VulkanSingleton) -> Result<ash::vk::RenderPass, ash::vk::Result>{
        unsafe{
            let subpass = ash::vk::SubpassDescription{
                flags : ash::vk::SubpassDescriptionFlags::empty(),
                pipeline_bind_point : ash::vk::PipelineBindPoint::GRAPHICS,
                ..Default::default()
            };
    
            let subpasses = [subpass];
            let info = ash::vk::RenderPassCreateInfo{
                ..Default::default()
            }.subpasses(&subpasses);
    
            vulkan.device.create_render_pass(&info, None)
        }
    }

    pub fn new(vulkan: &VulkanSingleton, render_pass : ash::vk::RenderPass, cache : ash::vk::PipelineCache, set_layouts : &[ash::vk::DescriptorSetLayout], info : PipelineState, dim : u32) -> Result<Self, ash::vk::Result>{
        use crate::shader_import::shader_options;
        let target = if cfg!(debug_assertions) {shader_options::Target::Debug} else {shader_options::Target::Release};
        let vertex = if info.texture_cordinate.is_some(){
            shader_options::VertexDataVariants::UvBasedColorInfo
        }else if info.color.is_some(){
            shader_options::VertexDataVariants::PerVertexColorInfo
        }else{
            shader_options::VertexDataVariants::NoColorInfo
        };

        let pipeline_layout: ash::vk::PipelineLayout = Self::init_pipeline_layout(&vulkan, &set_layouts, vertex == shader_options::VertexDataVariants::NoColorInfo);


        let options = shader_options::ShaderOptions{primitives : shader_options::PrimitiveType::TriangleList, vertex, target};

        let code = crate::shader_import::ALL_SHADERS.get_shader_source(options.to_file_id() as u64).unwrap();
        let code = unsafe{std::slice::from_raw_parts(code.as_ptr().cast::<u32>(), code.len() / 4)};
        let shader_module = new_shader_module(code, &vulkan)?;

        let pipeline: ash::vk::Pipeline = Self::init_pipeline(&vulkan, cache, render_pass, shader_module, pipeline_layout, info, dim);
        unsafe {vulkan.device.destroy_shader_module(shader_module, None)};

        Ok(Self {pipeline, layout : pipeline_layout})
    }
}

