use crate::{gpu::{Pipeline, VertexAtribInfo, VulkanSingleton}, io::obj::MeshView, split::{DescriptorState, Uniforms}, utils::any_as_u8_slice};
use crate::gpu::PersistentBufferAllocation;
use crate::split::data::InputBuffers;

pub struct PipelineState{
    pub textured_pipeline : Pipeline,
    pub untextured_pipeline : Pipeline,
    pub colored_pipeline : Pipeline,
}

impl PipelineState{
    pub fn new(vulkan : &VulkanSingleton, set_layouts: &[ash::vk::DescriptorSetLayout], render_pass: ash::vk::RenderPass, dim : u32) -> Self{
        let cache = ash::vk::PipelineCache::null();

        let position = VertexAtribInfo{format : ash::vk::Format::R32G32B32_SFLOAT, stride : 12};
        let texture_cordinate = VertexAtribInfo{format : ash::vk::Format::R32G32_SFLOAT, stride : 8};
        let color = VertexAtribInfo{format : ash::vk::Format::R32G32B32_SFLOAT, stride : 12};

        let topology = ash::vk::PrimitiveTopology::TRIANGLE_LIST;

        let textured_info = crate::gpu::PipelineState{topology, position, texture_cordinate : Some(texture_cordinate), color : None};
        let untextured_info = crate::gpu::PipelineState{topology, position, texture_cordinate : None, color : None};
        let colored_info = crate::gpu::PipelineState{topology, position, texture_cordinate : None, color : Some(color)};

        let textured_pipeline = Pipeline::new(vulkan, render_pass, cache, set_layouts, textured_info, dim).unwrap();
        let colored_pipeline = Pipeline::new(vulkan, render_pass, cache, set_layouts, colored_info, dim).unwrap();
        let untextured_pipeline = Pipeline::new(vulkan, render_pass, cache, set_layouts, untextured_info, dim).unwrap();


        Self { textured_pipeline, untextured_pipeline, colored_pipeline }
    }

    pub fn drop(&mut self, vulkan : &VulkanSingleton){
        self.textured_pipeline.drop(vulkan);
        self.untextured_pipeline.drop(vulkan);
        self.colored_pipeline.drop(vulkan);
    }
}

#[allow(unused)]
pub struct RazterState{
    pub vulkan: VulkanSingleton,
    describ: DescriptorState,

    pool: ash::vk::CommandPool,
    framebuffer : Framebuffer,
    sampler : ash::vk::Sampler,

    pipeline : PipelineState,

    max_draw_count : u32,
    pub scale : u32,
    pub depth : u32,
}

pub struct Framebuffer{
    pub render_pass: ash::vk::RenderPass,
    pub framebuffer: ash::vk::Framebuffer,
}

impl Framebuffer{
    pub fn new(vulkan: &VulkanSingleton, dim : u32) -> Self{
        let render_pass: ash::vk::RenderPass = Pipeline::empty_render_pass(&vulkan).unwrap();
        let framebuffer = ash::vk::FramebufferCreateInfo{render_pass,  width : dim, height : dim, layers : 1, ..Default::default()};
        let framebuffer: ash::vk::Framebuffer = unsafe{vulkan.device.create_framebuffer(&framebuffer, None).unwrap()};

        Self { render_pass, framebuffer }
    }

    pub fn drop(self, vulkan: &VulkanSingleton){
        unsafe{
            vulkan.device.destroy_render_pass(self.render_pass, None);
            vulkan.device.destroy_framebuffer(self.framebuffer, None);
        }
    }
}

impl RazterState{
    pub fn new(vulkan: VulkanSingleton, scale: u32, max_draw_count : u32) -> Self{
        let sampler = unsafe{
            let info = ash::vk::SamplerCreateInfo{
                mag_filter : ash::vk::Filter::NEAREST, 
                min_filter : ash::vk::Filter::NEAREST, 
                address_mode_u : ash::vk::SamplerAddressMode::REPEAT, address_mode_v : ash::vk::SamplerAddressMode::REPEAT, address_mode_w : ash::vk::SamplerAddressMode::REPEAT,
                mipmap_mode : ash::vk::SamplerMipmapMode::NEAREST,
                compare_enable : ash::vk::FALSE,
                min_lod : 0.0,
                max_lod : ash::vk::LOD_CLAMP_NONE,
                mip_lod_bias : 0.0,
        
                anisotropy_enable : ash::vk::FALSE,
                max_anisotropy : 0.0,
        
                unnormalized_coordinates : ash::vk::FALSE,
        
                ..Default::default()
            };
            vulkan.device.create_sampler(&info, None).unwrap()
        };

        let describ = DescriptorState::new(&vulkan, max_draw_count);
        let command_pool_info = ash::vk::CommandPoolCreateInfo{flags : ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER, ..Default::default()};
        let pool = unsafe{vulkan.device.create_command_pool(&command_pool_info, None)}.unwrap();
        let depth = (scale as f32).log2().ceil() as u32;
        let framebuffer = Framebuffer::new(&vulkan, scale);
        // let framebuffer = Framebuffer::new_colored(&vulkan, dim);
        let pipeline = PipelineState::new(&vulkan, &[describ.set.layout], framebuffer.render_pass, scale);


        Self { vulkan, describ, pool, framebuffer, pipeline, scale, depth, max_draw_count, sampler}
    }

    pub fn drop(mut self) {
        unsafe{
            self.vulkan.device.destroy_command_pool(self.pool, None);
            self.vulkan.device.destroy_sampler(self.sampler, None);
        }

        self.framebuffer.drop(&self.vulkan);
        self.describ.drop(&self.vulkan);
        self.pipeline.drop(&self.vulkan);
        self.vulkan.drop();
    }
}


pub struct Fragments{
    pub alloc : PersistentBufferAllocation,
    pub length : u64
}

impl Fragments{
    pub fn new(mut alloc : PersistentBufferAllocation, vulkan : &VulkanSingleton, max_voxels : u64) -> Result<Self, u64>{
        let length = unsafe{alloc.info.mapped_data.cast::<u64>().read()};
        if length > max_voxels as u64{
            alloc.drop(vulkan);
            return Err(length);
        }
            
        Ok(Self { alloc, length })
    }

    pub fn as_slice<'a>(&'a self) -> &'a [u64]{
        unsafe{std::slice::from_raw_parts(self.alloc.info.mapped_data.cast::<u64>().add(1), self.length as usize)}
    }

    pub fn drop(mut self, vulkan : &VulkanSingleton){
        self.alloc.drop(vulkan);
    }
}

pub fn razter_fragment(views : &[MeshView], state : &RazterState) -> Fragments{
    let vulkan = &state.vulkan;
    let [cmd, transfer_cmd] = state.vulkan.create_command_buffers(state.pool).unwrap();

    let num_voxels: u32 = 2u32.pow(25);

    let uniforms = Uniforms::new(views, state.depth, state.scale, num_voxels);
    let mut data: InputBuffers = InputBuffers::new(vulkan, views, uniforms, num_voxels, transfer_cmd);

    data.write(vulkan, &state.describ, state.sampler);

    loop {
    unsafe{
        let begin = ash::vk::RenderPassBeginInfo{
            render_pass : state.framebuffer.render_pass, 
            render_area : ash::vk::Rect2D{offset : ash::vk::Offset2D{x : 0, y : 0}, extent : ash::vk::Extent2D{width : state.scale, height : state.scale}},
            clear_value_count : 0,
            framebuffer : state.framebuffer.framebuffer,
            ..Default::default()
        };
        let contents = ash::vk::SubpassContents::INLINE;
        
        vulkan.device.begin_command_buffer(cmd, &Default::default()).unwrap();
        vulkan.device.cmd_begin_render_pass(cmd, &begin, contents);

        for (i, view) in data.data.views.iter().enumerate(){
            let instance_id = data.textures.texture_index[i];
            
            if (view.textcords.start != view.textcords.end) && instance_id.is_some(){
                vulkan.device.cmd_bind_pipeline(cmd, ash::vk::PipelineBindPoint::GRAPHICS, state.pipeline.textured_pipeline.pipeline);
                vulkan.device.cmd_bind_descriptor_sets(cmd, ash::vk::PipelineBindPoint::GRAPHICS, state.pipeline.textured_pipeline.layout, 0, &[state.describ.set.main], &[]);

                vulkan.device.cmd_bind_vertex_buffers(cmd, 1, &[data.data.textcord.as_ref().unwrap().buffer], &[view.textcords.start]);
            }else if view.color.start != view.color.end{
                vulkan.device.cmd_bind_pipeline(cmd, ash::vk::PipelineBindPoint::GRAPHICS, state.pipeline.colored_pipeline.pipeline);
                vulkan.device.cmd_bind_descriptor_sets(cmd, ash::vk::PipelineBindPoint::GRAPHICS, state.pipeline.colored_pipeline.layout, 0, &[state.describ.set.main], &[]);

                vulkan.device.cmd_bind_vertex_buffers(cmd, 1, &[data.data.color.as_ref().unwrap().buffer], &[view.color.start]);
            }else{
                vulkan.device.cmd_bind_pipeline(cmd, ash::vk::PipelineBindPoint::GRAPHICS, state.pipeline.untextured_pipeline.pipeline);
                vulkan.device.cmd_bind_descriptor_sets(cmd, ash::vk::PipelineBindPoint::GRAPHICS, state.pipeline.untextured_pipeline.layout, 0, &[state.describ.set.main], &[]);
                let color = match views[i].materail{
                    crate::io::obj::MaterailView::Diffuse { color } => *color,
                    crate::io::obj::MaterailView::Empty => [0.5, 0.5, 0.5],
                    _ => unreachable!()
                };
                vulkan.device.cmd_push_constants(cmd, state.pipeline.untextured_pipeline.layout, ash::vk::ShaderStageFlags::FRAGMENT, 0, any_as_u8_slice(&color));
            }

            vulkan.device.cmd_bind_vertex_buffers(cmd, 0, &[data.data.position.buffer], &[view.position.start]);


            let length = (view.index.end - view.index.start) / std::mem::size_of::<u32>() as u64;
            vulkan.device.cmd_bind_index_buffer(cmd, data.data.index.buffer, view.index.start, ash::vk::IndexType::UINT32);
            vulkan.device.cmd_draw_indexed(cmd, length as u32, 1, 0, 0, instance_id.unwrap_or(0));
        }

        vulkan.device.cmd_end_render_pass(cmd);
        vulkan.device.end_command_buffer(cmd).unwrap();
    }

    unsafe{
        let [semaphore] = vulkan.create_semaphores([false]);
        let fences= vulkan.create_fences([false]);
        let cmds = [cmd];
        let transfer_cmds = [transfer_cmd];

        let sumbit_1 = ash::vk::SubmitInfo{
            signal_semaphore_count : 1,
            p_signal_semaphores : &semaphore,
            
            ..Default::default()
        }.command_buffers(&transfer_cmds);

        let sumbit_2 = ash::vk::SubmitInfo{
            wait_semaphore_count : 1,
            p_wait_dst_stage_mask : &ash::vk::PipelineStageFlags::ALL_GRAPHICS,
            p_wait_semaphores : &semaphore,

            ..Default::default()
        }.command_buffers(&cmds);

        vulkan.device.queue_submit(vulkan.queue, &[sumbit_1, sumbit_2], fences[0]).unwrap();
        vulkan.device.wait_for_fences(&fences, true, 1_000_000_000).unwrap();

        vulkan.device.destroy_semaphore(semaphore, None);
        vulkan.device.destroy_fence(fences[0], None);
    }


    let frag  = unsafe{
        let flags = ash::vk::MemoryPropertyFlags::HOST_VISIBLE | ash::vk::MemoryPropertyFlags::HOST_CACHED;
        let usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER | ash::vk::BufferUsageFlags::TRANSFER_DST;
        let host_voxel_fragment_buffer = PersistentBufferAllocation::new_exclusive(flags, usage, data.output_size(), &vulkan);

        vulkan.device.begin_command_buffer(cmd, &Default::default()).unwrap();
        let region = ash::vk::BufferCopy{src_offset : 0, dst_offset : 0, size : data.output_size()};
        vulkan.device.cmd_copy_buffer(cmd, data.output.buffer, host_voxel_fragment_buffer.buffer, &[region]);
        vulkan.device.end_command_buffer(cmd).unwrap();

        let fences = vulkan.create_fences([false]);

        let cmds = [cmd];
        let sumbit = ash::vk::SubmitInfo::default().command_buffers(&cmds);
        vulkan.device.queue_submit(vulkan.queue, &[sumbit], fences[0]).unwrap();
        vulkan.device.wait_for_fences(&fences, true, 1_000_000_000).unwrap();
        vulkan.device.destroy_fence(fences[0], None);

        let out = Fragments::new(host_voxel_fragment_buffer, vulkan,data.num_voxels as u64);
        match out {
            Ok(out) => out,
            Err(num_voxels) => {
                data.resize_output(vulkan, transfer_cmd, num_voxels as u32);
                state.describ.update_output_buffer(vulkan, data.output.to_descriptor());
                continue;
            }
        }
    };

    data.drop(vulkan);
    return frag;
    }
}
