use crate::gpu::{GpuBufferAllocation, PersistentBufferAllocation, VulkanSingleton};
use crate::utils::any_as_u8_slice;
use crate::{io::obj::MeshView, split::{DescriptorState, Uniforms}};
use crate::split::textures::Textures;
use ash::vk::BufferUsageFlags;

pub struct VulkanViews<T>{
    pub position : T,
    pub index : T,
    pub textcords : T,
    pub color : T,
}

impl<T : std::fmt::Debug> std::fmt::Debug for VulkanViews<T>{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VulkanViews{{position : {:?}, index : {:?}, textcords : {:?}, color : {:?}}}", self.position, self.index, self.textcords, self.color)
    }
}

impl<T : Copy> Clone for VulkanViews<T>{
    fn clone(&self) -> Self {
        *self
    }
}
impl<T : Copy> Copy for VulkanViews<T>{}

impl<T> VulkanViews<T>{

    pub fn cast<D>(self, func : impl Fn(T) -> D) -> VulkanViews<D>{
        VulkanViews::<D>{
            position : func(self.position),
            index : func(self.index),
            textcords : func(self.textcords),
            color : func(self.color),
        }
    }
}

impl<T : Copy> VulkanViews<T>{
    pub fn from_val(val : T) -> Self{
        Self { position: val, index: val, textcords: val, color : val }
    }
}

pub struct InputBuffers{
    pub data : DataInputBuffers,
    pub uniforms : GpuBufferAllocation,
    uniform_data : Uniforms,

    pub textures : Textures,

    pub output : GpuBufferAllocation,
    pub num_voxels : u32
}


impl InputBuffers{
    pub fn output_size(&self) -> u64{
        self.num_voxels  as u64 * 8 + 8
    }

    // set to zero
    pub fn resize_output(&mut self, vulkan : &VulkanSingleton, cmd : ash::vk::CommandBuffer, num_voxels : u32){
        let output_size = (num_voxels * 8 + 8) as u64;

        let flags = ash::vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER | ash::vk::BufferUsageFlags::TRANSFER_DST | ash::vk::BufferUsageFlags::TRANSFER_SRC;
        let mut output = GpuBufferAllocation::new_exclusive(flags, usage, output_size, &vulkan);

        self.num_voxels = num_voxels;
        self.uniform_data.voxel_capacity = num_voxels;
        std::mem::swap(&mut output, &mut self.output);
        output.drop(vulkan);


        unsafe{
            vulkan.device.begin_command_buffer(cmd, &Default::default()).unwrap();
            vulkan.device.cmd_update_buffer(cmd, self.output.buffer, 0, &[0; 8]);
            vulkan.device.cmd_update_buffer(cmd, self.uniforms.buffer, 0, any_as_u8_slice(&self.uniform_data));
            vulkan.device.end_command_buffer(cmd).unwrap();
        }
    }

    pub fn new(vulkan : &VulkanSingleton, views : &[MeshView], uniforms : Uniforms, num_voxels : u32, cmd : ash::vk::CommandBuffer) -> Self{
        unsafe{vulkan.device.begin_command_buffer(cmd, &Default::default())}.unwrap();
        let output_size = (num_voxels * 8 + 8) as u64;

        let flags = ash::vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER | ash::vk::BufferUsageFlags::TRANSFER_DST | ash::vk::BufferUsageFlags::TRANSFER_SRC;
        let output = GpuBufferAllocation::new_exclusive(flags, usage, output_size, &vulkan);

        let usage = ash::vk::BufferUsageFlags::UNIFORM_BUFFER | ash::vk::BufferUsageFlags::TRANSFER_DST;
        let uniform_buffer = GpuBufferAllocation::new_exclusive(flags, usage, std::mem::size_of::<Uniforms>() as u64, &vulkan);

        let data = DataInputBuffers::new(vulkan, views, cmd);
        let textures = Textures::new(vulkan, views, cmd);

        unsafe{
            vulkan.device.cmd_fill_buffer(cmd, output.buffer, 0, 8, 0);

            vulkan.device.cmd_update_buffer(cmd, uniform_buffer.buffer, 0, any_as_u8_slice(&uniforms));
            vulkan.device.end_command_buffer(cmd).unwrap()
        }

        Self { data, uniforms: uniform_buffer, uniform_data : uniforms, output, textures, num_voxels}
    }

    pub fn write(&self, vulkan : &VulkanSingleton, describ : &DescriptorState, sampler : ash::vk::Sampler){
        let writes = self.textures.inner.iter().enumerate().map(|(idx, image)|{
            let write = ash::vk::DescriptorImageInfo{
                sampler,
                image_view : image.view,
                image_layout : ash::vk::ImageLayout::GENERAL
            };
            
            (write, idx as usize)
        }).collect::<Vec<_>>();

        describ.write(
            &vulkan, 
            ash::vk::DescriptorBufferInfo{buffer : self.output.buffer, range : self.output_size(), offset : 0}, 
            ash::vk::DescriptorBufferInfo{buffer : self.uniforms.buffer, range : std::mem::size_of::<Uniforms>() as u64, offset : 0},
            &writes,
        );
    }

    pub fn drop(mut self, vulkan : &VulkanSingleton) {
        self.textures.drop(vulkan);
        self.data.drop(vulkan);
        self.uniforms.drop(vulkan);
        self.output.drop(vulkan);
    }
}

pub struct DataInputBuffers{
    cpu : PersistentBufferAllocation,

    pub position : GpuBufferAllocation,
    pub index : GpuBufferAllocation,
    pub textcord : Option<GpuBufferAllocation>,
    pub color : Option<GpuBufferAllocation>,

    pub views : Box<[VulkanViews<std::ops::Range<u64>>]>,
}

impl DataInputBuffers{
    pub fn drop(mut self, vulkan : &VulkanSingleton){
        self.cpu.drop(vulkan);
        self.position.drop(vulkan);
        self.index.drop(vulkan);

        if let Some(mut text_cord) = self.textcord{
            text_cord.drop(vulkan);
        }
        if let Some(mut color) = self.color{
            color.drop(vulkan);
        }
    }

    pub fn new(vulkan : &VulkanSingleton, views : &[MeshView], cmd : ash::vk::CommandBuffer) -> Self{
        use std::mem::size_of;

        let mut total_size: VulkanViews<usize> = VulkanViews::from_val(0);
        for view in views{
            total_size.position += view.vertices.len() * size_of::<[f32; 3]>();
            total_size.index += view.indices.len() * size_of::<[u32; 3]>();
            total_size.textcords += view.texture_cords.map(|x|{x.len()}).unwrap_or(0) * size_of::<[f32; 2]>();
            total_size.color += view.color.map(|x|{x.len()}).unwrap_or(0) * size_of::<[f32; 3]>();
        }
        let total_size = total_size.cast(|x|{x as u64});

        let (offsets, total_sizes) = {
            let position = 0;
            let index = position + total_size.position;
            let textcords = index + total_size.index;
            let color = textcords + total_size.textcords;
            let total = color + total_size.color;

            let grouped = VulkanViews{position, index, textcords, color};

            (grouped, total)
        };
    
        let usage = ash::vk::BufferUsageFlags::TRANSFER_SRC;
        let flags = ash::vk::MemoryPropertyFlags::HOST_VISIBLE | ash::vk::MemoryPropertyFlags::HOST_CACHED;
        let cpu = PersistentBufferAllocation::new_exclusive(flags, usage, total_sizes as u64, vulkan);

        let mut size = VulkanViews::from_val(0);
        let views= views.iter().map(|view|{
            let old = size;
            size.position += view.vertices.len() * size_of::<[f32; 3]>();
            size.index += view.indices.len() * size_of::<[u32; 3]>();
            size.textcords += view.texture_cords.map(|x|{x.len()}).unwrap_or(0) * size_of::<[f32; 2]>();
            size.color += view.color.map(|x|{x.len()}).unwrap_or(0) * size_of::<[f32; 3]>();
                    
            let vulkan_view = VulkanViews{
                position : old.position..size.position,
                index  : old.index..size.index,
                textcords : old.textcords..size.textcords,
                color : old.color..size.color
            };  
    
            unsafe{
                let dst = cpu.info.mapped_data.cast::<u8>();
    
                std::ptr::copy_nonoverlapping(view.vertices.as_ptr(), dst.add(offsets.position as usize + vulkan_view.position.start).cast::<[f32; 3]>(), view.vertices.len());
                std::ptr::copy_nonoverlapping(view.indices.as_ptr(), dst.add(offsets.index as usize + vulkan_view.index.start).cast::<[u32; 3]>(), view.indices.len());
    
                if let Some(text_cords) = view.texture_cords{
                    std::ptr::copy_nonoverlapping(text_cords.as_ptr(), dst.add(offsets.textcords as usize + vulkan_view.textcords.start).cast::<[f32; 2]>(), text_cords.len());
                }
                if let Some(color) = view.color{
                    std::ptr::copy_nonoverlapping(color.as_ptr(), dst.add(offsets.color as usize + vulkan_view.color.start).cast::<[f32; 3]>(), color.len());
                }
            }
    
            vulkan_view.cast(|x : std::ops::Range<usize>|{(x.start as u64)..(x.end as u64)})
        }).collect::<Box<_>>();
    

        let flags = ash::vk::MemoryPropertyFlags::DEVICE_LOCAL;
        
        let position = GpuBufferAllocation::new_exclusive(flags, BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER, total_size.position, vulkan);
        let index = GpuBufferAllocation::new_exclusive(flags, BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::INDEX_BUFFER, total_size.index, vulkan);

        let textcord = if total_size.textcords != 0 {
            Some(GpuBufferAllocation::new_exclusive(flags, BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER, total_size.textcords, vulkan))
        }else{
            None
        };
        let color = if total_size.color != 0 {
            Some(GpuBufferAllocation::new_exclusive(flags, BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::VERTEX_BUFFER, total_size.color, vulkan))
        }else{
            None
        };

        unsafe{
            vulkan.device.flush_mapped_memory_ranges(&[cpu.to_memory_range_aligned(&vulkan)]).unwrap();

            let position_region  = ash::vk::BufferCopy{dst_offset : 0, src_offset : offsets.position, size : total_size.position};
            vulkan.device.cmd_copy_buffer(cmd, cpu.buffer, position.buffer, &[position_region]);

            let index_region  = ash::vk::BufferCopy{dst_offset : 0, src_offset : offsets.index, size : total_size.index};
            vulkan.device.cmd_copy_buffer(cmd, cpu.buffer, index.buffer, &[index_region]);

            if let Some(textcord) = &textcord{
                let textcord_region  = ash::vk::BufferCopy{dst_offset : 0, src_offset : offsets.textcords, size : total_size.textcords};
                vulkan.device.cmd_copy_buffer(cmd, cpu.buffer, textcord.buffer, &[textcord_region]);
            }

            if let Some(color) = &color{
                let color_region  = ash::vk::BufferCopy{dst_offset : 0, src_offset : offsets.color, size : total_size.color};
                vulkan.device.cmd_copy_buffer(cmd, cpu.buffer, color.buffer, &[color_region]);
            }
        }

        Self { cpu, position, index, textcord, color, views }
    }
}

