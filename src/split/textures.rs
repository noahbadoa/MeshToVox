use ash::vk::ComponentSwizzle;
use vk_mem::Alloc;
use std::collections::HashMap;

use crate::gpu::{PersistentBufferAllocation, VulkanSingleton};
use crate::io::obj::{MaterailView, MeshView};

pub struct ImageAllocPair{
    pub image : ash::vk::Image,
    pub alloc : vk_mem::Allocation,
    pub view: ash::vk::ImageView,
}

impl ImageAllocPair{
    pub fn drop(mut self, vulkan : &VulkanSingleton){
        unsafe{
            vulkan.device.destroy_image_view(self.view, None);
            vulkan.device.destroy_image(self.image, None);
            vulkan.allocator.free_memory(&mut self.alloc);
        }
    }
}


pub struct Textures{
    pub inner : Box<[ImageAllocPair]>,
    pub texture_index : Vec<Option<u32>>,
    host_buffer: Option<PersistentBufferAllocation>,
}

impl Textures{
    pub fn drop(self, vulkan : &VulkanSingleton){
        if let Some(mut host) = self.host_buffer{
            host.drop(vulkan);
        }
        for alloc in self.inner{
            alloc.drop(vulkan);
        }
    }

    pub fn new(vulkan : &VulkanSingleton, views : &[MeshView], cmd : ash::vk::CommandBuffer) -> Self{
        let mut unique: HashMap<usize, (&image::ImageBuffer<image::Rgba<u8>, Vec<u8>>, u32)> = HashMap::new();
        let mut texture_index = Vec::with_capacity(views.len());

        for view in views{
            if let MaterailView::Textured { image } = view.materail{
                let src = std::ptr::from_ref(image).addr();

                let length = unique.len() as u32;
                let idx = match unique.entry(src){
                    std::collections::hash_map::Entry::Occupied(occupied) => {
                        occupied.get().1
                    }

                    std::collections::hash_map::Entry::Vacant(vacant) => {
                        vacant.insert((image, length));
                        length
                    }
                };
                
                texture_index.push(Some(idx));
            }else{
                texture_index.push(None);
            }
        }

        let mut src_size = 0;
        for (unique_mat_view_index, _) in unique.values(){
            src_size += unique_mat_view_index.dimensions().0 * unique_mat_view_index.dimensions().1 * 4;
        }

        let mut out: Vec<Option<ImageAllocPair>> = (0..unique.len()).into_iter().map(|_|{None}).collect();
        if src_size == 0 {return Self { inner: Box::new([]), texture_index, host_buffer : None};}


        let usage = ash::vk::BufferUsageFlags::TRANSFER_SRC;
        let flags = ash::vk::MemoryPropertyFlags::HOST_VISIBLE | ash::vk::MemoryPropertyFlags::HOST_CACHED;
        let host_buffer: PersistentBufferAllocation = PersistentBufferAllocation::new_exclusive(flags, usage, src_size as u64, vulkan);
        
        let mut src_offset = 0;
        for (_, (unique_mat_view_index, mat_index)) in unique{
            let (width, height) = unique_mat_view_index.dimensions();
            let size = width * height * 4;
            let data = Self::record_image_copy(vulkan, &host_buffer, src_offset, unique_mat_view_index, cmd);

            out[mat_index as usize] = Some(data);
            src_offset += size as u64;
        }

        unsafe{vulkan.device.flush_mapped_memory_ranges(&[host_buffer.to_memory_range_aligned(&vulkan)])}.unwrap();
        let inner = out.into_iter().map(|x|{x.unwrap()}).collect::<Box<_>>();

        Self { inner, texture_index, host_buffer : Some(host_buffer) }
    }

    fn record_image_copy(vulkan : &VulkanSingleton, host_buffer : &PersistentBufferAllocation, host_buffer_offset : u64, host : &image::RgbaImage, cmd : ash::vk::CommandBuffer) -> ImageAllocPair{
        let (width, height) = host.dimensions();
        let data = ImageAllocPair::alloc(width, height, ash::vk::Format::R8G8B8A8_UINT, ash::vk::ImageUsageFlags::SAMPLED | ash::vk::ImageUsageFlags::TRANSFER_DST, ash::vk::MemoryPropertyFlags::DEVICE_LOCAL, vulkan);
        let size = width * height * 4;

        unsafe{
            std::ptr::copy_nonoverlapping(host.as_ptr(), host_buffer.info.mapped_data.cast::<u8>().add(host_buffer_offset as usize), size as usize);
            let region = ash::vk::BufferImageCopy{
                buffer_offset : host_buffer_offset as u64,
                
                buffer_image_height : 0,
                buffer_row_length : 0,

                image_subresource : ash::vk::ImageSubresourceLayers{
                    aspect_mask : ash::vk::ImageAspectFlags::COLOR,
                    mip_level : 0,
                    base_array_layer : 0,
                    layer_count : 1
                },
                image_extent : ash::vk::Extent3D{width, height, depth : 1},
                image_offset : ash::vk::Offset3D{x : 0, y : 0, z : 0},
            };
            
            let barrier = ash::vk::ImageMemoryBarrier{
                src_access_mask : ash::vk::AccessFlags::NONE,
                dst_access_mask : ash::vk::AccessFlags::TRANSFER_WRITE | ash::vk::AccessFlags::SHADER_READ,
                old_layout : ash::vk::ImageLayout::UNDEFINED,
                new_layout : ash::vk::ImageLayout::GENERAL,
                image : data.image,
                subresource_range : ash::vk::ImageSubresourceRange{
                    aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                    base_array_layer : 0,
                    base_mip_level : 0,
                    level_count : 1,
                    layer_count : 1
                },
                
                ..Default::default()
            };

            vulkan.device.cmd_pipeline_barrier(cmd, ash::vk::PipelineStageFlags::ALL_COMMANDS, ash::vk::PipelineStageFlags::ALL_COMMANDS, ash::vk::DependencyFlags::BY_REGION,
                &[], &[], &[barrier]
            );
            vulkan.device.cmd_copy_buffer_to_image(cmd, host_buffer.buffer, data.image, ash::vk::ImageLayout::GENERAL, &[region]);
        }

        data
    }
}

impl ImageAllocPair{
    pub fn alloc(width : u32, height : u32, format : ash::vk::Format, usage : ash::vk::ImageUsageFlags, flags : ash::vk::MemoryPropertyFlags, vulkan : &VulkanSingleton) -> Self{
        let subresource_range = ash::vk::ImageSubresourceRange{
            aspect_mask: ash::vk::ImageAspectFlags::COLOR,
            base_array_layer : 0,
            base_mip_level : 0,
            level_count : 1,
            layer_count : 1
        };

        let image_image = ash::vk::ImageCreateInfo{
            flags : ash::vk::ImageCreateFlags::empty(),
            image_type : ash::vk::ImageType::TYPE_2D,
            format,
            extent : ash::vk::Extent3D{width, height, depth : 1},
            mip_levels : 1,
            array_layers : 1,
            samples : ash::vk::SampleCountFlags::TYPE_1,
            tiling : ash::vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode : ash::vk::SharingMode::EXCLUSIVE,
            initial_layout : ash::vk::ImageLayout::UNDEFINED,

            p_queue_family_indices : &vulkan.properties.main_family_index,
            queue_family_index_count : 1,

            ..Default::default()
        };

        unsafe{
            let image = vulkan.device.create_image(&image_image, None).unwrap();
            let info = vk_mem::AllocationCreateInfo{required_flags : flags, ..Default::default()};
            let alloc = vulkan.allocator.allocate_memory_for_image(image, &info).unwrap();
            let info = vulkan.allocator.get_allocation_info(&alloc);
            vulkan.device.bind_image_memory(image, info.device_memory, info.offset).unwrap();
            
            let components = ash::vk::ComponentMapping{r : ComponentSwizzle::R, a : ComponentSwizzle::A, g : ComponentSwizzle::G, b : ComponentSwizzle::B};
            let view_info = ash::vk::ImageViewCreateInfo{
                image, format, view_type : ash::vk::ImageViewType::TYPE_2D, subresource_range, components, ..Default::default()
            };
    
            let view: ash::vk::ImageView = vulkan.device.create_image_view(&view_info, None).unwrap();

            Self{view, image, alloc}
        }
    }
}