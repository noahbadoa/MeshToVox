use crate::split::VulkanSingleton;

pub struct DescriptorState{
    pub pool : ash::vk::DescriptorPool,
    pub set : SetDescriptor,
}

impl DescriptorState{
    pub fn new(vulkan : &VulkanSingleton, draw_count : u32) -> Self{
        let pool_info = [
            ash::vk::DescriptorPoolSize{ty : ash::vk::DescriptorType::STORAGE_BUFFER, descriptor_count : 1},
            ash::vk::DescriptorPoolSize{ty : ash::vk::DescriptorType::UNIFORM_BUFFER, descriptor_count : 1},
            ash::vk::DescriptorPoolSize{ty : ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count : draw_count.max(1)},
        ];

        let pool = ash::vk::DescriptorPoolCreateInfo{
            max_sets : 1,
            pool_size_count : pool_info.len() as u32,
            p_pool_sizes : pool_info.as_ptr(),
            flags : ash::vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,

            ..Default::default()
        };
        
        let pool = unsafe{vulkan.device.create_descriptor_pool(&pool, None)}.unwrap();

        let output_storage = SetDescriptor::create(0, 1, ash::vk::DescriptorType::STORAGE_BUFFER, ash::vk::ShaderStageFlags::FRAGMENT);
        let uniforms = SetDescriptor::create(1, 1, ash::vk::DescriptorType::UNIFORM_BUFFER, ash::vk::ShaderStageFlags::ALL_GRAPHICS);
        let textures = SetDescriptor::create(3, draw_count, ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER, ash::vk::ShaderStageFlags::FRAGMENT);

        let set = SetDescriptor::new(&[output_storage, uniforms, textures], pool, vulkan);

        Self { pool, set }
    }

    pub fn update_output_buffer(&self, vulkan : &VulkanSingleton, output_buffer_info : ash::vk::DescriptorBufferInfo){
        let write1 = ash::vk::WriteDescriptorSet{
            dst_set : self.set.main,
            dst_binding : 0,
            dst_array_element : 0,
            descriptor_count : 1,
            descriptor_type : ash::vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info : &output_buffer_info,
            ..Default::default()
        };

        unsafe{vulkan.device.update_descriptor_sets(&[write1], &[])};
    }

    pub fn write(&self, vulkan : &VulkanSingleton, output_buffer_info : ash::vk::DescriptorBufferInfo, uniform_buffer_info : ash::vk::DescriptorBufferInfo, writes : &[(ash::vk::DescriptorImageInfo, usize)]){
        let write1 = ash::vk::WriteDescriptorSet{
            dst_set : self.set.main,
            dst_binding : 0,
            dst_array_element : 0,
            descriptor_count : 1,
            descriptor_type : ash::vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info : &output_buffer_info,
            ..Default::default()
        };

        let write2 = ash::vk::WriteDescriptorSet{
            dst_set : self.set.main,
            dst_binding : 1,
            dst_array_element : 0,
            descriptor_count : 1,
            descriptor_type : ash::vk::DescriptorType::UNIFORM_BUFFER,
            p_buffer_info : &uniform_buffer_info,
            ..Default::default()
        };

        let mut extra = writes.iter().map(|(write, index)|{
            ash::vk::WriteDescriptorSet{
                dst_set : self.set.main,
                dst_binding : 3,
                dst_array_element : *index as u32,
                descriptor_count : 1,
                descriptor_type : ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info : write,
                ..Default::default()
            }
        }).collect::<Vec<_>>();
        extra.extend([write1, write2]);

        unsafe{vulkan.device.update_descriptor_sets(extra.as_slice(), &[])};
    }

    pub fn drop(&mut self, vulkan : &VulkanSingleton){
        unsafe{
            vulkan.device.destroy_descriptor_pool(self.pool, None);
            self.set.drop(vulkan);
        }
    } 
}


#[derive(Debug)]
pub struct SetDescriptor{
    pub main : ash::vk::DescriptorSet,
    pub layout : ash::vk::DescriptorSetLayout,
}

impl SetDescriptor{
    pub const fn create(binding : u32, descriptor_count : u32, descriptor_type : ash::vk::DescriptorType, stage : ash::vk::ShaderStageFlags) -> ash::vk::DescriptorSetLayoutBinding<'static>{
        ash::vk::DescriptorSetLayoutBinding{binding, descriptor_count, descriptor_type, stage_flags : stage, p_immutable_samplers : std::ptr::null(), _marker : std::marker::PhantomData}
    }

    pub fn new(layouts : &[ash::vk::DescriptorSetLayoutBinding<'static>], pool: ash::vk::DescriptorPool, vulkan : &VulkanSingleton) -> Self{
        let set_layouts_info = ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(layouts);
        let layout  = unsafe{vulkan.device.create_descriptor_set_layout(&set_layouts_info, None)}.expect("worked");
        let individual = vulkan.allocate_descriptor_sets(pool, [layout]).expect("worked");

        Self {layout, main : individual[0]}
    }

    pub fn drop(&self,  vulkan : &VulkanSingleton){unsafe{
        //okay DescriptorSet(s) don't need to be destoryed; destoryed with pool
        vulkan.device.destroy_descriptor_set_layout(self.layout, None);
    }}
}

