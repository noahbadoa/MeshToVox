use super::VulkanSingleton;

const TIMELINE_INFO : ash::vk::SemaphoreTypeCreateInfo = ash::vk::SemaphoreTypeCreateInfo{
    semaphore_type : ash::vk::SemaphoreType::TIMELINE, initial_value : 0, p_next : std::ptr::null(), _marker : std::marker::PhantomData, s_type : ash::vk::StructureType::SEMAPHORE_TYPE_CREATE_INFO,
};

impl VulkanSingleton{
    pub fn create_semaphores<const N : usize>(&self, timeline : [bool; N]) -> [ash::vk::Semaphore; N]{
        let mut output = [ash::vk::Semaphore::null(); N];

        for i in 0..N{
            unsafe{
                if timeline[i]{
                    let info = ash::vk::SemaphoreCreateInfo{p_next : (&TIMELINE_INFO as *const ash::vk::SemaphoreTypeCreateInfo).cast::<std::ffi::c_void>(), ..Default::default()};
                    output[i] = self.device.create_semaphore(&info, None).expect("worked");
                }else{
                    output[i] = self.device.create_semaphore(&Default::default(), None).expect("worked");
                }   
            }
        }

        output
    }

    pub fn create_fences<const N : usize>(&self, signaled : [bool; N]) -> [ash::vk::Fence; N]{
        let mut output = [ash::vk::Fence::null(); N];
        let mut info = ash::vk::FenceCreateInfo::default();

        for i in 0..N{
            unsafe{
                info.flags = if signaled[i]{ash::vk::FenceCreateFlags::SIGNALED} else{ash::vk::FenceCreateFlags::empty()};

                output[i] = self.device.create_fence(&info, None).expect("worked");
            }
        }

        output
    }

    pub fn create_command_buffers<const N : usize>(&self, command_pool : ash::vk::CommandPool) -> Result<[ash::vk::CommandBuffer; N], ash::vk::Result>{unsafe{
        let info = ash::vk::CommandBufferAllocateInfo{
            command_pool, command_buffer_count : N as u32, level : ash::vk::CommandBufferLevel::PRIMARY, ..Default::default()
        };

        let mut out = std::mem::MaybeUninit::<[ash::vk::CommandBuffer; N]>::uninit();
        let result: ash::vk::Result = (self.device.fp_v1_0().allocate_command_buffers)(self.device.handle(), &info, out.as_mut_ptr().cast::<ash::vk::CommandBuffer>());

        result.result().map(|_|{out.assume_init()})
    }}

    pub fn allocate_descriptor_sets<const N : usize>(&self, descriptor_pool : ash::vk::DescriptorPool, layouts : [ash::vk::DescriptorSetLayout; N]) -> Result<[ash::vk::DescriptorSet; N], ash::vk::Result>{unsafe{
        let info = ash::vk::DescriptorSetAllocateInfo{descriptor_pool, descriptor_set_count : N as u32, p_set_layouts : layouts.as_ptr(), ..Default::default()};

        let mut out = std::mem::MaybeUninit::<[ash::vk::DescriptorSet; N]>::uninit();
        let result = (self.device.fp_v1_0().allocate_descriptor_sets)(self.device.handle(), &info, out.as_mut_ptr().cast::<ash::vk::DescriptorSet>());

        result.result().map(|_|{out.assume_init()})
    }}
}


use vk_mem::Alloc;
pub struct PersistentBufferAllocation{
    pub allocation : vk_mem::Allocation,
    pub buffer : ash::vk::Buffer,
    pub info: vk_mem::AllocationInfo,
}


impl std::fmt::Debug for PersistentBufferAllocation{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PersistentBufferAllocation {{size : {:?}}}", self.info.size)
    }
}

unsafe impl Sync for PersistentBufferAllocation{}
unsafe impl Send for PersistentBufferAllocation{}

#[allow(unused)]
impl PersistentBufferAllocation{
    pub fn new_exclusive(flags : ash::vk::MemoryPropertyFlags, usage : ash::vk::BufferUsageFlags, size : u64, vulkan : &VulkanSingleton) -> Self{unsafe{
        let buffer_info = ash::vk::BufferCreateInfo{size, usage, sharing_mode : ash::vk::SharingMode::EXCLUSIVE, ..Default::default()};
        let create_info = vk_mem::AllocationCreateInfo{flags : vk_mem::AllocationCreateFlags::MAPPED, required_flags : flags, usage : vk_mem::MemoryUsage::Unknown, ..Default::default()};

        let (buffer, allocation) = vulkan.allocator.create_buffer(&buffer_info, &create_info).unwrap();
        let mut info: vk_mem::AllocationInfo = vulkan.allocator.get_allocation_info(&allocation);
        info.size = size;

        Self { allocation, buffer, info }
    }}

    pub fn as_slice<'a>(&'a self) -> &'a [u8]{
        unsafe{std::slice::from_raw_parts(self.info.mapped_data.cast::<u8>(), self.info.size as usize)}
    }
    pub fn as_mut_slice<'a>(&'a self) -> &'a mut [u8]{
        unsafe{std::slice::from_raw_parts_mut(self.info.mapped_data.cast::<u8>(), self.info.size as usize)}
    }
    pub const fn to_descriptor(&self) -> ash::vk::DescriptorBufferInfo{
        ash::vk::DescriptorBufferInfo{buffer : self.buffer, offset : 0, range : ash::vk::WHOLE_SIZE}
    }
    pub fn to_memory_range_aligned(&self, vulkan : &VulkanSingleton) -> ash::vk::MappedMemoryRange<'static>{
        let denom = vulkan.properties.device.limits.non_coherent_atom_size;
        let offset = (self.info.offset / denom) * denom;

        let size = self.info.size + (self.info.offset - offset);
        let size = ((size + (denom - 1)) / denom) * denom;

        ash::vk::MappedMemoryRange{memory : self.info.device_memory, offset, size, ..Default::default()}
    }
    pub fn drop<'a, 'b>(&'a mut self, vulkan : &'b VulkanSingleton){unsafe{
        vulkan.allocator.destroy_buffer(self.buffer, &mut self.allocation);
    }}
}

#[derive(Debug)]
pub struct GpuBufferAllocation{
    pub allocation : vk_mem::Allocation,
    pub buffer : ash::vk::Buffer,
}

unsafe impl Sync for GpuBufferAllocation{}
unsafe impl Send for GpuBufferAllocation{}

#[allow(unused)]
impl GpuBufferAllocation{
    pub fn new_exclusive(flags : ash::vk::MemoryPropertyFlags, usage : ash::vk::BufferUsageFlags, size : u64, vulkan : &VulkanSingleton) -> Self{unsafe{
        let buffer_info = ash::vk::BufferCreateInfo{size, usage, sharing_mode : ash::vk::SharingMode::EXCLUSIVE, ..Default::default()};
        let create_info = vk_mem::AllocationCreateInfo{flags : vk_mem::AllocationCreateFlags::empty(), required_flags : flags, usage : vk_mem::MemoryUsage::Unknown, ..Default::default()};
        let (buffer, allocation) = vulkan.allocator.create_buffer(&buffer_info, &create_info).unwrap();

        Self { allocation, buffer}
    }}

    pub fn to_descriptor(&self) -> ash::vk::DescriptorBufferInfo{
        ash::vk::DescriptorBufferInfo{buffer : self.buffer, offset : 0, range : ash::vk::WHOLE_SIZE}
    }
    pub fn to_memory_range(info: vk_mem::AllocationInfo) -> ash::vk::MappedMemoryRange<'static>{
        ash::vk::MappedMemoryRange{memory : info.device_memory, offset : info.offset, size : info.size, ..Default::default()}
    }
    pub fn drop<'a, 'b>(&'a mut self, vulkan : &'b VulkanSingleton){unsafe{
        vulkan.allocator.destroy_buffer(self.buffer, &mut self.allocation);
    }}
}
