use std::ptr::addr_of;


#[allow(unused)]
pub struct Properties{
    pub memory: ash::vk::PhysicalDeviceMemoryProperties,
    pub device: ash::vk::PhysicalDeviceProperties,
    pub queue_families : Vec<ash::vk::QueueFamilyProperties>,

    pub main_family_index : u32,
}

impl Properties{
    pub fn new(instance: &ash::Instance, physical_device: ash::vk::PhysicalDevice) -> Self{unsafe{
        let memory: ash::vk::PhysicalDeviceMemoryProperties = instance.get_physical_device_memory_properties(physical_device);
        let device: ash::vk::PhysicalDeviceProperties = instance.get_physical_device_properties(physical_device);
        let queue_families: Vec<ash::vk::QueueFamilyProperties> = instance.get_physical_device_queue_family_properties(physical_device);

        let main_family_index = queue_families.iter().position(|x|{
            let a = x.queue_flags.contains(ash::vk::QueueFlags::COMPUTE);
            let b = x.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS);
            let c = x.queue_flags.contains(ash::vk::QueueFlags::TRANSFER);
            
            a && b && c
        }).expect("not sure if this is gauntured to always work") as u32;

        Self { memory, device, queue_families, main_family_index}
    }}
}

#[repr(C)]
pub struct VulkanSingleton{
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device : ash::Device,
    pub queue : ash::vk::Queue,

    pub physical_device: ash::vk::PhysicalDevice,
    pub properties : Properties,
    
    pub allocator : std::mem::ManuallyDrop<vk_mem::Allocator>,

    pub debug : Option<DebugInstace>,
}

impl std::fmt::Debug for VulkanSingleton{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VulkanSingleton")
    }
}

unsafe impl Send for VulkanSingleton{}
unsafe impl Sync for VulkanSingleton{}

pub struct DebugInstace{
    instance : ash::ext::debug_utils::Instance,
    messenger : ash::vk::DebugUtilsMessengerEXT,
}

impl DebugInstace{
    pub fn drop(&self){
        unsafe{self.instance.destroy_debug_utils_messenger(self.messenger, None)};
    }

    pub fn new(entry : &ash::Entry, instance : &ash::Instance) -> Self{
        let debug_instance = ash::ext::debug_utils::Instance::new(entry, instance);

        let debug_info = ash::vk::DebugUtilsMessengerCreateInfoEXT{
            pfn_user_callback : Some(vulkan_debug_callback),
            message_severity : ash::vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | ash::vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | ash::vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            message_type : ash::vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | ash::vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | ash::vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            ..Default::default()
        };

        let messenger = unsafe{debug_instance.create_debug_utils_messenger(&debug_info, None)}.expect("debug layers");

        DebugInstace{messenger, instance : debug_instance}
    }
}

pub unsafe extern "system" fn vulkan_debug_callback(
    message_severity: ash::vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: ash::vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const ash::vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> ash::vk::Bool32 {

    unsafe{
        let callback_data = *p_callback_data;
        let message_id_number: i32 = callback_data.message_id_number as i32;

        let message_id_name = if callback_data.p_message_id_name.is_null() {
            std::borrow::Cow::from("")
        } else {
            std::ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
        };

        let message = if callback_data.p_message.is_null() {
            std::borrow::Cow::from("")
        } else {
            std::ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
        };

        if let ash::vk::DebugUtilsMessageSeverityFlagsEXT::INFO = message_severity{
            println!("{}", message);
            return ash::vk::FALSE;
        } 
    
        println!(
            "{:?}: {:?} [{} ({})] : {}",
            message_severity,
            message_type,
            message_id_name,
            &message_id_number.to_string(),
            message,
        );
    }

    match message_severity {
        ash::vk::DebugUtilsMessageSeverityFlagsEXT::ERROR =>{
            panic!()
        }

        _ => ash::vk::FALSE
    }
}

impl VulkanSingleton{
    fn new_instance(entry : &ash::Entry, debug : bool) -> ash::Instance{unsafe{        
        let app_info = ash::vk::ApplicationInfo {
            api_version: ash::vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };
        
        let mut extensions = Vec::new();

        let printf_extenion = ash::vk::ValidationFeaturesEXT{..Default::default()}.enabled_validation_features(&[
            ash::vk::ValidationFeatureEnableEXT::GPU_ASSISTED,
            ash::vk::ValidationFeatureEnableEXT::GPU_ASSISTED_RESERVE_BINDING_SLOT,
            
            ash::vk::ValidationFeatureEnableEXT::DEBUG_PRINTF,
            ash::vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION,
            //ash::vk::ValidationFeatureEnableEXT::BEST_PRACTICES,
        ]);

        let value = ash::vk::TRUE;
        let setting = ash::vk::LayerSettingEXT{
            p_layer_name : c"VK_LAYER_KHRONOS_validation".as_ptr(), p_setting_name : c"gpuav_safe_mode".as_ptr(), 
            ty : ash::vk::LayerSettingTypeEXT::BOOL32, value_count : std::mem::size_of_val(&value) as u32, p_values : std::ptr::addr_of!(value).cast(), ..Default::default()
        };

        let settings = [setting];
        let mut settings = ash::vk::LayerSettingsCreateInfoEXT::default().settings(&settings);
        settings.p_next = std::ptr::addr_of!(printf_extenion).cast();

        let instance_info = if debug {
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());

            ash::vk::InstanceCreateInfo {
                p_application_info: &app_info,

                enabled_extension_count : extensions.len() as u32,
                pp_enabled_extension_names : extensions.as_ptr(),

                enabled_layer_count : 1,
                pp_enabled_layer_names : &[c"VK_LAYER_KHRONOS_validation".as_ptr()] as *const _,

                p_next : addr_of!(settings).cast(),
                
                ..Default::default()
            }
        } else {
            ash::vk::InstanceCreateInfo {
            p_application_info: &app_info,

            enabled_extension_count : extensions.len() as u32,
            pp_enabled_extension_names : extensions.as_ptr(),

            ..Default::default()
            }
        };

        
        entry.create_instance(&instance_info, None).unwrap()
    }}

    pub fn new(debug : bool, device_types : &[ash::vk::PhysicalDeviceType]) -> VulkanSingleton{unsafe{
        // let entry: ash::Entry = ash::Entry::linked();
        let entry: ash::Entry = ash::Entry::load().unwrap();
        let instance = Self::new_instance(&entry, debug);
        let debug_instance = if debug {Some(DebugInstace::new(&entry, &instance))} else {None};
        let device_list: Vec<ash::vk::PhysicalDevice> = instance.enumerate_physical_devices().unwrap();
        
        let mut v12 = ash::vk::PhysicalDeviceVulkan12Features{
            runtime_descriptor_array : ash::vk::TRUE,
            shader_buffer_int64_atomics : ash::vk::TRUE,
            shader_int8 : ash::vk::TRUE,
            ..Default::default()
        };

        let v11: ash::vk::PhysicalDeviceVulkan11Features<'_> = ash::vk::PhysicalDeviceVulkan11Features{
            shader_draw_parameters : ash::vk::TRUE,
            p_next : std::ptr::addr_of_mut!(v12).cast(),
            ..Default::default()
        };

        let features = ash::vk::PhysicalDeviceFeatures{
            shader_int64 : ash::vk::TRUE,
            geometry_shader : ash::vk::TRUE,
            fragment_stores_and_atomics : ash::vk::TRUE,
            ..Default::default()
        };

        let enabled_extension_names = [ash::ext::conservative_rasterization::NAME.as_ptr()];
        
        let mut suitable_devices = device_list.iter().filter(|device|{
            let properties = instance.get_physical_device_properties(**device);
            let extensions = instance.enumerate_device_extension_properties(**device).unwrap();
            // let features = instance.get_physical_device_features(**device);

            let has_properties = device_types.iter().any(|x|{*x == properties.device_type});
            let has_extensions = enabled_extension_names.iter().all(|extension|{
                extensions.iter().any(|x|{
                    libc::strcmp(x.extension_name.as_ptr(), *extension) == 0
                })
            });
       

            has_properties && has_extensions
        });

        let physical_device: ash::vk::PhysicalDevice = *suitable_devices.next().expect("suitable vulkan device");

        let properties = Properties::new(&instance, physical_device);


        let queue_priorities = 1.0f32;
        let p_queue_priorities = addr_of!(queue_priorities);

        let main_queue = ash::vk::DeviceQueueCreateInfo{queue_count : 1, p_queue_priorities, queue_family_index : properties.main_family_index as u32, ..Default::default()};
        let queue_create_infos = [main_queue];

        let info  = ash::vk::DeviceCreateInfo{
            p_enabled_features : addr_of!(features),
            p_next : addr_of!(v11).cast::<core::ffi::c_void>(),
            ..Default::default()
        }.queue_create_infos(queue_create_infos.as_slice()).enabled_extension_names(&enabled_extension_names);
        
        
        let logical_deivce: ash::Device = instance.create_device(physical_device, &info, None).unwrap();
        let queue: ash::vk::Queue = logical_deivce.get_device_queue(properties.main_family_index as u32, 0);

        let allocator_info = vk_mem::AllocatorCreateInfo::new(&instance, &logical_deivce, physical_device);
        let allocator = vk_mem::Allocator::new(allocator_info).expect("worked");


        VulkanSingleton{
            entry,
            instance, 
            device : logical_deivce,
            physical_device,
            properties,
            queue,
            allocator : std::mem::ManuallyDrop::new(allocator),
            debug : debug_instance
        }}
    }

    pub fn drop(&mut self){
        unsafe{
            std::mem::ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);

            if let Some(ref debug) = self.debug{debug.drop();}
            self.instance.destroy_instance(None);
        }
    }
}
