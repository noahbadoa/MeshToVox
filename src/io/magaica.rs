
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

pub const fn _const_assert(){
    let mut counter = 0;
    loop {
        if encode(decode(counter)) != counter{panic!()}
        if counter == u8::MAX{break;}
        counter += 1;
    }
}

pub const _ : () = _const_assert();

pub fn save_as_magica_voxel<S: AsRef<std::path::Path> + ?Sized>(data : impl Iterator<Item = ([u32; 3], image::Rgb<u8>)>, file_path : &S, size : u32) -> std::io::Result<()> {
    let mut vox = vox_writer::VoxWriter::create(size as i32, size as i32, size as i32);
    
    for index in 0..u8::MAX{
        let color = decode(index);
        vox.add_color(color.0[0], color.0[1], color.0[1], 0, index);
    }

    for (cord, color) in data{
        let color_idx = encode(color);
        vox.add_voxel(cord[0] as i32, cord[1] as i32, cord[2] as i32, color_idx as i32);
    }
    
    let path = file_path.as_ref().as_os_str();
    vox.save_to_file(path.to_str().unwrap().to_string())
}
