[[vk::binding(0, 0)]]
RWStructuredBuffer<uint64_t> voxel_fragment_buffer;

[[vk::binding(1, 0)]]
ConstantBuffer<Uniforms> constants;

[[vk::binding(3, 0)]]
Sampler2D<uint8_t4> image[];


struct Uniforms {
    float4 shift_and_scale;

    uint32_t voxel_depth;
    uint32_t max_storage_capacity;
    uint32_t voxel_resoultion;
    uint32_t pad_2;
}

struct VSInput {
    float3 position : Position;

#if defined(PerVertexUvs)
    float2 uvs : TEXCOORD0;
#elif defined(PerVertexColor)
    float3 color : COLOR0;
#endif
};

struct VSOutput {
    float3 position : Position;;
    uint instance_id;

#if defined(PerVertexUvs)
    float2 uvs : TEXCOORD0;
#elif defined(PerVertexColor)
    float3 color : COLOR0;
#endif

    static VSOutput from_input(VSInput input, uint instance_id) {
        VSOutput out;
        out.position = input.position;
        out.instance_id = instance_id;

#if defined(PerVertexUvs)
        out.uvs = input.uvs;
#elif defined(PerVertexColor)
        out.color = input.color;
#endif
        return out;
    }
};

struct GSOutput {
    float4 position : SV_Position;
#if defined(PerVertexUvs)
    float2 uvs : TEXCOORD0;
#elif defined(PerVertexColor)
    float3 color : COLOR0;
#endif
    nointerpolation uint axis : AXIS_VALUE;
    nointerpolation uint instance_id;

    static GSOutput from_vertex(VSOutput vertex, uint axis) {
        float3 transposed = (axis == 0) ? vertex.position.yzx : (axis == 1 ? vertex.position.zxy : vertex.position.xyz);
        transposed.z = (transposed.z + 1.0) * 0.5;

        GSOutput out;
        out.position = float4(transposed.x, transposed.y, transposed.z, 1.0);
        out.axis = axis;
        out.instance_id = vertex.instance_id;
#if defined(PerVertexUvs)
        out.uvs = vertex.uvs;
#elif defined(PerVertexColor)
        out.color = vertex.color;
#endif
        return out;
    }
}

uint max_normal_axis(VSOutput input[3]) {
    float3 delta_1 = input[1].position - input[0].position;
    float3 delta_2 = input[2].position - input[0].position;
    float3 normal = abs(cross(delta_1, delta_2));

    uint max_index = 0;
    for (int i = 1; i < 3; i++) {
        if (normal[i] > normal[max_index]) {
            max_index = i;
        }
    }

    return max_index;
}

int component_min(int3 val) {
    return min(min(val.x, val.y), val.z);
}

int component_max(int3 val) {
    return max(max(val.x, val.y), val.z);
}

Optional<uint3> GetVoxePos(float3 frag_cord, uint axis) {
    frag_cord.z *= constants.voxel_resoultion;

    int3 frag_voxel_signed = (int3)frag_cord;
    if ((component_min(frag_voxel_signed) < 0) || (component_max(frag_voxel_signed) >= constants.voxel_resoultion)) {
        return none;
    }
    uint3 frag_voxel = (uint3)frag_voxel_signed;

    return axis == 0 ? frag_voxel.zxy : (axis == 1 ? frag_voxel.yzx : frag_voxel.xyz);
}

[shader("vertex")]
VSOutput vertex(VSInput input, uint vertex_id: SV_VertexID, uint instance_id: SV_INSTANCEID) {
    float3 shifted = (input.position - constants.shift_and_scale.xyz) * constants.shift_and_scale.w;
    shifted = (shifted * 2.0) - 1.0;

    input.position = shifted;
    return VSOutput::from_input(input, instance_id);
}


[shader("geometry")]
[maxvertexcount(3)]
void geometry(triangle VSOutput input[3], inout TriangleStream<GSOutput> outputStream) {
    uint max_index = max_normal_axis(input);

    for (int i = 0; i < 3; i++) {
        outputStream.Append(GSOutput::from_vertex(input[i], max_index));
    }

    outputStream.RestartStrip();
}

[[vk::push_constant]]
ConstantBuffer<float3> uniform_color;

// https://en.wikipedia.org/wiki/SRGB
float linear_to_srgb(float linear) {
    if (linear <= 0.0031308) {
        return 12.92 * linear;
    } else {
        return ((1.055) * pow(linear, 1.0 / 2.4)) - 0.055;
    }
}

uint8_t quantize(float linear) {
    // float srgb = linear_to_srgb(linear);
    return uint8_t(linear * 255.0);
}

[shader("fragment")]
void fragment(GSOutput input) {
    Optional<uint3> mabey_pos = GetVoxePos(input.position.xyz, input.axis);
    if (!mabey_pos.hasValue) { discard;}

    uint3 transposed = mabey_pos.value;

#if defined(PerVertexUvs)
    uint8_t4 color = image[input.instance_id].Sample(input.uvs);
#elif defined(PerVertexColor)
    uint8_t4 color = uint8_t4(quantize(input.color.x), quantize(input.color.y), quantize(input.color.z), 255);
#elif 0 == 0
    uint8_t4 color = uint8_t4(quantize(uniform_color.x), quantize(uniform_color.y), quantize(uniform_color.z), 255);
#endif


    uint64_t previous;
    InterlockedAdd(voxel_fragment_buffer[0], 1, previous);

    uint64_t3 expanded_pos = (uint64_t3)transposed;
    uint base_offset = 3 * constants.voxel_depth;
    
    uint64_t packed = expanded_pos.x | (expanded_pos.y << constants.voxel_depth) | (expanded_pos.z << (2 * constants.voxel_depth));
    packed |= ((uint64_t)color.x << base_offset) | ((uint64_t)color.y << (base_offset + 8)) | (uint64_t)color.z << (base_offset + 16);
    
    voxel_fragment_buffer[1 + min(previous, constants.max_storage_capacity - 1)] = packed;
}
