// GPU Polyline shader — renders polylines as tube geometry entirely on the GPU.
//
// Instead of pre-computing cylinder/sphere meshes on the CPU, this shader
// takes polyline points as a vertex buffer and expands each point into a
// tube cross-section in the vertex shader using point sprites.
//
// # Vertex Input
//
// Each vertex is a polyline point with:
// - @location(0) position: vec3 (point position) + padding
// - @location(1) thickness: f32 (tube radius)
// - @location(2) segment_id: f32 (which segment this point belongs to)
//
// The vertex shader computes the tube geometry by offsetting each point
// along perpendicular axes based on the angle parameter.

struct PolylineVertex {
    @location(0) position: vec3<f32>,
    @location(1) thickness: f32,
    @location(2) segment_id: f32,
    @location(3) angle: f32,
    @location(4) t: f32,  // position within segment [0,1]
}

struct PolylineOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

// Uniform buffer for view/projection matrices
struct ViewUniforms {
    view_proj: mat4x4<f32>,
    _padding: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> view_uniforms: ViewUniforms;

/// Compute a perpendicular vector for tube cross-section.
/// Given a segment direction, returns two orthogonal vectors spanning
/// the cross-section plane.
fn tube_perpendiculars(dir: vec3<f32>) -> vec2<vec3<f32>> {
    // Avoid degeneracy by choosing the least-aligned axis
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(dot(dir, up)) > 0.999) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(dir, up));
    let corrected_up = cross(dir, right);
    return vec2<vec3<f32>>(right, corrected_up);
}

@vertex
fn vertex(input: PolylineVertex) -> PolylineOutput {
    let pos = input.position;
    let radius = input.thickness;
    let angle = input.angle;
    
    // For a single point, we approximate the direction from the segment_id
    // In a full implementation, this would come from the point buffer
    // For now, we use a simple sphere expansion around each point
    
    // Compute offset from angle for tube cross-section
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    
    // Simple sphere-like expansion (no direction-based tube shaping)
    // This creates a sphere at each point, which is simpler than tubes
    let phi = asin(clamp(sin_a, -1.0, 1.0));
    let theta = angle;
    
    let offset = vec3<f32>(
        cos(theta) * cos(phi),
        sin(phi),
        sin(theta) * cos(phi)
    ) * radius;
    
    let world_pos = pos + offset;
    let normal = normalize(offset);
    
    let clip_pos = view_uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    
    return PolylineOutput {
        clip_position: clip_pos,
        world_position: vec4<f32>(world_pos, 1.0),
        world_normal: normal,
        uv: vec2<f32>(input.t, angle / 6.28318),
    };
}

@fragment
fn fragment(input: PolylineOutput) -> @location(0) vec4<f32> {
    // Simple diffuse shading
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let normal = normalize(input.world_normal);
    let diff = max(dot(normal, light_dir), 0.0);
    
    let color = vec3<f32>(0.5 + diff * 0.5, 0.5 + diff * 0.5, 0.8 + diff * 0.2);
    return vec4<f32>(color, 1.0);
}
