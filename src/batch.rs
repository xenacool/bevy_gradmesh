//! Instance batching for BevyGradmesh.
//!
//! Groups shapes that share the same mesh geometry and material variant into
//! batches, so thousands of shapes can be rendered in far fewer draw calls.
//!
//! # Design
//!
//! Each batch is keyed by (mesh, blend_mode, render_state).  Within a batch,
//! every instance contributes:
//! - A `Transform` (12 f32s stored as 4x`Vec4`)
//! - Per-instance gradmesh uniforms (5x`Vec4` = color, secondary, direction,
//!   flags, dash)
//!
//! At render time the batch's shared mesh is drawn once with instancing; the
//! vertex shader reads per-instance attributes to compute world position and
//! gradient shading.

use bevy::math::{Mat4, Quat, Vec3};
use bevy::render::mesh::Mesh;
use bevy::render::render_resource::ShaderType;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::RenderState;
use crate::appearance::AppearanceSnapshot;
#[cfg(feature = "pbr")]
use crate::culling::AABB;
use crate::material::{GradMeshUniformBuffer, Vec4};
use crate::mix_space::MixSpace;
use crate::shape::{Shape, ShapeDetail};
use crate::style::{BlendMode, ColorMode, Dash, DashType, Style};
use bevy::color::ColorToComponents;
use bevy::prelude::Color;

// --- Per-instance GPU data ---

/// Per-instance transform stored as 4xvec4 (column-major Mat4 flattened).
/// Matches what the vertex shader expects for instanced transforms.
#[derive(Debug, Clone, Copy, Default, ShaderType)]
pub struct InstanceTransform {
    pub col0: Vec4,
    pub col1: Vec4,
    pub col2: Vec4,
    pub col3: Vec4,
}

impl InstanceTransform {
    /// Build from a Bevy `Transform`-like (position, rotation, scale).
    pub fn from_trs(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        let mat = Mat4::from_scale_rotation_translation(scale, rotation, position);
        Self {
            col0: Vec4::new(mat.x_axis.x, mat.x_axis.y, mat.x_axis.z, mat.x_axis.w),
            col1: Vec4::new(mat.y_axis.x, mat.y_axis.y, mat.y_axis.z, mat.y_axis.w),
            col2: Vec4::new(mat.z_axis.x, mat.z_axis.y, mat.z_axis.z, mat.z_axis.w),
            col3: Vec4::new(mat.w_axis.x, mat.w_axis.y, mat.w_axis.z, mat.w_axis.w),
        }
    }
}

/// Per-instance gradmesh properties (color, secondary, direction, flags, dash, deformation, appearance).
/// 7xVec4 = 112 bytes per instance.
#[derive(Debug, Clone, Copy, Default, ShaderType)]
pub struct InstanceProperties {
    pub color: Vec4,
    pub secondary_color: Vec4,
    pub direction: Vec4,
    pub flags: Vec4,
    pub dash: Vec4,
    pub deformation: Vec4,
    pub appearance: Vec4,
}

/// One instance as the vertex shader reads it: transform then properties, 64 + 112 = 176 bytes.
///
/// This is the record both instanced pipelines upload through Bevy's encase-based
/// `BufferVec`. Its size is the `array_stride` of their instance vertex layouts, and a
/// unit test pins the two together.
#[derive(Debug, Clone, Copy, Default, ShaderType)]
pub struct InstanceData {
    pub transform: InstanceTransform,
    pub properties: InstanceProperties,
}

impl InstanceProperties {
    pub fn from_uniform(buf: &GradMeshUniformBuffer) -> Self {
        Self {
            color: buf.color,
            secondary_color: buf.secondary_color,
            direction: buf.direction,
            flags: buf.flags,
            dash: buf.dash,
            deformation: Vec4::default(),
            appearance: Vec4::default(),
        }
    }

    /// Build instance properties from loose values.
    ///
    /// Colours are sRGB components; the shaders convert them to linear. When you have a
    /// [`Style`], prefer `InstanceProperties::from(&style)`.
    pub fn from_style(
        color: [f32; 4],
        secondary: [f32; 4],
        direction: [f32; 4],
        color_mode: ColorMode,
        blend_mode: BlendMode,
        space: MixSpace,
        dash: Option<Dash>,
    ) -> Self {
        let cm = match color_mode {
            ColorMode::Single => 0.0,
            ColorMode::DualDirectional => 1.0,
            ColorMode::DualRadial => 2.0,
            // Height bands only exist on the material path.
            ColorMode::HeightBands => 0.0,
        };
        let bm = match blend_mode {
            BlendMode::Opaque => 0.0,
            BlendMode::Transparent => 1.0,
            BlendMode::Additive => 2.0,
            BlendMode::Multiplicative => 3.0,
            BlendMode::Screen => 4.0,
            BlendMode::Dither => 5.0,
            BlendMode::Subtractive => 6.0,
            BlendMode::Lighten => 7.0,
            BlendMode::ColorDodge => 8.0,
            BlendMode::LinearBurn => 9.0,
            BlendMode::ColorBurn => 10.0,
            BlendMode::Darken => 11.0,
            BlendMode::Custom(_) => 12.0,
        };
        let (dash_count, dash_type, dash_spacing, dash_offset) = match dash {
            Some(d) => {
                let type_base = match d.style {
                    DashType::Normal => 0.1,
                    DashType::Chevron => 0.4,
                    DashType::Rounded => 0.7,
                };
                let type_code = type_base + 0.075 * d.modifier.clamp(-1.0, 1.0);
                (d.count as f32, type_code, d.spacing, d.offset)
            }
            None => (0.0, 0.0, 0.0, 0.0),
        };

        Self {
            color: Vec4::new(color[0], color[1], color[2], color[3]),
            secondary_color: Vec4::new(secondary[0], secondary[1], secondary[2], secondary[3]),
            direction: Vec4::new(direction[0], direction[1], direction[2], direction[3]),
            flags: Vec4::new(cm, bm, 0.0, space.gpu_code()),
            dash: Vec4::new(dash_count, dash_type, dash_spacing, dash_offset),
            deformation: Vec4::default(),
            appearance: Vec4::default(),
        }
    }

    /// Set appearance animation parameters.
    pub fn with_appearance(mut self, appearance: AppearanceSnapshot) -> Self {
        self.appearance = appearance.to_vec4();
        self
    }

    /// Build instance properties from a [`Style`](style::Style).
    ///
    /// Equivalent to `InstanceProperties::from(style)`.
    pub fn from_style_descriptor(style: &Style) -> Self {
        Self::from(style)
    }
}

/// Extracts the colours, direction, blend mode, mix space and dash from a `Style`.
///
/// PBR-only style features (roughness, metallic, emissive, noise, stripes, rim, height bands)
/// are not available on the instanced and standalone paths and are ignored. Colours are
/// passed as sRGB components because the shaders convert them to linear themselves.
impl From<&Style> for InstanceProperties {
    fn from(style: &Style) -> Self {
        let color = style.color.to_srgba().to_f32_array();
        let secondary = style
            .secondary_color
            .unwrap_or(Color::WHITE)
            .to_srgba()
            .to_f32_array();
        let direction = [
            style.direction.x,
            style.direction.y,
            style.direction.z,
            0.0,
        ];
        Self::from_style(
            color,
            secondary,
            direction,
            style.color_mode,
            style.blend_mode.clone(),
            style.space,
            style.dash,
        )
    }
}

// --- Batch key ---

/// Identifies a render batch: shapes with the same mesh geometry and material
/// variant can share a single draw call.
#[derive(Debug, Clone)]
pub struct BatchKey {
    /// Recorder-local mesh identifier, unique per distinct shape (kind + parameters + detail).
    pub mesh_id: u64,
    /// Blend mode determines the render pass and sorting.
    pub blend_mode: BlendMode,
    /// Depth state affects render order.
    pub render_state: RenderState,
}

impl PartialEq for BatchKey {
    fn eq(&self, other: &Self) -> bool {
        self.mesh_id == other.mesh_id
            && self.blend_mode == other.blend_mode
            && self.render_state == other.render_state
    }
}

impl Eq for BatchKey {}

impl Hash for BatchKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.mesh_id.hash(state);
        self.blend_mode.hash(state);
        self.render_state.hash(state);
    }
}

// --- Mesh cache key ---

/// Canonical key for procedural mesh caching.
#[deprecated(
    since = "0.3.3",
    note = "unused; the recorder now keys its mesh cache on `Shape` directly"
)]
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct MeshCacheKey {
    /// Shape descriptor (kind + parameters + detail).
    pub shape_hash: u64,
}

// --- Batch ---

/// A single render batch: one mesh + N instances.
///
/// Supports dirty tracking so only changed batches upload new GPU data
/// each frame
pub struct Batch {
    /// Shared mesh geometry.
    pub mesh: Mesh,
    /// Per-instance transforms.
    pub transforms: Vec<InstanceTransform>,
    /// Per-instance properties.
    pub properties: Vec<InstanceProperties>,
    /// Per-instance deformation parameters.
    pub deformations: Vec<crate::deformation::DeformationParams>,
    /// Number of active instances.
    pub count: usize,
    /// Capacity (for growth).
    pub capacity: usize,
    /// Whether transform data has changed since last render.
    pub transform_pending: bool,
    /// Whether property data has changed since last render.
    pub property_pending: bool,
    /// Monotonically increasing counter; incremented when transforms change.
    pub transform_revision: u32,
    /// Monotonically increasing counter; incremented when properties change.
    pub property_revision: u32,
}

impl Batch {
    const INITIAL_CAPACITY: usize = 256;

    pub fn new(mesh: Mesh) -> Self {
        Self {
            mesh,
            transforms: Vec::with_capacity(Self::INITIAL_CAPACITY),
            properties: Vec::with_capacity(Self::INITIAL_CAPACITY),
            deformations: Vec::with_capacity(Self::INITIAL_CAPACITY),
            count: 0,
            capacity: Self::INITIAL_CAPACITY,
            transform_pending: false,
            property_pending: false,
            transform_revision: 0,
            property_revision: 0,
        }
    }

    /// Add an instance to this batch.
    pub fn push(&mut self, transform: InstanceTransform, properties: InstanceProperties) {
        if self.count >= self.capacity {
            let new_cap = self.capacity.saturating_mul(2);
            self.capacity = new_cap;
        }
        self.transforms.push(transform);
        self.properties.push(properties);
        self.deformations
            .push(crate::deformation::DeformationParams::none());
        self.count += 1;
        self.mark_dirty();
    }

    /// Set deformation for an instance at the given index.
    pub fn set_deformation(
        &mut self,
        index: usize,
        deformation: crate::deformation::DeformationParams,
    ) {
        if index < self.count {
            self.deformations[index] = deformation;
            if !self.property_pending {
                self.property_revision += 1;
                self.property_pending = true;
            }
        }
    }

    /// Update the transform at the given index.
    pub fn set_transform(&mut self, index: usize, transform: InstanceTransform) {
        if index < self.count {
            self.transforms[index] = transform;
            if !self.transform_pending {
                self.transform_revision += 1;
                self.transform_pending = true;
            }
        }
    }

    /// Update the properties at the given index.
    pub fn set_properties(&mut self, index: usize, properties: InstanceProperties) {
        if index < self.count {
            self.properties[index] = properties;
            if !self.property_pending {
                self.property_revision += 1;
                self.property_pending = true;
            }
        }
    }

    /// Clear all instances (for immediate mode frame reset).
    pub fn clear(&mut self) {
        self.transforms.clear();
        self.properties.clear();
        self.deformations.clear();
        self.count = 0;
        self.mark_dirty();
    }

    /// Returns true if the batch has no instances.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Mark both transforms and properties as dirty.
    pub fn mark_dirty(&mut self) {
        if !self.transform_pending {
            self.transform_revision += 1;
            self.transform_pending = true;
        }
        if !self.property_pending {
            self.property_revision += 1;
            self.property_pending = true;
        }
    }

    /// Acknowledge that changes have been uploaded/rendered.
    /// Called after a frame's render pass completes.
    pub fn acknowledge_changes(&mut self) {
        self.transform_pending = false;
        self.property_pending = false;
    }

    /// Returns true if any data needs to be uploaded this frame.
    pub fn is_dirty(&self) -> bool {
        self.transform_pending || self.property_pending
    }
}

// --- Grouping ---

/// Recorder-local mesh cache: each distinct shape with its sequential id and built mesh.
type RecorderMeshCache = HashMap<Shape, (u64, Mesh)>;

/// Returns the cached mesh id for `shape`, building the mesh on first use only.
fn mesh_id_for(cache: &mut RecorderMeshCache, shape: &Shape) -> u64 {
    if let Some((id, _)) = cache.get(shape) {
        return *id;
    }
    let id = cache.len() as u64;
    cache.insert(shape.clone(), (id, shape.build()));
    id
}

/// Appends one instance to the batch that matches its shape, blend mode and render state,
/// creating that batch on first use. This is the single place that decides which instances
/// share a draw call.
fn place_instance(
    batches: &mut HashMap<BatchKey, Batch>,
    mesh_cache: &mut RecorderMeshCache,
    shape: &Shape,
    render_state: RenderState,
    transform: InstanceTransform,
    properties: InstanceProperties,
) {
    let key = BatchKey {
        mesh_id: mesh_id_for(mesh_cache, shape),
        blend_mode: BatchRecorder::blend_mode_from_flags(properties.flags),
        render_state,
    };
    let batch = batches
        .entry(key)
        .or_insert_with(|| Batch::new(mesh_cache[shape].1.clone()));
    batch.push(transform, properties);
}

/// One instance to group: what to draw, where, and how.
#[derive(Debug, Clone)]
pub struct InstanceRecord {
    pub shape: Shape,
    pub render_state: RenderState,
    pub transform: InstanceTransform,
    pub properties: InstanceProperties,
}

/// Groups `instances` into batches that share a mesh, blend mode and render state.
///
/// Pure: no ECS, no GPU and no recorder state. The order of the returned batches is
/// unspecified; instances keep their input order within a batch. `BatchRecorder` uses the
/// same grouping rule, so this is the way to test or preview what a recording will batch.
pub fn group_instances(instances: &[InstanceRecord]) -> Vec<(BatchKey, Batch)> {
    let mut batches = HashMap::new();
    let mut mesh_cache = RecorderMeshCache::new();
    for instance in instances {
        place_instance(
            &mut batches,
            &mut mesh_cache,
            &instance.shape,
            instance.render_state,
            instance.transform,
            instance.properties,
        );
    }
    batches.into_iter().collect()
}

// --- Batch Recorder ---

/// Accumulates draw commands and groups them into batches.
///
/// Unlike `ImmediateRecorder` (which spawns one entity per shape), the batch
/// recorder collects instances and renders them in grouped draw calls.
///
/// # Example
///
/// ```ignore
/// let mut recorder = BatchRecorder::new();
/// recorder.begin();
/// recorder.sphere(Vec3::ZERO, 0.5, ShapeDetail::Medium, imm_opaque(1.0, 0.0, 0.0));
/// recorder.sphere(Vec3::X, 0.5, ShapeDetail::Medium, imm_opaque(1.0, 0.0, 0.0));
/// // Both spheres share a batch -> 1 draw call instead of 2.
/// let batches = recorder.end();
/// ```
pub struct BatchRecorder {
    /// The collected batches, keyed by mesh+blend+render state.
    pub batches: HashMap<BatchKey, Batch>,
    /// Cached meshes keyed on the shape itself, each with a recorder-local id used in
    /// [`BatchKey::mesh_id`]. Ids are sequential, so two shapes can never collide.
    mesh_cache: RecorderMeshCache,
    /// Current transform state for relative positioning.
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
    /// Current render state.
    render_state: RenderState,
    /// Pending deformation for the next instance.
    pending_deformation: crate::deformation::DeformationParams,
    /// Pending appearance for the next instance.
    pending_appearance: AppearanceSnapshot,
    /// State stack for push/pop.
    state_stack: Vec<(Vec3, Quat, Vec3)>,
}

impl BatchRecorder {
    pub fn new() -> Self {
        Self {
            batches: HashMap::new(),
            mesh_cache: RecorderMeshCache::new(),
            pending_deformation: crate::deformation::DeformationParams::none(),
            pending_appearance: AppearanceSnapshot::static_appearance(),
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            render_state: RenderState::default(),
            state_stack: Vec::new(),
        }
    }

    /// Start a new recording frame. Clears previous batches.
    pub fn begin(&mut self) {
        for batch in self.batches.values_mut() {
            batch.clear();
        }
        // Remove empty batches
        self.batches.retain(|_, b| !b.is_empty());
        self.position = Vec3::ZERO;
        self.rotation = Quat::IDENTITY;
        self.scale = Vec3::ONE;
        self.render_state = RenderState::default();
        self.state_stack.clear();
    }

    /// Finalize recording and return the batches.
    pub fn end(&mut self) -> &HashMap<BatchKey, Batch> {
        // Remove empty batches
        self.batches.retain(|_, b| !b.is_empty());
        &self.batches
    }

    // --- State management ---

    pub fn push_state(&mut self) {
        self.state_stack
            .push((self.position, self.rotation, self.scale));
    }

    pub fn pop_state(&mut self) {
        if let Some(state) = self.state_stack.pop() {
            self.position = state.0;
            self.rotation = state.1;
            self.scale = state.2;
        }
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    pub fn set_rotation(&mut self, rotation: Quat) {
        self.rotation = rotation;
    }

    pub fn set_scale(&mut self, scale: Vec3) {
        self.scale = scale;
    }

    pub fn set_render_state(&mut self, state: RenderState) {
        self.render_state = state;
    }

    /// Set deformation for the next instance added.
    /// Call this before a drawing method to apply deformation to that instance.
    pub fn set_deformation(&mut self, deformation: crate::deformation::DeformationParams) {
        // Store in a temporary field; applied in add_instance
        self.pending_deformation = deformation;
    }

    /// Set appearance animation for the next instance added.
    /// Call this before a drawing method to apply color animation to that instance.
    pub fn set_appearance(&mut self, appearance: AppearanceSnapshot) {
        self.pending_appearance = appearance;
    }

    // --- Drawing methods ---


    /// Add an instance with the given shape and properties to the current batch.
    /// Used internally by `RetainedWorld` to replay retained slots.
    pub fn add_instance(&mut self, shape: &Shape, mut properties: InstanceProperties) {
        let transform = InstanceTransform::from_trs(self.position, self.rotation, self.scale);

        // Gradients span the instance: carry its half-extent (world units) to the shader in
        // `direction.w`, which is zero for every recorder style and otherwise unused.
        if properties.flags.x > 0.5 {
            let half = (shape.bounds().half_extents() * self.scale.abs()).max_element();
            properties.direction.w = half.max(1e-4);
        }

        // Apply pending deformation to properties
        let deformation = self.pending_deformation;
        properties.deformation = deformation.to_vec4();
        self.pending_deformation = crate::deformation::DeformationParams::none();

        // Apply pending appearance to properties
        let appearance = self.pending_appearance;
        properties.appearance = appearance.to_vec4();
        self.pending_appearance = AppearanceSnapshot::static_appearance();

        place_instance(
            &mut self.batches,
            &mut self.mesh_cache,
            shape,
            self.render_state,
            transform,
            properties,
        );
    }

    fn blend_mode_from_flags(flags: Vec4) -> BlendMode {
        match flags.as_array()[1] as i32 {
            0 => BlendMode::Opaque,
            1 => BlendMode::Transparent,
            2 => BlendMode::Additive,
            3 => BlendMode::Multiplicative,
            4 => BlendMode::Screen,
            5 => BlendMode::Dither,
            6 => BlendMode::Subtractive,
            7 => BlendMode::Lighten,
            8 => BlendMode::ColorDodge,
            9 => BlendMode::LinearBurn,
            10 => BlendMode::ColorBurn,
            11 => BlendMode::Darken,
            _ => BlendMode::Opaque,
        }
    }

    /// Draw a sphere.
    pub fn sphere(&mut self, radius: f32, detail: ShapeDetail, color: [f32; 4]) {
        let shape = Shape::sphere(radius, detail);
        let props = InstanceProperties::from_style(
            color,
            color,
            [0.0, 1.0, 0.0, 0.0],
            ColorMode::Single,
            BlendMode::Opaque,
            MixSpace::default(),
            None,
        );
        self.add_instance(&shape, props);
    }

    /// Draw a cuboid.
    pub fn cuboid(&mut self, size: Vec3, roundness: f32, detail: ShapeDetail, color: [f32; 4]) {
        let shape = Shape::cuboid(size, roundness, detail);
        let props = InstanceProperties::from_style(
            color,
            color,
            [0.0, 1.0, 0.0, 0.0],
            ColorMode::Single,
            BlendMode::Opaque,
            MixSpace::default(),
            None,
        );
        self.add_instance(&shape, props);
    }

    /// Draw a cylinder.
    pub fn cylinder(&mut self, radius: f32, height: f32, detail: ShapeDetail, color: [f32; 4]) {
        let shape = Shape::cylinder(radius, height, detail);
        let props = InstanceProperties::from_style(
            color,
            color,
            [0.0, 1.0, 0.0, 0.0],
            ColorMode::Single,
            BlendMode::Opaque,
            MixSpace::default(),
            None,
        );
        self.add_instance(&shape, props);
    }

    /// Draw a cone.
    pub fn cone(&mut self, radius: f32, height: f32, detail: ShapeDetail, color: [f32; 4]) {
        let shape = Shape::cone(radius, height, detail);
        let props = InstanceProperties::from_style(
            color,
            color,
            [0.0, 1.0, 0.0, 0.0],
            ColorMode::Single,
            BlendMode::Opaque,
            MixSpace::default(),
            None,
        );
        self.add_instance(&shape, props);
    }

    /// Draw a capsule.
    pub fn capsule(&mut self, radius: f32, height: f32, detail: ShapeDetail, color: [f32; 4]) {
        let shape = Shape::capsule(radius, height, detail);
        let props = InstanceProperties::from_style(
            color,
            color,
            [0.0, 1.0, 0.0, 0.0],
            ColorMode::Single,
            BlendMode::Opaque,
            MixSpace::default(),
            None,
        );
        self.add_instance(&shape, props);
    }

    /// Draw a torus.
    pub fn torus(
        &mut self,
        radius: f32,
        thickness: f32,
        start_angle: f32,
        span: f32,
        _detail: ShapeDetail,
        color: [f32; 4],
    ) {
        let shape = Shape::torus(radius, thickness, start_angle, span, _detail);
        let props = InstanceProperties::from_style(
            color,
            color,
            [0.0, 1.0, 0.0, 0.0],
            ColorMode::Single,
            BlendMode::Opaque,
            MixSpace::default(),
            None,
        );
        self.add_instance(&shape, props);
    }

    /// Draw a pyramid.
    pub fn pyramid(
        &mut self,
        sides: u32,
        radius: f32,
        height: f32,
        detail: ShapeDetail,
        color: [f32; 4],
    ) {
        let shape = Shape::pyramid(sides, radius, height, detail);
        let props = InstanceProperties::from_style(
            color,
            color,
            [0.0, 1.0, 0.0, 0.0],
            ColorMode::Single,
            BlendMode::Opaque,
            MixSpace::default(),
            None,
        );
        self.add_instance(&shape, props);
    }

    /// Draw a hemisphere.
    pub fn hemisphere(&mut self, radius: f32, capped: bool, detail: ShapeDetail, color: [f32; 4]) {
        let shape = Shape::hemisphere(radius, capped, detail);
        let props = InstanceProperties::from_style(
            color,
            color,
            [0.0, 1.0, 0.0, 0.0],
            ColorMode::Single,
            BlendMode::Opaque,
            MixSpace::default(),
            None,
        );
        self.add_instance(&shape, props);
    }

    /// Draw an ellipsoid.
    pub fn ellipsoid(&mut self, size: Vec3, detail: ShapeDetail, color: [f32; 4]) {
        let shape = Shape::ellipsoid(size, detail);
        let props = InstanceProperties::from_style(
            color,
            color,
            [0.0, 1.0, 0.0, 0.0],
            ColorMode::Single,
            BlendMode::Opaque,
            MixSpace::default(),
            None,
        );
        self.add_instance(&shape, props);
    }

    /// Draw a regular prism.
    pub fn regular_prism(
        &mut self,
        sides: u32,
        radius: f32,
        height: f32,
        roundness: f32,
        detail: ShapeDetail,
        color: [f32; 4],
    ) {
        let shape = Shape::regular_prism(sides, radius, height, roundness, detail);
        let props = InstanceProperties::from_style(
            color,
            color,
            [0.0, 1.0, 0.0, 0.0],
            ColorMode::Single,
            BlendMode::Opaque,
            MixSpace::default(),
            None,
        );
        self.add_instance(&shape, props);
    }

    // --- Dual-directional variants ---

    /// Draw a sphere with dual-directional gradient shading.
    pub fn sphere_dual_directional(
        &mut self,
        radius: f32,
        detail: ShapeDetail,
        dark: [f32; 3],
        light: [f32; 3],
        direction: [f32; 3],
        space: MixSpace,
    ) {
        let shape = Shape::sphere(radius, detail);
        let dir = Vec3::new(direction[0], direction[1], direction[2]).normalize();
        let props = InstanceProperties::from_style(
            [dark[0], dark[1], dark[2], 1.0],
            [light[0], light[1], light[2], 1.0],
            [dir.x, dir.y, dir.z, 0.0],
            ColorMode::DualDirectional,
            BlendMode::Opaque,
            space,
            None,
        );
        self.add_instance(&shape, props);
    }

    /// Draw a cuboid with dual-directional gradient shading.
    pub fn cuboid_dual_directional(
        &mut self,
        size: Vec3,
        roundness: f32,
        detail: ShapeDetail,
        dark: [f32; 3],
        light: [f32; 3],
        direction: [f32; 3],
        space: MixSpace,
    ) {
        let shape = Shape::cuboid(size, roundness, detail);
        let dir = Vec3::new(direction[0], direction[1], direction[2]).normalize();
        let props = InstanceProperties::from_style(
            [dark[0], dark[1], dark[2], 1.0],
            [light[0], light[1], light[2], 1.0],
            [dir.x, dir.y, dir.z, 0.0],
            ColorMode::DualDirectional,
            BlendMode::Opaque,
            space,
            None,
        );
        self.add_instance(&shape, props);
    }

    // --- Transparent variants ---

    /// Draw a sphere with transparency.
    pub fn sphere_transparent(&mut self, radius: f32, detail: ShapeDetail, color: [f32; 4]) {
        let shape = Shape::sphere(radius, detail);
        let props = InstanceProperties::from_style(
            color,
            color,
            [0.0, 1.0, 0.0, 0.0],
            ColorMode::Single,
            BlendMode::Transparent,
            MixSpace::default(),
            None,
        );
        self.add_instance(&shape, props);
    }

    /// Draw a sphere with additive blending.
    pub fn sphere_additive(&mut self, radius: f32, detail: ShapeDetail, color: [f32; 3]) {
        let shape = Shape::sphere(radius, detail);
        let props = InstanceProperties::from_style(
            [color[0], color[1], color[2], 1.0],
            [color[0], color[1], color[2], 1.0],
            [0.0, 1.0, 0.0, 0.0],
            ColorMode::Single,
            BlendMode::Additive,
            MixSpace::default(),
            None,
        );
        self.add_instance(&shape, props);
    }

    // --- Rendering ---

    #[cfg(feature = "pbr")]
    /// Spawns entities with `InstancedBatchData` components from the collected
    /// batches.  Each batch becomes one entity that the GPU instanced render
    /// pipeline draws with a single `draw_indexed` call.
    ///
    /// Returns the spawned entities so the caller can manage their lifecycle.
    pub fn render_batches(
        &mut self,
        commands: &mut bevy::ecs::system::Commands,
        mut meshes: bevy::ecs::system::ResMut<bevy::asset::Assets<Mesh>>,
        parent: Option<bevy::prelude::Entity>,
    ) -> Vec<bevy::prelude::Entity> {
        use crate::render_pipeline::InstancedBufferData;
        use bevy::prelude::*;

        let mut spawned = Vec::new();

        for (key, batch) in self.batches.iter() {
            if batch.is_empty() {
                continue;
            }

            let mesh_handle = meshes.add(batch.mesh.clone());

            // Build the GPU buffer data for this batch with deformation
            let buffer_data = InstancedBufferData::from_slices_with_deformation(
                &batch.transforms[..batch.count],
                &batch.properties[..batch.count],
                &batch.deformations[..batch.count],
            );

            // Extract custom shader handle from blend mode, if any
            let custom_shader = match &key.blend_mode {
                BlendMode::Custom(handle) => Some(handle.clone()),
                _ => None,
            };

            // Spawn a single entity per batch with InstancedBatchData
            let mesh_id = mesh_handle.id();
            let mesh_aabb = AABB::from_mesh(&batch.mesh);
            // Determine LOD level from the shape detail encoded in the mesh
            let vertex_count = batch.mesh.count_vertices();
            let lod_level = if vertex_count < 50 {
                0
            }
            // Minimal
            else if vertex_count < 100 {
                1
            }
            // Low
            else if vertex_count < 200 {
                2
            }
            // Medium
            else if vertex_count < 400 {
                3
            }
            // High
            else {
                4
            }; // Extreme
            let entity_id = commands
                .spawn((
                    Mesh3d(mesh_handle.clone()),
                    crate::instanced_render::InstancedBatchData {
                        mesh_id,
                        instance_count: buffer_data.instance_count,
                        transforms: buffer_data.transforms,
                        properties: buffer_data.properties,
                        dirty: batch.is_dirty(),
                        render_state: key.render_state,
                        mesh_aabb,
                        lod_level,
                        lod_transition_weight: 0.0,
                        custom_shader,
                    },
                    Transform::default(),
                    GlobalTransform::default(),
                ))
                .id();

            if let Some(p) = parent {
                commands.entity(p).add_child(entity_id);
            }

            spawned.push(entity_id);
        }

        // Acknowledge all batch changes after rendering
        for batch in self.batches.values_mut() {
            batch.acknowledge_changes();
        }

        spawned
    }

    /// Acknowledge all dirty flags without rendering (skip frame).
    pub fn acknowledge_all(&mut self) {
        for batch in self.batches.values_mut() {
            batch.acknowledge_changes();
        }
    }

    // --- Statistics ---

    /// Returns total instance count across all batches.
    pub fn total_instances(&self) -> usize {
        self.batches.values().map(|b| b.count).sum()
    }

    /// Returns the number of batches (draw calls).
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Returns the number of dirty batches that need GPU upload.
    pub fn dirty_batch_count(&self) -> usize {
        self.batches.values().filter(|b| b.is_dirty()).count()
    }
}

impl Default for BatchRecorder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_data_matches_the_vertex_layout_stride() {
        // Both instanced pipelines declare this stride in their vertex layouts.
        const VERTEX_LAYOUT_STRIDE: u64 = 176;
        let gpu_size = <InstanceData as ShaderType>::min_size().get();
        assert_eq!(gpu_size, VERTEX_LAYOUT_STRIDE);
        assert_eq!(
            gpu_size,
            <InstanceTransform as ShaderType>::min_size().get()
                + <InstanceProperties as ShaderType>::min_size().get()
        );
        assert_eq!(std::mem::size_of::<InstanceData>() as u64, gpu_size);
    }

    #[test]
    fn recorder_and_group_instances_agree() {
        let records = grouping_tests::sample_records(&[(0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, 0)]);
        let mut recorder = BatchRecorder::new();
        recorder.begin();
        for r in &records {
            recorder.set_render_state(r.render_state);
            recorder.add_instance(&r.shape, r.properties);
        }
        recorder.end();
        let mut from_recorder: Vec<usize> = recorder.batches.values().map(|b| b.count).collect();
        let mut from_function: Vec<usize> =
            group_instances(&records).iter().map(|(_, b)| b.count).collect();
        from_recorder.sort_unstable();
        from_function.sort_unstable();
        assert_eq!(from_recorder, from_function);
        assert_eq!(from_function, vec![1, 1, 2]);
    }

    #[test]
    fn instance_properties_from_style_keeps_srgb_colors() {
        let style = Style::dual_directional(
            Color::srgb(1.0, 0.5, 0.25),
            Color::srgb(0.0, 0.5, 1.0),
            Vec3::X,
        );
        let props = InstanceProperties::from(&style);
        assert!((props.color.x - 1.0).abs() < 1e-4);
        assert!((props.color.y - 0.5).abs() < 1e-4);
        assert!((props.color.z - 0.25).abs() < 1e-4);
        assert!((props.secondary_color.z - 1.0).abs() < 1e-4);
        assert_eq!(props.flags.x, 1.0, "dual directional colour mode");
        assert_eq!(props.direction.x, 1.0);
        assert_eq!(
            InstanceProperties::from_style_descriptor(&style).color.y,
            props.color.y
        );
    }

    #[test]
    fn batch_key_equality() {
        let key1 = BatchKey {
            mesh_id: 42,
            blend_mode: BlendMode::Opaque,
            render_state: RenderState::default(),
        };
        let key2 = BatchKey {
            mesh_id: 42,
            blend_mode: BlendMode::Opaque,
            render_state: RenderState::default(),
        };
        assert_eq!(key1, key2);
    }

    #[test]
    fn batch_key_differs_by_blend_mode() {
        let key1 = BatchKey {
            mesh_id: 42,
            blend_mode: BlendMode::Opaque,
            render_state: RenderState::default(),
        };
        let key2 = BatchKey {
            mesh_id: 42,
            blend_mode: BlendMode::Transparent,
            render_state: RenderState::default(),
        };
        assert_ne!(key1, key2);
    }

    #[test]
    fn batch_grows() {
        let mut batch = Batch::new(Mesh::new(
            PrimitiveTopology::TriangleList,
            bevy_asset::RenderAssetUsages::RENDER_WORLD,
        ));
        for _ in 0..512 {
            batch.push(InstanceTransform::default(), InstanceProperties::default());
        }
        assert_eq!(batch.count, 512);
    }

    #[test]
    fn batch_clear() {
        let mut batch = Batch::new(Mesh::new(
            PrimitiveTopology::TriangleList,
            bevy_asset::RenderAssetUsages::RENDER_WORLD,
        ));
        batch.push(InstanceTransform::default(), InstanceProperties::default());
        batch.push(InstanceTransform::default(), InstanceProperties::default());
        batch.clear();
        assert!(batch.is_empty());
    }

    #[test]
    fn recorder_groups_same_shape() {
        let mut rec = BatchRecorder::new();
        rec.begin();
        rec.set_position(Vec3::ZERO);
        rec.sphere(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);
        rec.set_position(Vec3::X);
        rec.sphere(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);
        rec.set_position(Vec3::Y);
        rec.sphere(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);
        let batches = rec.end();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches.values().next().unwrap().count, 3);
    }

    #[test]
    fn recorder_separates_different_blend_modes() {
        let mut rec = BatchRecorder::new();
        rec.begin();
        rec.set_position(Vec3::ZERO);
        rec.sphere(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);
        rec.set_position(Vec3::X);
        rec.sphere_transparent(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 0.5]);
        let batches = rec.end();
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn instance_transform_from_trs() {
        let t = InstanceTransform::from_trs(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, Vec3::ONE);
        // Translation should be in col3 (column-major)
        let arr = t.col3.as_array();
        assert!((arr[0] - 1.0).abs() < 1e-5);
        assert!((arr[1] - 2.0).abs() < 1e-5);
        assert!((arr[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn recorder_push_pop_state() {
        let mut rec = BatchRecorder::new();
        rec.begin();
        rec.set_position(Vec3::new(1.0, 0.0, 0.0));
        rec.push_state();
        rec.set_position(Vec3::new(2.0, 0.0, 0.0));
        rec.pop_state();
        assert_eq!(rec.position, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn batch_dirty_tracking_push() {
        let mut batch = Batch::new(Mesh::new(
            PrimitiveTopology::TriangleList,
            bevy_asset::RenderAssetUsages::RENDER_WORLD,
        ));
        assert!(!batch.is_dirty());
        batch.push(InstanceTransform::default(), InstanceProperties::default());
        assert!(batch.is_dirty());
        assert!(batch.transform_pending);
        assert!(batch.property_pending);
        assert_eq!(batch.transform_revision, 1);
        assert_eq!(batch.property_revision, 1);
    }

    #[test]
    fn batch_dirty_tracking_acknowledge() {
        let mut batch = Batch::new(Mesh::new(
            PrimitiveTopology::TriangleList,
            bevy_asset::RenderAssetUsages::RENDER_WORLD,
        ));
        batch.push(InstanceTransform::default(), InstanceProperties::default());
        assert!(batch.is_dirty());
        batch.acknowledge_changes();
        assert!(!batch.is_dirty());
        assert!(!batch.transform_pending);
        assert!(!batch.property_pending);
        // Revisions should not reset
        assert_eq!(batch.transform_revision, 1);
        assert_eq!(batch.property_revision, 1);
    }

    #[test]
    fn batch_dirty_tracking_set_transform() {
        let mut batch = Batch::new(Mesh::new(
            PrimitiveTopology::TriangleList,
            bevy_asset::RenderAssetUsages::RENDER_WORLD,
        ));
        batch.push(InstanceTransform::default(), InstanceProperties::default());
        batch.acknowledge_changes();
        assert!(!batch.is_dirty());
        batch.set_transform(0, InstanceTransform::default());
        assert!(batch.transform_pending);
        assert!(!batch.property_pending);
        assert_eq!(batch.transform_revision, 2);
    }

    #[test]
    fn batch_dirty_tracking_set_properties() {
        let mut batch = Batch::new(Mesh::new(
            PrimitiveTopology::TriangleList,
            bevy_asset::RenderAssetUsages::RENDER_WORLD,
        ));
        batch.push(InstanceTransform::default(), InstanceProperties::default());
        batch.acknowledge_changes();
        assert!(!batch.is_dirty());
        batch.set_properties(0, InstanceProperties::default());
        assert!(!batch.transform_pending);
        assert!(batch.property_pending);
        assert_eq!(batch.property_revision, 2);
    }

    #[test]
    fn batch_dirty_tracking_clear() {
        let mut batch = Batch::new(Mesh::new(
            PrimitiveTopology::TriangleList,
            bevy_asset::RenderAssetUsages::RENDER_WORLD,
        ));
        batch.push(InstanceTransform::default(), InstanceProperties::default());
        batch.acknowledge_changes();
        batch.clear();
        assert!(batch.is_dirty());
        assert!(batch.transform_pending);
        assert!(batch.property_pending);
    }

    #[test]
    fn recorder_dirty_batch_count() {
        let mut rec = BatchRecorder::new();
        rec.begin();
        rec.set_position(Vec3::ZERO);
        rec.sphere(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(rec.dirty_batch_count(), 1);
        for batch in rec.batches.values_mut() {
            batch.acknowledge_changes();
        }
        assert_eq!(rec.dirty_batch_count(), 0);
    }

    #[test]
    fn recorder_acknowledge_all() {
        let mut rec = BatchRecorder::new();
        rec.begin();
        rec.set_position(Vec3::ZERO);
        rec.sphere(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 1.0]);
        rec.set_position(Vec3::X);
        rec.sphere_transparent(0.5, ShapeDetail::Medium, [1.0, 0.0, 0.0, 0.5]);
        assert_eq!(rec.dirty_batch_count(), 2);
        rec.acknowledge_all();
        assert_eq!(rec.dirty_batch_count(), 0);
    }

    use bevy::render::mesh::PrimitiveTopology;
}

#[cfg(test)]
mod grouping_tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashSet;

    /// Distinct blend modes, indexed by the second element of a sample tuple.
    const BLENDS: [BlendMode; 3] = [
        BlendMode::Opaque,
        BlendMode::Transparent,
        BlendMode::Additive,
    ];

    /// Builds one record per `(shape, blend, layer)` tuple. Equal tuples are instances that
    /// must share a batch; each record sits at `x = its position in the input`.
    pub(super) fn sample_records(samples: &[(usize, usize, i32)]) -> Vec<InstanceRecord> {
        samples
            .iter()
            .enumerate()
            .map(|(i, &(shape, blend, layer))| InstanceRecord {
                shape: Shape::sphere(0.5 + shape as f32, ShapeDetail::Low),
                render_state: RenderState {
                    layer,
                    ..RenderState::default()
                },
                transform: InstanceTransform::from_trs(
                    Vec3::new(i as f32, 0.0, 0.0),
                    Quat::IDENTITY,
                    Vec3::ONE,
                ),
                properties: InstanceProperties::from_style(
                    [1.0; 4],
                    [1.0; 4],
                    [0.0, 1.0, 0.0, 0.0],
                    ColorMode::Single,
                    BLENDS[blend].clone(),
                    MixSpace::default(),
                    None,
                ),
            })
            .collect()
    }

    #[test]
    fn no_instances_make_no_batches() {
        assert!(group_instances(&[]).is_empty());
    }

    #[test]
    fn one_instance_makes_one_batch() {
        let batches = group_instances(&sample_records(&[(0, 0, 0)]));
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].1.count, 1);
    }

    #[test]
    fn identical_instances_share_one_batch_in_input_order() {
        let batches = group_instances(&sample_records(&[(0, 0, 0); 40]));
        assert_eq!(batches.len(), 1);
        let batch = &batches[0].1;
        assert_eq!(batch.count, 40);
        for (i, transform) in batch.transforms.iter().enumerate() {
            assert_eq!(transform.col3.x, i as f32);
        }
    }

    #[test]
    fn instances_that_differ_in_shape_blend_or_layer_never_share_a_batch() {
        let samples = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 2, -1)];
        let batches = group_instances(&sample_records(&samples));
        assert_eq!(batches.len(), samples.len());
        assert!(batches.iter().all(|(_, b)| b.count == 1));
    }

    proptest! {
        #[test]
        fn grouping_matches_the_distinct_keys(
            samples in prop::collection::vec((0usize..3, 0usize..3, 0i32..3), 0..60)
        ) {
            let batches = group_instances(&sample_records(&samples));

            let mut expected: Vec<usize> = {
                let distinct: HashSet<_> = samples.iter().copied().collect();
                distinct
                    .iter()
                    .map(|key| samples.iter().filter(|s| *s == key).count())
                    .collect()
            };
            let mut actual: Vec<usize> = batches.iter().map(|(_, b)| b.count).collect();
            expected.sort_unstable();
            actual.sort_unstable();
            prop_assert_eq!(actual, expected);

            for (key, batch) in &batches {
                prop_assert_eq!(batch.transforms.len(), batch.count);
                prop_assert_eq!(batch.properties.len(), batch.count);
                prop_assert_eq!(batch.deformations.len(), batch.count);
                let layer = key.render_state.layer;
                prop_assert!((0..3).contains(&layer));
                prop_assert!(BLENDS.contains(&key.blend_mode));
                // Within a batch, instances keep the order they were given in.
                let xs: Vec<f32> = batch.transforms.iter().map(|t| t.col3.x).collect();
                prop_assert!(xs.windows(2).all(|w| w[0] < w[1]));
            }
        }
    }
}
