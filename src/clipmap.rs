//! GPU geometry-clipmap terrain (feature `clipmap`).
//!
//! A [`Clipmap`] draws an endless heightfield around one camera with a fixed
//! amount of geometry. The land is drawn as `levels` square rings of the same
//! grid mesh; every ring has twice the cell size of the one inside it, so the
//! detail drops with distance by itself. Nothing about it is chunked, and
//! nothing has to be rebuilt when the camera moves.
//!
//! # How it works
//!
//! - Heights live in a texture array with one layer per ring. The layers are
//!   *toroidal*: when the camera moves, only the strips of texels that came
//!   into view are recomputed from your height source and written to the GPU.
//! - The vertex shader reads its height from the texture. Near the outer edge
//!   of a ring it blends the height and normal towards what the next coarser
//!   ring has at that spot (*geomorphing*), and the blend is complete on the
//!   edge itself, so rings meet without cracks, steps or popping.
//! - Holes are a flag next to each height. A hole is discarded per pixel, so it
//!   is see-through in the depth and shadow passes too, and it has no skirt.
//!
//! # Using it
//!
//! ```ignore
//! app.add_plugins(ClipmapPlugin);
//!
//! commands.spawn(Clipmap::new(
//!     camera,
//!     HeightSource::heights(|p| 8.0 * (p.x * 0.05).sin() * (p.y * 0.05).cos()),
//!     ClipmapSettings::default(),
//! ));
//! ```
//!
//! # One camera per clipmap
//!
//! A clipmap follows exactly one camera, the one given to [`Clipmap::new`]: its
//! rings are centred on that camera and its detail is chosen by the distance to
//! it. Bevy draws every entity in every camera, so with two cameras give each
//! its own `Clipmap`, and put each camera and its clipmap on separate
//! `RenderLayers`; a `RenderLayers` on the clipmap entity is copied to its
//! rings.
//!
//! # Height sources
//!
//! This release reads heights from a callback ([`HeightSource::callback`] and
//! [`HeightSource::heights`]). The callback gets a position in the terrain's
//! local space (`x`, `z` as a [`Vec2`]) and returns the height there, or `None`
//! for a hole. It is called on the CPU for each texel that scrolls into view,
//! so it must be cheap and must return the same answer for the same position
//! (call [`Clipmap::refresh`] after changing what it returns). [`HeightSource`]
//! is `#[non_exhaustive]`: reading heights from an image will be added as
//! another variant.
//!
//! # Holes
//!
//! A cell with a hole at any of its four corners is not drawn, in every pass,
//! so a hole is see-through and casts no shadow. The edge of a hole follows the
//! grid cells of the ring it is in, so on a coarse ring (far from the camera) a
//! small hole is a coarse stair-step and a hole smaller than one cell of that
//! ring may not show at all. Use [`Clipmap::height_at`] for physics: it is
//! exact at every distance.
//!
//! # Limits that cannot be got wrong
//!
//! - Settings are sanitised when they are built: levels `1..=8`, a ring of at
//!   least 16 cells and a multiple of 4, a positive cell size.
//! - Only the translation, rotation and scale of the entity's transform apply;
//!   the rings move with the camera inside it.
//! - Keep the terrain within about 100 000 units of the origin: positions are
//!   single precision.

use std::sync::Arc;

use bevy::app::{App, Plugin, PostUpdate};
use bevy::asset::{Asset, AssetId, Assets, Handle, RenderAssetUsages, embedded_asset};
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers, Visibility};
use bevy::color::{Color, LinearRgba};
use bevy::ecs::prelude::*;
use bevy::image::Image;
use bevy::math::{IVec2, Mat4, UVec2, Vec2};
use bevy::mesh::{Indices, Mesh, Mesh3d, MeshTag, PrimitiveTopology};
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialPlugin, MeshMaterial3d, StandardMaterial,
};
use bevy::reflect::Reflect;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    AsBindGroup, Extent3d, Origin3d, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::renderer::RenderQueue;
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::shader::ShaderRef;
use bevy::transform::components::{GlobalTransform, Transform};
use bevy::transform::TransformSystems;

use crate::material::Vec4;

/// The most rings a clipmap draws.
pub const MAX_CLIPMAP_LEVELS: usize = 8;

/// Fixed number of height bands a clipmap can colour by.
pub const MAX_CLIPMAP_BANDS: usize = 4;

// ─── Height source ───

type HeightFn = dyn Fn(Vec2) -> Option<f32> + Send + Sync;

/// Where a [`Clipmap`] gets its heights from.
///
/// `#[non_exhaustive]`: more sources (an image) will be added.
#[derive(Clone)]
#[non_exhaustive]
pub enum HeightSource {
    /// A function of the position in the terrain's local space (`x`, `z`);
    /// `None` is a hole.
    Callback(Arc<HeightFn>),
}

impl HeightSource {
    /// A height function that can also say where there are holes.
    pub fn callback(f: impl Fn(Vec2) -> Option<f32> + Send + Sync + 'static) -> Self {
        Self::Callback(Arc::new(f))
    }

    /// A height function without holes.
    pub fn heights(f: impl Fn(Vec2) -> f32 + Send + Sync + 'static) -> Self {
        Self::Callback(Arc::new(move |p| Some(f(p))))
    }

    /// The height at `p`, or `None` for a hole or a non-finite height.
    pub fn sample(&self, p: Vec2) -> Option<f32> {
        match self {
            Self::Callback(f) => f(p).filter(|h| h.is_finite()),
        }
    }
}

impl std::fmt::Debug for HeightSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Callback(_) => f.write_str("HeightSource::Callback(..)"),
        }
    }
}

// ─── Settings ───

/// Shape and look of a [`Clipmap`]. Out-of-range values are brought into range
/// when they are set, so every value a `ClipmapSettings` holds works.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ClipmapSettings {
    levels: usize,
    grid: u32,
    cell_size: f32,
    morph: f32,
    base_color: Color,
    bands: Vec<(f32, Color)>,
}

impl Default for ClipmapSettings {
    /// Six rings of 64 cells, one unit per cell on the finest ring: detail of
    /// one unit out to 32, then halving every ring, to a view distance of 1024.
    fn default() -> Self {
        Self {
            levels: 6,
            grid: 64,
            cell_size: 1.0,
            morph: 0.3,
            base_color: Color::srgb(0.45, 0.5, 0.35),
            bands: Vec::new(),
        }
    }
}

impl ClipmapSettings {
    /// Number of rings, `1..=`[`MAX_CLIPMAP_LEVELS`].
    #[must_use]
    pub fn with_levels(mut self, levels: usize) -> Self {
        self.levels = levels.clamp(1, MAX_CLIPMAP_LEVELS);
        self
    }

    /// Cells along one side of a ring. Rounded up to a multiple of 4 and kept
    /// within `16..=256`: the ring edges must fall on the next ring's grid, and
    /// below 16 cells a ring has no room to blend.
    #[must_use]
    pub fn with_grid(mut self, cells: u32) -> Self {
        self.grid = (cells.clamp(16, 256) + 3) / 4 * 4;
        self
    }

    /// Size of one cell on the finest ring, in the terrain's local units.
    /// A value that is not positive and finite gives the default, `1.0`.
    #[must_use]
    pub fn with_cell_size(mut self, cell_size: f32) -> Self {
        self.cell_size = if cell_size.is_finite() && cell_size > 0.0 {
            cell_size
        } else {
            1.0
        };
        self
    }

    /// Fraction of each ring, measured inward from its outer edge, over which
    /// it blends into the next coarser ring. Kept within `0.05..=0.9`: a
    /// narrower band would show the change, a wider one would never settle.
    #[must_use]
    pub fn with_morph(mut self, fraction: f32) -> Self {
        self.morph = if fraction.is_finite() {
            fraction.clamp(0.05, 0.9)
        } else {
            0.3
        };
        self
    }

    /// The colour of the land when there are no height bands.
    #[must_use]
    pub fn with_base_color(mut self, color: Color) -> Self {
        self.base_color = color;
        self
    }

    /// Colours the land by height. Each pair is `(height, colour)`; a colour
    /// holds from its height up to the next one, and the lowest one holds
    /// below everything. Heights are in the terrain's local space.
    ///
    /// At most [`MAX_CLIPMAP_BANDS`] bands: more are dropped from the top and a
    /// warning is logged. They are sorted by height. Bands are absolute, not
    /// fractions of the terrain's height as in
    /// [`Style::with_height_bands_on`](crate::Style::with_height_bands_on),
    /// because a clipmap has no bounds to take a fraction of.
    #[must_use]
    pub fn with_height_bands(mut self, bands: &[(f32, Color)]) -> Self {
        let mut bands: Vec<(f32, Color)> =
            bands.iter().copied().filter(|(h, _)| h.is_finite()).collect();
        bands.sort_by(|a, b| a.0.total_cmp(&b.0));
        if bands.len() > MAX_CLIPMAP_BANDS {
            bevy::log::warn!(
                "a clipmap colours by at most {MAX_CLIPMAP_BANDS} height bands; dropping the {} highest",
                bands.len() - MAX_CLIPMAP_BANDS
            );
            bands.truncate(MAX_CLIPMAP_BANDS);
        }
        self.bands = bands;
        self
    }

    /// Number of rings.
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// Cells along one side of a ring.
    pub fn grid(&self) -> u32 {
        self.grid
    }

    /// Size of one cell on the finest ring.
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Distance from the camera to the edge of the outermost ring.
    pub fn view_distance(&self) -> f32 {
        self.grid as f32 * 0.5 * self.cell_size * (1u32 << (self.levels - 1)) as f32
    }

    /// Texels along one side of a level's height texture: the ring plus two
    /// texels of margin on every side for normals and blending.
    pub(crate) fn texture_size(&self) -> u32 {
        self.grid + 4
    }
}

// ─── The component ───

/// An endless heightfield that follows one camera. See the [module
/// documentation](self).
///
/// Spawn it with a `Transform` to place the terrain; the plugin adds the rings
/// as children. The settings are fixed once the clipmap is spawned: despawn it
/// and spawn another to change them.
#[derive(Component, Clone, Debug)]
#[require(Transform, Visibility)]
pub struct Clipmap {
    camera: Entity,
    source: HeightSource,
    settings: ClipmapSettings,
    refresh: bool,
}

impl Clipmap {
    /// A clipmap that follows `camera`.
    pub fn new(camera: Entity, source: HeightSource, settings: ClipmapSettings) -> Self {
        Self {
            camera,
            source,
            settings,
            refresh: false,
        }
    }

    /// The camera it follows.
    pub fn camera(&self) -> Entity {
        self.camera
    }

    /// Its settings.
    pub fn settings(&self) -> &ClipmapSettings {
        &self.settings
    }

    /// Its height source.
    pub fn source(&self) -> &HeightSource {
        &self.source
    }

    /// Replaces the height source and recomputes every ring.
    pub fn set_source(&mut self, source: HeightSource) {
        self.source = source;
        self.refresh = true;
    }

    /// Recomputes every ring from the height source: call it after changing
    /// what the source returns. Costs one callback per texel of every ring.
    pub fn refresh(&mut self) {
        self.refresh = true;
    }

    /// The height at `p` in the terrain's local space, or `None` at a hole.
    ///
    /// This is the exact value of the source. Far from the camera the screen
    /// shows a coarser resample of it, so use this, not the mesh, for
    /// physics.
    pub fn height_at(&self, p: Vec2) -> Option<f32> {
        self.source.sample(p)
    }
}

// ─── Scrolling arithmetic (pure, so it can be tested without a GPU) ───

/// Centre of `level`'s window, in that level's texels: the camera rounded to
/// a multiple of two texels, so that the centre of every finer ring is a grid
/// point of the ring around it.
pub(crate) fn level_center(camera: Vec2, cell_size: f32, level: usize) -> IVec2 {
    let texel = cell_size * (1u32 << level) as f32;
    ((camera / (2.0 * texel)).round() * 2.0).as_ivec2()
}

/// Smallest corner of the `size`-texel window centred on `center`.
fn window_min(center: IVec2, size: i32) -> IVec2 {
    center - IVec2::splat(size / 2)
}

/// Rectangles `(min, size)` of the texels that are in the window around `new`
/// and were not in the one around `old`. A first fill (`old` is `None`) or a
/// jump of a whole window is the whole window.
pub(crate) fn refill_rects(old: Option<IVec2>, new: IVec2, size: i32) -> Vec<(IVec2, IVec2)> {
    let n_min = window_min(new, size);
    let Some(old) = old else {
        return vec![(n_min, IVec2::splat(size))];
    };
    let d = new - old;
    if d == IVec2::ZERO {
        return Vec::new();
    }
    if d.x.abs() >= size || d.y.abs() >= size {
        return vec![(n_min, IVec2::splat(size))];
    }
    let o_min = window_min(old, size);
    let mut rects = Vec::new();
    // Columns that came into view, over the full height of the new window.
    if d.x != 0 {
        let x = if d.x > 0 { o_min.x + size } else { n_min.x };
        rects.push((IVec2::new(x, n_min.y), IVec2::new(d.x.abs(), size)));
    }
    // Rows that came into view, over the columns both windows share.
    if d.y != 0 {
        let y = if d.y > 0 { o_min.y + size } else { n_min.y };
        rects.push((
            IVec2::new(n_min.x.max(o_min.x), y),
            IVec2::new(size - d.x.abs(), d.y.abs()),
        ));
    }
    rects
}

/// A piece of a rectangle of texels after wrapping it into the texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Piece {
    /// Where the piece starts in the texture.
    pub origin: UVec2,
    /// Its size in texels.
    pub size: UVec2,
    /// The (unwrapped) texel index of its first texel.
    pub texel_min: IVec2,
}

/// Splits a rectangle of texel indices at the seams of the toroidal texture.
pub(crate) fn wrap_pieces(min: IVec2, size: IVec2, texture: i32) -> Vec<Piece> {
    let mut pieces = Vec::new();
    let mut y = min.y;
    let mut rows = size.y;
    while rows > 0 {
        let ty = y.rem_euclid(texture);
        let h = rows.min(texture - ty);
        let mut x = min.x;
        let mut cols = size.x;
        while cols > 0 {
            let tx = x.rem_euclid(texture);
            let w = cols.min(texture - tx);
            pieces.push(Piece {
                origin: UVec2::new(tx as u32, ty as u32),
                size: UVec2::new(w as u32, h as u32),
                texel_min: IVec2::new(x, y),
            });
            x += w;
            cols -= w;
        }
        y += h;
        rows -= h;
    }
    pieces
}

/// `(height, hole)` of one texel: a hole is `(0, 1)`.
fn texel_value(source: &HeightSource, texel: IVec2, texel_size: f32) -> [f32; 2] {
    match source.sample(texel.as_vec2() * texel_size) {
        Some(h) => [h, 0.0],
        None => [0.0, 1.0],
    }
}

/// Computes the texels of one piece, row by row, as interleaved `(height, hole)`.
fn fill_piece(source: &HeightSource, piece: &Piece, texel_size: f32) -> Vec<f32> {
    let mut texels = Vec::with_capacity((piece.size.x * piece.size.y * 2) as usize);
    for j in 0..piece.size.y as i32 {
        for i in 0..piece.size.x as i32 {
            texels.extend(texel_value(source, piece.texel_min + IVec2::new(i, j), texel_size));
        }
    }
    texels
}

// ─── The grid mesh ───

/// The one square grid every ring is drawn with: vertices at the integer
/// points `-grid/2..=grid/2` of the xz plane, at height 0. The shader moves
/// them; the NORMAL, UV_0 and COLOR attributes are placeholders it overwrites,
/// present because the shader sends data to the fragment stage through them.
pub(crate) fn ring_mesh(grid: u32) -> Mesh {
    let half = (grid / 2) as i32;
    let side = grid + 1;
    let mut positions = Vec::with_capacity((side * side) as usize);
    for z in -half..=half {
        for x in -half..=half {
            positions.push([x as f32, 0.0, z as f32]);
        }
    }
    let mut indices = Vec::with_capacity((grid * grid * 6) as usize);
    for z in 0..grid {
        for x in 0..grid {
            let a = z * side + x;
            let b = a + 1;
            let c = a + side;
            let d = c + 1;
            // Counter-clockwise seen from above (+y).
            indices.extend([a, c, b, b, c, d]);
        }
    }
    let count = positions.len();
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0f32, 1.0, 0.0]; count]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0f32, 0.0]; count]);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, vec![[1.0f32, 1.0, 1.0, 1.0]; count]);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

// ─── Material ───

/// The uniform block of the clipmap shader; the layout mirrors
/// `ClipmapUniform` in `shader/clipmap.wesl`.
#[derive(Clone, Copy, Debug, Default, Reflect, bevy::render::render_resource::ShaderType)]
pub struct ClipmapUniform {
    local_from_world: Mat4,
    params: Vec4,
    morph: Vec4,
    levels: [Vec4; MAX_CLIPMAP_LEVELS],
    bands: [Vec4; MAX_CLIPMAP_BANDS],
    band_heights: Vec4,
    band_info: Vec4,
}

impl ClipmapUniform {
    pub(crate) fn new(settings: &ClipmapSettings) -> Self {
        let mut uniform = Self {
            local_from_world: Mat4::IDENTITY,
            params: Vec4::new(
                settings.cell_size,
                settings.grid as f32,
                settings.texture_size() as f32,
                settings.levels as f32,
            ),
            morph: Vec4::new(settings.morph, 0.0, 0.0, 0.0),
            ..Self::default()
        };
        let count = settings.bands.len();
        let mut heights = [f32::MAX; 4];
        for slot in 0..MAX_CLIPMAP_BANDS {
            // Unused slots repeat the last band; the first band holds below all.
            let (h, color) = settings.bands[slot.min(count.saturating_sub(1))..]
                .first()
                .copied()
                .unwrap_or((0.0, Color::WHITE));
            let linear: LinearRgba = color.to_linear();
            uniform.bands[slot] = Vec4::new(linear.red, linear.green, linear.blue, linear.alpha);
            if slot < count {
                heights[slot] = h;
            }
        }
        uniform.band_heights = Vec4::new(heights[0], heights[1], heights[2], heights[3]);
        uniform.band_info = Vec4::new(count as f32, 0.0, 0.0, 0.0);
        uniform
    }

    /// Records where each level's window is and where the camera is in it.
    pub(crate) fn set_view(
        &mut self,
        settings: &ClipmapSettings,
        camera: Vec2,
        centers: &[IVec2],
        local_from_world: Mat4,
    ) {
        self.local_from_world = local_from_world;
        for (level, center) in centers.iter().enumerate() {
            let texel = settings.cell_size * (1u32 << level) as f32;
            let rel = camera / texel - center.as_vec2();
            self.levels[level] = Vec4::new(center.x as f32, center.y as f32, rel.x, rel.y);
        }
    }
}

/// Extension of [`StandardMaterial`] that draws a clipmap.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct ClipmapExtension {
    #[uniform(100)]
    uniforms: ClipmapUniform,
    #[texture(101, dimension = "2d_array", sample_type = "float", filterable = false)]
    heights: Handle<Image>,
}

fn clipmap_shader() -> ShaderRef {
    ShaderRef::Path(
        bevy::asset::AssetPath::from_path_buf(bevy::asset::embedded_path!("shader/clipmap.wesl"))
            .with_source("embedded"),
    )
}

impl MaterialExtension for ClipmapExtension {
    fn vertex_shader() -> ShaderRef {
        clipmap_shader()
    }
    fn fragment_shader() -> ShaderRef {
        clipmap_shader()
    }
    fn prepass_vertex_shader() -> ShaderRef {
        clipmap_shader()
    }
    fn prepass_fragment_shader() -> ShaderRef {
        clipmap_shader()
    }
    fn deferred_vertex_shader() -> ShaderRef {
        clipmap_shader()
    }
    fn deferred_fragment_shader() -> ShaderRef {
        clipmap_shader()
    }
}

/// The material every ring of a clipmap is drawn with.
pub type ClipmapMaterial = ExtendedMaterial<StandardMaterial, ClipmapExtension>;

// ─── Uploads ───

/// One rectangle of one layer of a height texture, to write on the GPU.
#[derive(Clone, Debug)]
pub struct TextureUpload {
    image: AssetId<Image>,
    layer: u32,
    origin: UVec2,
    size: UVec2,
    texels: Vec<f32>,
}

/// The texture writes of this frame, carried to the render world with the
/// frame's camera position, so that the two always agree.
#[derive(Component, Clone, Debug, Default, ExtractComponent)]
#[extract_app(RenderApp)]
pub struct ClipmapUploads(Vec<TextureUpload>);

fn apply_uploads(
    queue: Res<RenderQueue>,
    images: Res<RenderAssets<GpuImage>>,
    batches: Query<&ClipmapUploads>,
    mut waiting: Local<Vec<TextureUpload>>,
) {
    for batch in &batches {
        waiting.extend(batch.0.iter().cloned());
    }
    // A write whose texture is not on the GPU yet waits for the next frame.
    waiting.retain(|upload| {
        let Some(gpu) = images.get(upload.image) else {
            return true;
        };
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &gpu.texture,
                mip_level: 0,
                origin: Origin3d {
                    x: upload.origin.x,
                    y: upload.origin.y,
                    z: upload.layer,
                },
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(&upload.texels),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(upload.size.x * 8),
                rows_per_image: Some(upload.size.y),
            },
            Extent3d {
                width: upload.size.x,
                height: upload.size.y,
                depth_or_array_layers: 1,
            },
        );
        false
    });
}

// ─── Systems ───

/// What the plugin keeps for each clipmap.
#[derive(Component)]
struct ClipmapState {
    material: Handle<ClipmapMaterial>,
    image: Handle<Image>,
    /// The centre of each level's window when its texels were last filled.
    centers: Vec<Option<IVec2>>,
}

fn spawn_clipmaps(
    mut commands: Commands,
    new: Query<(Entity, &Clipmap, Option<&RenderLayers>), Without<ClipmapState>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<ClipmapMaterial>>,
) {
    for (entity, clipmap, layers) in &new {
        let settings = &clipmap.settings;
        let size = settings.texture_size();
        let mut image = Image::new_uninit(
            Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: settings.levels as u32,
            },
            TextureDimension::D2,
            TextureFormat::Rg32Float,
            RenderAssetUsages::RENDER_WORLD,
        );
        image.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        });
        let image = images.add(image);
        let material = materials.add(ClipmapMaterial {
            base: StandardMaterial {
                base_color: settings.base_color,
                perceptual_roughness: 0.9,
                cull_mode: None,
                // Makes Bevy run this fragment shader in the depth and shadow
                // passes, which is what lets holes be see-through there. The
                // colour is opaque, so the cutoff itself never discards.
                alpha_mode: bevy_material::AlphaMode::Mask(0.5),
                ..Default::default()
            },
            extension: ClipmapExtension {
                uniforms: ClipmapUniform::new(settings),
                heights: image.clone(),
            },
        });
        let mesh = meshes.add(ring_mesh(settings.grid));
        commands.entity(entity).with_children(|parent| {
            for level in 0..settings.levels {
                let mut ring = parent.spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(material.clone()),
                    MeshTag::new(level as u32),
                    Transform::default(),
                    Visibility::default(),
                    NoFrustumCulling,
                ));
                if let Some(layers) = layers {
                    ring.insert(layers.clone());
                }
            }
        });
        commands.entity(entity).insert((
            ClipmapState {
                material,
                image,
                centers: vec![None; settings.levels],
            },
            ClipmapUploads::default(),
        ));
    }
}

fn update_clipmaps(
    mut terrains: Query<(
        &mut Clipmap,
        &mut ClipmapState,
        &mut ClipmapUploads,
        &GlobalTransform,
    )>,
    cameras: Query<&GlobalTransform>,
    mut materials: ResMut<Assets<ClipmapMaterial>>,
) {
    for (mut clipmap, mut state, mut uploads, terrain) in &mut terrains {
        uploads.0.clear();
        let Ok(camera) = cameras.get(clipmap.camera) else {
            continue;
        };
        if clipmap.refresh {
            clipmap.refresh = false;
            state.centers.iter_mut().for_each(|c| *c = None);
        }
        let local = terrain.affine().inverse().transform_point3(camera.translation());
        let camera_xz = Vec2::new(local.x, local.z);
        let settings = clipmap.settings.clone();
        let size = settings.texture_size() as i32;

        let mut centers = Vec::with_capacity(settings.levels);
        for level in 0..settings.levels {
            let center = level_center(camera_xz, settings.cell_size, level);
            let texel_size = settings.cell_size * (1u32 << level) as f32;
            for (min, extent) in refill_rects(state.centers[level], center, size) {
                for piece in wrap_pieces(min, extent, size) {
                    uploads.0.push(TextureUpload {
                        image: state.image.id(),
                        layer: level as u32,
                        origin: piece.origin,
                        size: piece.size,
                        texels: fill_piece(&clipmap.source, &piece, texel_size),
                    });
                }
            }
            state.centers[level] = Some(center);
            centers.push(center);
        }

        if let Some(mut material) = materials.get_mut(&state.material) {
            material.extension.uniforms.set_view(
                &settings,
                camera_xz,
                &centers,
                terrain.to_matrix().inverse(),
            );
        }
    }
}

/// Adds [`Clipmap`] terrain. Needs the render plugins (`DefaultPlugins`).
#[derive(Default, Clone, Copy, Debug)]
pub struct ClipmapPlugin;

impl Plugin for ClipmapPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shader/clipmap.wesl");
        app.add_plugins((
            MaterialPlugin::<ClipmapMaterial>::default(),
            ExtractComponentPlugin::<ClipmapUploads>::default(),
        ))
        .add_systems(
            PostUpdate,
            (
                spawn_clipmaps.before(TransformSystems::Propagate),
                update_clipmaps.after(TransformSystems::Propagate),
            ),
        );
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render.add_systems(Render, apply_uploads.in_set(RenderSystems::PrepareResources));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn rect_texels(rects: &[(IVec2, IVec2)]) -> HashSet<(i32, i32)> {
        let mut set = HashSet::new();
        for (min, size) in rects {
            for y in 0..size.y {
                for x in 0..size.x {
                    assert!(set.insert((min.x + x, min.y + y)), "texel listed twice");
                }
            }
        }
        set
    }

    fn window(center: IVec2, size: i32) -> HashSet<(i32, i32)> {
        let min = window_min(center, size);
        (0..size)
            .flat_map(|y| (0..size).map(move |x| (min.x + x, min.y + y)))
            .collect()
    }

    #[test]
    fn settings_cannot_be_out_of_range() {
        let s = ClipmapSettings::default()
            .with_levels(99)
            .with_grid(3)
            .with_cell_size(f32::NAN)
            .with_morph(5.0);
        assert_eq!(s.levels(), MAX_CLIPMAP_LEVELS);
        assert_eq!(s.grid(), 16);
        assert_eq!(s.cell_size(), 1.0);
        assert_eq!(s.morph, 0.9);
        assert_eq!(ClipmapSettings::default().with_levels(0).levels(), 1);
        // Rounded up to a multiple of 4.
        assert_eq!(ClipmapSettings::default().with_grid(50).grid(), 52);
        assert_eq!(ClipmapSettings::default().with_grid(10_000).grid(), 256);
    }

    #[test]
    fn view_distance_doubles_with_every_level() {
        let a = ClipmapSettings::default().with_levels(3).view_distance();
        let b = ClipmapSettings::default().with_levels(4).view_distance();
        assert_eq!(b, a * 2.0);
        assert_eq!(a, 64.0 * 0.5 * 4.0);
    }

    #[test]
    fn height_bands_are_sorted_and_capped() {
        let c = Color::WHITE;
        let s = ClipmapSettings::default()
            .with_height_bands(&[(9.0, c), (1.0, c), (f32::NAN, c), (5.0, c), (3.0, c), (7.0, c)]);
        let heights: Vec<f32> = s.bands.iter().map(|b| b.0).collect();
        assert_eq!(heights, [1.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn uniform_repeats_the_last_band_and_marks_unused_heights() {
        let s = ClipmapSettings::default().with_height_bands(&[
            (0.0, Color::srgb(1.0, 0.0, 0.0)),
            (10.0, Color::srgb(0.0, 1.0, 0.0)),
        ]);
        let u = ClipmapUniform::new(&s);
        assert_eq!(u.band_info.x, 2.0);
        assert_eq!(u.band_heights.y, 10.0);
        assert_eq!(u.band_heights.z, f32::MAX);
        assert_eq!(u.bands[3], u.bands[1]);
        assert_eq!(ClipmapUniform::new(&ClipmapSettings::default()).band_info.x, 0.0);
    }

    #[test]
    fn finer_centres_are_grid_points_of_the_next_ring() {
        for step in 0..400 {
            let camera = Vec2::new(step as f32 * 0.37 - 70.0, step as f32 * -0.91 + 30.0);
            for level in 0..5 {
                let fine = level_center(camera, 1.0, level);
                let coarse = level_center(camera, 1.0, level + 1);
                // In fine texels, the coarse centre is a multiple of 4 and the
                // fine centre a multiple of 2, so the fine one is a coarse texel.
                assert_eq!(fine.x % 2, 0);
                assert_eq!(fine.y % 2, 0);
                assert_eq!(coarse.x % 2, 0);
                // The camera is never more than a texel off its centre.
                let texel = (1u32 << level) as f32;
                let off = camera / texel - fine.as_vec2();
                assert!(off.abs().max_element() <= 1.0 + 1e-4, "{off:?}");
            }
        }
    }

    #[test]
    fn refill_is_exactly_the_new_window_minus_the_old() {
        let size = 12;
        let old = IVec2::new(4, -6);
        for dx in -14..=14 {
            for dy in -14..=14 {
                let new = old + IVec2::new(dx, dy);
                let listed = rect_texels(&refill_rects(Some(old), new, size));
                let expected: HashSet<_> =
                    window(new, size).difference(&window(old, size)).copied().collect();
                assert_eq!(listed, expected, "move by ({dx}, {dy})");
            }
        }
    }

    #[test]
    fn first_fill_is_the_whole_window() {
        let rects = refill_rects(None, IVec2::new(2, 2), 8);
        assert_eq!(rect_texels(&rects), window(IVec2::new(2, 2), 8));
    }

    #[test]
    fn wrapped_pieces_cover_a_rectangle_once_and_stay_in_the_texture() {
        let texture = 10;
        for (min, size) in [
            (IVec2::new(-3, 4), IVec2::new(10, 10)),
            (IVec2::new(7, -8), IVec2::new(5, 9)),
            (IVec2::new(0, 0), IVec2::new(10, 10)),
            (IVec2::new(-25, 33), IVec2::new(1, 10)),
        ] {
            let pieces = wrap_pieces(min, size, texture);
            let mut seen = HashSet::new();
            for piece in &pieces {
                assert!(piece.origin.x + piece.size.x <= texture as u32);
                assert!(piece.origin.y + piece.size.y <= texture as u32);
                for j in 0..piece.size.y as i32 {
                    for i in 0..piece.size.x as i32 {
                        let t = piece.texel_min + IVec2::new(i, j);
                        assert_eq!(t.x.rem_euclid(texture), piece.origin.x as i32 + i);
                        assert_eq!(t.y.rem_euclid(texture), piece.origin.y as i32 + j);
                        assert!(seen.insert((t.x, t.y)));
                    }
                }
            }
            assert_eq!(seen.len() as i32, size.x * size.y);
        }
    }

    /// Plays back what the GPU would hold: after any walk of the camera, every
    /// texel of every window must be what the source says for that texel.
    #[test]
    fn scrolling_keeps_every_window_equal_to_the_source() {
        let settings = ClipmapSettings::default().with_levels(3).with_grid(16);
        let size = settings.texture_size() as i32;
        let source = HeightSource::callback(|p| {
            if (p.x as i32 + p.y as i32) % 11 == 0 {
                None
            } else {
                Some((p.x * 0.31).sin() * 3.0 + p.y * 0.07)
            }
        });
        let mut textures = vec![vec![[f32::NAN; 2]; (size * size) as usize]; settings.levels];
        let mut centers: Vec<Option<IVec2>> = vec![None; settings.levels];
        let mut camera = Vec2::new(5.0, -3.0);
        for step in 0..120 {
            // Slow drifts, and now and then a jump larger than a window.
            camera += Vec2::new(0.9, -0.55) * ((step % 7) as f32 + 0.3);
            if step == 60 {
                camera += Vec2::new(5000.0, -9000.0);
            }
            for level in 0..settings.levels {
                let center = level_center(camera, settings.cell_size, level);
                let texel_size = settings.cell_size * (1u32 << level) as f32;
                for (min, extent) in refill_rects(centers[level], center, size) {
                    for piece in wrap_pieces(min, extent, size) {
                        let texels = fill_piece(&source, &piece, texel_size);
                        for j in 0..piece.size.y {
                            for i in 0..piece.size.x {
                                let at = ((piece.origin.y + j) * size as u32 + piece.origin.x + i)
                                    as usize;
                                let from = ((j * piece.size.x + i) * 2) as usize;
                                textures[level][at] = [texels[from], texels[from + 1]];
                            }
                        }
                    }
                }
                centers[level] = Some(center);
                let min = window_min(center, size);
                for y in 0..size {
                    for x in 0..size {
                        let t = min + IVec2::new(x, y);
                        let at = (t.y.rem_euclid(size) * size + t.x.rem_euclid(size)) as usize;
                        assert_eq!(
                            textures[level][at],
                            texel_value(&source, t, texel_size),
                            "step {step}, level {level}, texel {t}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn non_finite_heights_are_holes() {
        let source = HeightSource::heights(|p| if p.x > 0.0 { f32::NAN } else { 2.0 });
        assert_eq!(texel_value(&source, IVec2::new(1, 0), 1.0), [0.0, 1.0]);
        assert_eq!(texel_value(&source, IVec2::new(-1, 0), 1.0), [2.0, 0.0]);
    }

    #[test]
    fn the_grid_mesh_has_the_expected_size() {
        let mesh = ring_mesh(16);
        assert_eq!(mesh.count_vertices(), 17 * 17);
        assert_eq!(mesh.indices().unwrap().len(), 16 * 16 * 6);
    }

    #[test]
    fn grid_triangles_face_up() {
        let mesh = ring_mesh(16);
        let positions: Vec<[f32; 3]> = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap()
            .to_vec();
        let indices: Vec<u32> = mesh.indices().unwrap().iter().map(|i| i as u32).collect();
        for tri in indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|k| bevy::math::Vec3::from(positions[tri[k] as usize]));
            assert!((b - a).cross(c - a).y > 0.0);
        }
    }
}
