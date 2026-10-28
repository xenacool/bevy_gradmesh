//! Comprehensive character creator showcase — demonstrates all BevyGradmesh facial features.
//!
//! Shows multiple character variants side-by-side, each with different combinations
//! of eye shapes, eyebrow styles, nose shapes, mouth shapes, blush, hair, facial hair,
//! and glasses.
//!
//! Run with: `cargo run --example character_creator`

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod common;

use bevy::camera::visibility::InheritedVisibility;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy_gradmesh::{
    dual_directional, dual_radial, opaque, Shape, ShapeDetail,
    GradMeshMaterial, GradMeshPlugin, transparent,
};
use common::screenshot::AutoScreenshotPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BevyGradmesh Character Creator".into(),
                resolution: (1600, 900).into(),
                ..default()
            }),
            ..default()
        }).set(RenderPlugin {
            synchronous_pipeline_compilation: true,
            ..default()
        }))
        .add_plugins(GradMeshPlugin)
        .add_plugins(
            AutoScreenshotPlugin::new("screenshot_character_creator.png")
                .with_title("Character Creator")
                .with_description("7 characters with varied facial features and accessories")
        )
        .add_systems(Startup, setup)
        .add_systems(Update, rotate_characters)
        .run();
}

/// Slowly rotates all characters so the user can inspect them from all angles.
fn rotate_characters(time: Res<Time>, mut query: Query<&mut Transform, With<CharacterMarker>>) {
    let t = time.elapsed_secs() * 0.3;
    for mut transform in &mut query {
        transform.rotation = Quat::from_euler(EulerRot::YXZ, t, 0.0, 0.0);
    }
}

/// Marker component for character root entities.
#[derive(Component)]
struct CharacterMarker;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<bevy::render::mesh::Mesh>>,
    mut materials: ResMut<Assets<GradMeshMaterial>>,
) {
    // Camera — centered to see all 7 characters
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 3.0, 14.0).looking_at(Vec3::new(0.0, 1.0, -2.0), Vec3::Y),
    ));

    // Lighting — soft directional + ambient, tuned so characters aren't blown out
    commands.spawn(DirectionalLight {
        illuminance: 3_000.0,
        ..default()
    });
    commands.spawn(PointLight {
        intensity: 20_000.0,
        ..default()
    });

    // Ground plane — large enough for all 7 characters
    let ground_mesh = meshes.add(Shape::rectangle(Vec2::new(20.0, 12.0), ShapeDetail::Minimal).build());
    commands.spawn((
        Mesh3d(ground_mesh),
        MeshMaterial3d(materials.add(opaque(0.15, 0.15, 0.18))),
        Transform::from_xyz(0.0, -0.5, -3.0).with_rotation(Quat::from_rotation_x(-std::f32::consts::PI / 2.0)),
    ));

    // Character 1: Classic — classic eyes, arch eyebrows, button nose, smile, no blush, short parted hair, no facial hair, no glasses
    let char1 = CharacterConfig {
        position: Vec3::new(-3.0, 0.0, 0.0),
        skin_dark: [0.74, 0.62, 0.53],
        skin_light: [0.96, 0.80, 0.69],
        hair_dark: [0.10, 0.07, 0.05],
        hair_light: [0.20, 0.14, 0.10],
        eye_color: [0.18, 0.38, 0.68],
        eye_shape: EyeShape::ClassicDank,
        eyebrow_shape: EyebrowShape::ClassicArch,
        eyebrow_color: [0.10, 0.07, 0.05],
        nose_shape: NoseShape::Button,
        mouth_shape: MouthShape::Smile,
        blush_shape: BlushShape::None,
        hair_style: HairStyle::ShortParted,
        facial_hair: FacialHairStyle::None,
        glasses: GlassesStyle::None,
        label: "Classic",
    };
    char1.build(&mut commands, &mut meshes, &mut materials);

    // Character 2: Anime — wide eyes, straight brows, triangle nose, grin, soft blush, spiky hair, no facial hair, round glasses
    let char2 = CharacterConfig {
        position: Vec3::new(0.0, 0.0, 0.0),
        skin_dark: [0.80, 0.68, 0.58],
        skin_light: [0.98, 0.85, 0.75],
        hair_dark: [0.55, 0.15, 0.10],
        hair_light: [0.80, 0.25, 0.18],
        eye_color: [0.60, 0.15, 0.10],
        eye_shape: EyeShape::AnimeWide,
        eyebrow_shape: EyebrowShape::BoldStraight,
        eyebrow_color: [0.55, 0.15, 0.10],
        nose_shape: NoseShape::TrianglePoint,
        mouth_shape: MouthShape::WideGrin,
        blush_shape: BlushShape::SoftBlush,
        hair_style: HairStyle::SpikyAnime,
        facial_hair: FacialHairStyle::None,
        glasses: GlassesStyle::RoundFrames,
        label: "Anime",
    };
    char2.build(&mut commands, &mut meshes, &mut materials);

    // Character 3: Rugged — sharp eyes, angled brows, bridge nose, smirk, freckles, ponytail, goatee, sunglasses
    let char3 = CharacterConfig {
        position: Vec3::new(3.0, 0.0, 0.0),
        skin_dark: [0.55, 0.40, 0.30],
        skin_light: [0.78, 0.60, 0.48],
        hair_dark: [0.15, 0.12, 0.08],
        hair_light: [0.25, 0.20, 0.14],
        eye_color: [0.20, 0.50, 0.20],
        eye_shape: EyeShape::CoolSharp,
        eyebrow_shape: EyebrowShape::SharpAngled,
        eyebrow_color: [0.15, 0.12, 0.08],
        nose_shape: NoseShape::BridgeLine,
        mouth_shape: MouthShape::Smirk,
        blush_shape: BlushShape::Freckles,
        hair_style: HairStyle::Ponytail,
        facial_hair: FacialHairStyle::Goatee,
        glasses: GlassesStyle::Sunglasses,
        label: "Rugged",
    };
    char3.build(&mut commands, &mut meshes, &mut materials);

    // Character 4: Cute — oval eyes, rounded brows, soft nose, cat mouth, round cheeks, twin buns, no facial hair, headband
    let char4 = CharacterConfig {
        position: Vec3::new(-3.0, 0.0, -3.0),
        skin_dark: [0.85, 0.72, 0.62],
        skin_light: [0.97, 0.88, 0.80],
        hair_dark: [0.70, 0.40, 0.30],
        hair_light: [0.90, 0.60, 0.50],
        eye_color: [0.50, 0.20, 0.60],
        eye_shape: EyeShape::CuteOval,
        eyebrow_shape: EyebrowShape::CuteRounded,
        eyebrow_color: [0.70, 0.40, 0.30],
        nose_shape: NoseShape::SoftRound,
        mouth_shape: MouthShape::CatMouth,
        blush_shape: BlushShape::RoundCheeks,
        hair_style: HairStyle::TwinBuns,
        facial_hair: FacialHairStyle::None,
        glasses: GlassesStyle::Headband,
        label: "Cute",
    };
    char4.build(&mut commands, &mut meshes, &mut materials);

    // Character 5: Sleepy — sleepy eyes, bushy brows, straight nose, neutral mouth, no blush, bob fringe, stubble, square glasses
    let char5 = CharacterConfig {
        position: Vec3::new(0.0, 0.0, -3.0),
        skin_dark: [0.65, 0.52, 0.42],
        skin_light: [0.88, 0.72, 0.62],
        hair_dark: [0.30, 0.25, 0.20],
        hair_light: [0.45, 0.38, 0.30],
        eye_color: [0.30, 0.45, 0.30],
        eye_shape: EyeShape::Sleepy,
        eyebrow_shape: EyebrowShape::ThickBushy,
        eyebrow_color: [0.30, 0.25, 0.20],
        nose_shape: NoseShape::Straight,
        mouth_shape: MouthShape::NeutralLine,
        blush_shape: BlushShape::None,
        hair_style: HairStyle::BobFringe,
        facial_hair: FacialHairStyle::Stubble,
        glasses: GlassesStyle::SquareFrames,
        label: "Sleepy",
    };
    char5.build(&mut commands, &mut meshes, &mut materials);

    // Character 6: Gentle — arch eyes, dot brows, button nose, O mouth, no blush, bald, mustache, no glasses
    let char6 = CharacterConfig {
        position: Vec3::new(3.0, 0.0, -3.0),
        skin_dark: [0.70, 0.58, 0.48],
        skin_light: [0.92, 0.78, 0.68],
        hair_dark: [0.12, 0.10, 0.08],
        hair_light: [0.22, 0.18, 0.14],
        eye_color: [0.25, 0.40, 0.55],
        eye_shape: EyeShape::GentleArch,
        eyebrow_shape: EyebrowShape::ShortDots,
        eyebrow_color: [0.12, 0.10, 0.08],
        nose_shape: NoseShape::Button,
        mouth_shape: MouthShape::SmallO,
        blush_shape: BlushShape::None,
        hair_style: HairStyle::Bald,
        facial_hair: FacialHairStyle::ClassicMustache,
        glasses: GlassesStyle::None,
        label: "Gentle",
    };
    char6.build(&mut commands, &mut meshes, &mut materials);

    // Character 7: Full beard — gentle eyes, bold brows, triangle nose, wide grin, soft blush, short parted, full beard, round glasses
    let char7 = CharacterConfig {
        position: Vec3::new(-3.0, 0.0, -6.0),
        skin_dark: [0.60, 0.48, 0.38],
        skin_light: [0.82, 0.66, 0.56],
        hair_dark: [0.18, 0.14, 0.10],
        hair_light: [0.30, 0.24, 0.18],
        eye_color: [0.40, 0.30, 0.20],
        eye_shape: EyeShape::GentleArch,
        eyebrow_shape: EyebrowShape::BoldStraight,
        eyebrow_color: [0.18, 0.14, 0.10],
        nose_shape: NoseShape::TrianglePoint,
        mouth_shape: MouthShape::WideGrin,
        blush_shape: BlushShape::SoftBlush,
        hair_style: HairStyle::ShortParted,
        facial_hair: FacialHairStyle::FullBeard,
        glasses: GlassesStyle::RoundFrames,
        label: "Bearded",
    };
    char7.build(&mut commands, &mut meshes, &mut materials);

}

// ─── Character data enums ───

#[derive(Clone, Copy)]
enum EyeShape {
    ClassicDank,
    AnimeWide,
    CoolSharp,
    GentleArch,
    Sleepy,
    CuteOval,
}

#[derive(Clone, Copy)]
enum EyebrowShape {
    ClassicArch,
    BoldStraight,
    SharpAngled,
    CuteRounded,
    ThickBushy,
    ShortDots,
}

#[derive(Clone, Copy)]
enum NoseShape {
    Button,
    TrianglePoint,
    SoftRound,
    BridgeLine,
    Straight,
}

#[derive(Clone, Copy)]
enum MouthShape {
    Smile,
    WideGrin,
    NeutralLine,
    Smirk,
    SmallO,
    CatMouth,
}

#[derive(Clone, Copy)]
enum BlushShape {
    None,
    SoftBlush,
    RoundCheeks,
    Freckles,
}

#[derive(Clone, Copy)]
enum HairStyle {
    Bald,
    ShortParted,
    SpikyAnime,
    BobFringe,
    Ponytail,
    TwinBuns,
}

#[derive(Clone, Copy)]
enum FacialHairStyle {
    None,
    Stubble,
    ClassicMustache,
    Goatee,
    FullBeard,
}

#[derive(Clone, Copy)]
enum GlassesStyle {
    None,
    RoundFrames,
    SquareFrames,
    Sunglasses,
    Headband,
}

// ─── Character configuration ───

struct CharacterConfig {
    position: Vec3,
    skin_dark: [f32; 3],
    skin_light: [f32; 3],
    hair_dark: [f32; 3],
    hair_light: [f32; 3],
    eye_color: [f32; 3],
    eye_shape: EyeShape,
    eyebrow_shape: EyebrowShape,
    eyebrow_color: [f32; 3],
    nose_shape: NoseShape,
    mouth_shape: MouthShape,
    blush_shape: BlushShape,
    hair_style: HairStyle,
    facial_hair: FacialHairStyle,
    glasses: GlassesStyle,
    label: &'static str,
}

impl CharacterConfig {
    fn build(
        &self,
        commands: &mut Commands,
        meshes: &mut Assets<bevy::render::mesh::Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
    ) {
        let root = commands
            .spawn((
                Transform::from_translation(self.position),
                InheritedVisibility::default(),
                CharacterMarker,
            ))
            .id();

        let skin_mat = materials.add(dual_directional(
            self.skin_dark,
            self.skin_light,
            [0.2, 0.9, 0.4],
        ));
        let skin_flat = materials.add(opaque(
            self.skin_dark[0].max(self.skin_light[0] * 0.8),
            self.skin_dark[1].max(self.skin_light[1] * 0.8),
            self.skin_dark[2].max(self.skin_light[2] * 0.8),
        ));
        let _hair_mat = materials.add(dual_directional(
            self.hair_dark,
            self.hair_light,
            [0.1, 0.9, 0.3],
        ));

        // ─── Body ───
        // Torso
        self.add_part(
            commands, meshes, &skin_mat, root,
            Shape::cuboid(Vec3::new(0.48, 0.34, 0.34), 0.35, ShapeDetail::High),
            Transform::from_xyz(0.0, 0.5, 0.0),
        );
        // Spine
        self.add_part(
            commands, meshes, &skin_mat, root,
            Shape::cuboid(Vec3::new(0.44, 0.21, 0.30), 0.35, ShapeDetail::High),
            Transform::from_xyz(0.0, 0.25, 0.0),
        );
        // Hips
        self.add_part(
            commands, meshes, &skin_mat, root,
            Shape::cuboid(Vec3::new(0.44, 0.19, 0.32), 0.35, ShapeDetail::High),
            Transform::from_xyz(0.0, 0.07, 0.0),
        );

        // Arms
        for side in [-1, 1] {
            let x = side as f32 * 0.34;
            // Upper arm
            self.add_part(
                commands, meshes, &skin_mat, root,
                Shape::cylinder(0.10, 0.24, ShapeDetail::Medium),
                Transform::from_xyz(x, 0.62, 0.0),
            );
            // Lower arm
            self.add_part(
                commands, meshes, &skin_mat, root,
                Shape::cylinder(0.09, 0.20, ShapeDetail::Medium),
                Transform::from_xyz(x, 0.38, 0.0),
            );
            // Hand
            self.add_part(
                commands, meshes, &skin_mat, root,
                Shape::sphere(0.12, ShapeDetail::Medium),
                Transform::from_xyz(x, 0.24, 0.0),
            );
            // Thumb
            self.add_part(
                commands, meshes, &skin_mat, root,
                Shape::sphere(0.05, ShapeDetail::Low),
                Transform::from_xyz(x + side as f32 * 0.08, 0.26, 0.04),
            );
        }

        // Legs
        for side in [-1, 1] {
            let x = side as f32 * 0.15;
            // Upper leg
            self.add_part(
                commands, meshes, &skin_mat, root,
                Shape::cylinder(0.10, 0.24, ShapeDetail::Medium),
                Transform::from_xyz(x, 0.12, 0.0),
            );
            // Lower leg
            self.add_part(
                commands, meshes, &skin_mat, root,
                Shape::cylinder(0.09, 0.16, ShapeDetail::Medium),
                Transform::from_xyz(x, -0.12, 0.0),
            );
            // Foot
            self.add_part(
                commands, meshes, &skin_mat, root,
                Shape::cuboid(Vec3::new(0.17, 0.11, 0.25), 0.45, ShapeDetail::High),
                Transform::from_xyz(x, -0.22, -0.012)
                    .with_rotation(Quat::from_euler(EulerRot::XZY, -136.04_f32.to_radians(), 0.0, 0.0)),
            );
        }

        // ─── Head ───
        // Head base (sphere)
        self.add_part(
            commands, meshes, &skin_mat, root,
            Shape::sphere(0.42, ShapeDetail::High),
            Transform::from_xyz(0.0, 1.0, 0.0),
        );
        // Ears
        for side in [-1, 1] {
            self.add_part(
                commands, meshes, &skin_flat, root,
                Shape::ellipsoid(Vec3::new(0.12, 0.18, 0.14), ShapeDetail::Medium),
                Transform::from_xyz(side as f32 * 0.41, 1.0, 0.0)
                    .with_rotation(Quat::from_euler(EulerRot::ZXY, side as f32 * 12.0_f32.to_radians(), 0.0, 0.0)),
            );
        }

        // ─── Facial Features ───
        self.build_eyes(commands, meshes, materials, root);
        self.build_eyebrows(commands, meshes, materials, root);
        self.build_nose(commands, meshes, materials, root);
        self.build_mouth(commands, meshes, materials, root);
        self.build_blush(commands, meshes, materials, root);
        self.build_hair(commands, meshes, materials, root);
        self.build_facial_hair(commands, meshes, materials, root);
        self.build_glasses(commands, meshes, materials, root);

        // ─── Label ───
        commands.entity(root).with_children(|parent| {
            parent.spawn((
                Text2d(self.label.to_string()),
                TextColor(Color::srgb(1.0, 1.0, 1.0)),
                Transform::from_xyz(0.0, 1.8, 0.0),
            ));
        });
    }

    // ─── Eyes ───

    fn build_eyes(
        &self,
        commands: &mut Commands,
        meshes: &mut Assets<bevy::render::mesh::Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        root: Entity,
    ) {
        let (sclera_scale, iris_scale) = match self.eye_shape {
            EyeShape::ClassicDank => (Vec3::new(0.18, 0.22, 0.08), Vec3::new(0.10, 0.14, 0.06)),
            EyeShape::AnimeWide => (Vec3::new(0.24, 0.26, 0.08), Vec3::new(0.16, 0.20, 0.06)),
            EyeShape::CoolSharp => (Vec3::new(0.22, 0.14, 0.08), Vec3::new(0.12, 0.11, 0.06)),
            EyeShape::GentleArch => (Vec3::new(0.20, 0.18, 0.08), Vec3::new(0.11, 0.12, 0.06)),
            EyeShape::Sleepy => (Vec3::new(0.22, 0.12, 0.08), Vec3::new(0.10, 0.09, 0.06)),
            EyeShape::CuteOval => {
                // Spheres for cute oval eyes
                self.add_part(
                    commands, meshes,
                    &materials.add(opaque(0.98, 0.98, 0.98)), root,
                    Shape::sphere(0.11, ShapeDetail::Medium),
                    Transform::from_xyz(-0.22, 1.05, 0.38),
                );
                self.add_part(
                    commands, meshes,
                    &materials.add(dual_radial(
                        [self.eye_color[0] * 1.2, self.eye_color[1] * 1.2, self.eye_color[2] * 1.2],
                        [self.eye_color[0] * 0.7, self.eye_color[1] * 0.7, self.eye_color[2] * 0.7],
                    )), root,
                    Shape::sphere(0.07, ShapeDetail::Medium),
                    Transform::from_xyz(-0.22, 1.05, 0.40),
                );
                self.add_part(
                    commands, meshes,
                    &materials.add(opaque(0.06, 0.06, 0.08)), root,
                    Shape::sphere(0.035, ShapeDetail::Low),
                    Transform::from_xyz(-0.22, 1.05, 0.42),
                );
                self.add_part(
                    commands, meshes,
                    &materials.add(opaque(1.0, 1.0, 1.0)), root,
                    Shape::sphere(0.016, ShapeDetail::Low),
                    Transform::from_xyz(-0.195, 1.08, 0.43),
                );
                self.add_part(
                    commands, meshes,
                    &materials.add(opaque(0.98, 0.98, 0.98)), root,
                    Shape::sphere(0.11, ShapeDetail::Medium),
                    Transform::from_xyz(0.22, 1.05, 0.38),
                );
                self.add_part(
                    commands, meshes,
                    &materials.add(dual_radial(
                        [self.eye_color[0] * 1.2, self.eye_color[1] * 1.2, self.eye_color[2] * 1.2],
                        [self.eye_color[0] * 0.7, self.eye_color[1] * 0.7, self.eye_color[2] * 0.7],
                    )), root,
                    Shape::sphere(0.07, ShapeDetail::Medium),
                    Transform::from_xyz(0.22, 1.05, 0.40),
                );
                self.add_part(
                    commands, meshes,
                    &materials.add(opaque(0.06, 0.06, 0.08)), root,
                    Shape::sphere(0.035, ShapeDetail::Low),
                    Transform::from_xyz(0.22, 1.05, 0.42),
                );
                self.add_part(
                    commands, meshes,
                    &materials.add(opaque(1.0, 1.0, 1.0)), root,
                    Shape::sphere(0.016, ShapeDetail::Low),
                    Transform::from_xyz(0.245, 1.08, 0.43),
                );
                return;
            }
        };

        let sclera_mat = materials.add(opaque(0.98, 0.98, 0.98));
        let iris_mat = materials.add(dual_radial(
            [self.eye_color[0] * 1.2, self.eye_color[1] * 1.2, self.eye_color[2] * 1.2],
            [self.eye_color[0] * 0.7, self.eye_color[1] * 0.7, self.eye_color[2] * 0.7],
        ));
        let pupil_mat = materials.add(opaque(0.06, 0.06, 0.08));
        let shine_mat = materials.add(opaque(1.0, 1.0, 1.0));

        for side in [-1, 1] {
            let x = side as f32 * 0.22;
            let z_fwd = 0.38;

            // Sclera
            self.add_part(
                commands, meshes, &sclera_mat, root,
                Shape::ellipsoid(sclera_scale, ShapeDetail::Medium),
                Transform::from_xyz(x, 1.05, z_fwd),
            );
            // Iris
            self.add_part(
                commands, meshes, &iris_mat, root,
                Shape::ellipsoid(iris_scale, ShapeDetail::Medium),
                Transform::from_xyz(x, 1.05, z_fwd + 0.02),
            );
            // Pupil
            self.add_part(
                commands, meshes, &pupil_mat, root,
                Shape::sphere(0.035, ShapeDetail::Low),
                Transform::from_xyz(x, 1.05, z_fwd + 0.035),
            );
            // Shine highlight
            self.add_part(
                commands, meshes, &shine_mat, root,
                Shape::sphere(0.016, ShapeDetail::Low),
                Transform::from_xyz(x + side as f32 * 0.025, 1.08, z_fwd + 0.045),
            );
        }
    }

    // ─── Eyebrows ───

    fn build_eyebrows(
        &self,
        commands: &mut Commands,
        meshes: &mut Assets<bevy::render::mesh::Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        root: Entity,
    ) {
        let brow_mat = materials.add(opaque(
            self.eyebrow_color[0],
            self.eyebrow_color[1],
            self.eyebrow_color[2],
        ));
        let z_fwd = 0.40;

        let brow_shape = match self.eyebrow_shape {
            EyebrowShape::ClassicArch => {
                Shape::cuboid(Vec3::new(0.18, 0.04, 0.04), 0.5, ShapeDetail::Medium)
            }
            EyebrowShape::BoldStraight => {
                Shape::cuboid(Vec3::new(0.22, 0.06, 0.04), 0.2, ShapeDetail::Medium)
            }
            EyebrowShape::SharpAngled => {
                Shape::cuboid(Vec3::new(0.20, 0.05, 0.04), 0.2, ShapeDetail::Medium)
            }
            EyebrowShape::CuteRounded => {
                Shape::capsule(0.025, 0.18, ShapeDetail::Medium)
            }
            EyebrowShape::ThickBushy => {
                Shape::ellipsoid(Vec3::new(0.22, 0.08, 0.06), ShapeDetail::Medium)
            }
            EyebrowShape::ShortDots => {
                Shape::sphere(0.04, ShapeDetail::Low)
            }
        };

        for side in [-1, 1] {
            let x = side as f32 * 0.22;
            let rot = match self.eyebrow_shape {
                EyebrowShape::SharpAngled => {
                    Quat::from_euler(EulerRot::ZXY, side as f32 * 15.0_f32.to_radians(), 0.0, 0.0)
                }
                _ => Quat::from_euler(EulerRot::ZXY, side as f32 * 5.0_f32.to_radians(), 0.0, 0.0),
            };
            self.add_part(
                commands, meshes, &brow_mat, root,
                brow_shape.clone(),
                Transform::from_xyz(x, 1.18, z_fwd).with_rotation(rot),
            );
        }
    }

    // ─── Nose ───

    fn build_nose(
        &self,
        commands: &mut Commands,
        meshes: &mut Assets<bevy::render::mesh::Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        root: Entity,
    ) {
        let nose_mat = materials.add(opaque(
            self.skin_dark[0] * 0.92,
            self.skin_dark[1] * 0.92,
            self.skin_dark[2] * 0.92,
        ));
        let z_fwd = 0.41;

        let nose_shape = match self.nose_shape {
            NoseShape::Button => Shape::sphere(0.045, ShapeDetail::Medium),
            NoseShape::TrianglePoint => Shape::cone(0.045, 0.08, ShapeDetail::Medium),
            NoseShape::SoftRound => Shape::ellipsoid(Vec3::new(0.06, 0.05, 0.05), ShapeDetail::Medium),
            NoseShape::BridgeLine => Shape::cuboid(Vec3::new(0.03, 0.12, 0.05), 0.4, ShapeDetail::Medium),
            NoseShape::Straight => Shape::cuboid(Vec3::new(0.04, 0.10, 0.06), 0.3, ShapeDetail::Medium),
        };

        self.add_part(
            commands, meshes, &nose_mat, root,
            nose_shape,
            Transform::from_xyz(0.0, 0.96, z_fwd),
        );
    }

    // ─── Mouth ───

    fn build_mouth(
        &self,
        commands: &mut Commands,
        meshes: &mut Assets<bevy::render::mesh::Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        root: Entity,
    ) {
        let lip_col = [0.85, 0.45, 0.45];
        let mouth_col = [
            self.skin_dark[0] * 0.72 * 0.6 + lip_col[0] * 0.4,
            self.skin_dark[1] * 0.72 * 0.6 + lip_col[1] * 0.4,
            self.skin_dark[2] * 0.72 * 0.6 + lip_col[2] * 0.4,
        ];
        let mouth_mat = materials.add(opaque(mouth_col[0], mouth_col[1], mouth_col[2]));
        let z_fwd = 0.40;

        match self.mouth_shape {
            MouthShape::Smile => {
                self.add_part(
                    commands, meshes, &mouth_mat, root,
                    Shape::torus(0.08, 0.016, std::f32::consts::PI * 1.1, std::f32::consts::PI * 0.8),
                    Transform::from_xyz(0.0, 0.85, z_fwd)
                        .with_rotation(Quat::from_euler(EulerRot::XZY, std::f32::consts::PI / 2.0, 0.0, 0.0)),
                );
            }
            MouthShape::WideGrin => {
                self.add_part(
                    commands, meshes, &mouth_mat, root,
                    Shape::torus(0.12, 0.02, std::f32::consts::PI * 1.05, std::f32::consts::PI * 0.95),
                    Transform::from_xyz(0.0, 0.86, z_fwd)
                        .with_rotation(Quat::from_euler(EulerRot::XZY, std::f32::consts::PI / 2.0, 0.0, 0.0)),
                );
            }
            MouthShape::NeutralLine => {
                self.add_part(
                    commands, meshes, &mouth_mat, root,
                    Shape::cuboid(Vec3::new(0.14, 0.022, 0.03), 0.5, ShapeDetail::Medium),
                    Transform::from_xyz(0.0, 0.82, z_fwd),
                );
            }
            MouthShape::Smirk => {
                self.add_part(
                    commands, meshes, &mouth_mat, root,
                    Shape::torus(0.09, 0.018, std::f32::consts::PI * 1.22, std::f32::consts::PI * 0.8),
                    Transform::from_xyz(0.02, 0.84, z_fwd)
                        .with_rotation(Quat::from_euler(EulerRot::XYZ, std::f32::consts::PI / 2.0, 10.0_f32.to_radians(), 0.0)),
                );
            }
            MouthShape::SmallO => {
                self.add_part(
                    commands, meshes, &mouth_mat, root,
                    Shape::torus(0.035, 0.016, 0.0, 2.0 * std::f32::consts::PI),
                    Transform::from_xyz(0.0, 0.82, z_fwd)
                        .with_rotation(Quat::from_euler(EulerRot::XZY, std::f32::consts::PI / 2.0, 0.0, 0.0)),
                );
            }
            MouthShape::CatMouth => {
                for side in [-1, 1] {
                    self.add_part(
                        commands, meshes, &mouth_mat, root,
                        Shape::torus(0.045, 0.014, std::f32::consts::PI * 1.05, std::f32::consts::PI * 0.95),
                        Transform::from_xyz(side as f32 * 0.05, 0.82, z_fwd)
                            .with_rotation(Quat::from_euler(EulerRot::XZY, std::f32::consts::PI / 2.0, 0.0, 0.0)),
                    );
                }
            }
        }
    }

    // ─── Blush ───

    fn build_blush(
        &self,
        commands: &mut Commands,
        meshes: &mut Assets<bevy::render::mesh::Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        root: Entity,
    ) {
        match self.blush_shape {
            BlushShape::None => {}
            BlushShape::SoftBlush => {
                let blush_mat = materials.add(transparent(0.95, 0.45, 0.50, 0.55));
                for side in [-1, 1] {
                    self.add_part(
                        commands, meshes, &blush_mat, root,
                        Shape::ellipsoid(Vec3::new(0.12, 0.06, 0.04), ShapeDetail::Medium),
                        Transform::from_xyz(side as f32 * 0.28, 0.93, 0.38),
                    );
                }
            }
            BlushShape::RoundCheeks => {
                let blush_mat = materials.add(transparent(0.95, 0.45, 0.50, 0.55));
                for side in [-1, 1] {
                    self.add_part(
                        commands, meshes, &blush_mat, root,
                        Shape::sphere(0.07, ShapeDetail::Medium),
                        Transform::from_xyz(side as f32 * 0.28, 0.93, 0.38),
                    );
                }
            }
            BlushShape::Freckles => {
                let freckle_mat = materials.add(opaque(0.55, 0.35, 0.25));
                for side in [-1, 1] {
                    for i in 0..3 {
                        self.add_part(
                            commands, meshes, &freckle_mat, root,
                            Shape::sphere(0.012, ShapeDetail::Low),
                            Transform::from_xyz(
                                side as f32 * (0.26 + i as f32 * 0.03),
                                0.93 + (i % 2) as f32 * 0.02,
                                0.40,
                            ),
                        );
                    }
                }
            }
        }
    }

    // ─── Hair ───

    fn build_hair(
        &self,
        commands: &mut Commands,
        meshes: &mut Assets<bevy::render::mesh::Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        root: Entity,
    ) {
        let hair_mat = materials.add(dual_directional(
            self.hair_dark,
            self.hair_light,
            [0.1, 0.9, 0.3],
        ));
        let head_y = 1.0;

        match self.hair_style {
            HairStyle::Bald => {
                // Stubble — transparent hemisphere
                let stubble_mat = materials.add(transparent(
                    self.hair_dark[0] * 0.9,
                    self.hair_dark[1] * 0.9,
                    self.hair_dark[2] * 0.9,
                    0.3,
                ));
                self.add_part(
                    commands, meshes, &stubble_mat, root,
                    Shape::hemisphere(0.422, true, ShapeDetail::High),
                    Transform::from_xyz(0.0, head_y + 0.10, -0.04)
                        .with_rotation(Quat::from_euler(EulerRot::XZY, -20.0_f32.to_radians(), 0.0, 0.0)),
                );
            }
            _ => {
                // Base hair cap
                let vol = 1.0;
                self.add_part(
                    commands, meshes, &hair_mat, root,
                    Shape::hemisphere(0.435 * vol, true, ShapeDetail::High),
                    Transform::from_xyz(0.0, head_y + 0.10 * vol, -0.04 * vol)
                        .with_rotation(Quat::from_euler(EulerRot::XZY, -20.0_f32.to_radians(), 0.0, 0.0)),
                );
                // Back cap
                self.add_part(
                    commands, meshes, &hair_mat, root,
                    Shape::hemisphere(0.425 * vol, true, ShapeDetail::High),
                    Transform::from_xyz(0.0, head_y + 0.12 * vol, -0.04 * vol)
                        .with_rotation(Quat::from_euler(EulerRot::XZY, -90.0_f32.to_radians(), 0.0, 0.0)),
                );
                // Side hair
                for side in [-1, 1] {
                    self.add_part(
                        commands, meshes, &hair_mat, root,
                        Shape::cuboid(Vec3::new(0.06, 0.18, 0.22) * vol, 0.35, ShapeDetail::Medium),
                        Transform::from_xyz(side as f32 * 0.415 * vol, head_y + 0.16, -0.02 * vol),
                    );
                }

                // Style-specific additions
                match self.hair_style {
                    HairStyle::ShortParted => {
                        // Bangs
                        self.add_part(
                            commands, meshes, &hair_mat, root,
                            Shape::cuboid(Vec3::new(0.40, 0.14, 0.20), 0.35, ShapeDetail::Medium),
                            Transform::from_xyz(0.10, head_y + 0.34, 0.28)
                                .with_rotation(Quat::from_euler(EulerRot::XYZ, 6.0_f32.to_radians(), 0.0, -14.0_f32.to_radians())),
                        );
                        // Sideburns
                        for side in [-1, 1] {
                            self.add_part(
                                commands, meshes, &hair_mat, root,
                                Shape::cuboid(Vec3::new(0.06, 0.20, 0.16), 0.35, ShapeDetail::Medium),
                                Transform::from_xyz(side as f32 * 0.415, head_y + 0.12, 0.04),
                            );
                        }
                    }
                    HairStyle::SpikyAnime => {
                        // Top spikes
                        for i in 0..5 {
                            let x = (i as f32 - 2.0) * 0.12;
                            let z_rot = (i as f32 - 2.0) * 18.0_f32.to_radians();
                            self.add_part(
                                commands, meshes, &hair_mat, root,
                                Shape::cone(0.11, 0.34, ShapeDetail::Medium),
                                Transform::from_xyz(x, head_y + 0.44, 0.02)
                                    .with_rotation(Quat::from_euler(EulerRot::XZY, -15.0_f32.to_radians(), 0.0, z_rot)),
                            );
                        }
                        // Front spikes
                        for i in 0..3 {
                            let x = (i as f32 - 1.0) * 0.14;
                            let z_rot = (i as f32 - 1.0) * 20.0_f32.to_radians();
                            self.add_part(
                                commands, meshes, &hair_mat, root,
                                Shape::cone(0.09, 0.24, ShapeDetail::Medium),
                                Transform::from_xyz(x, head_y + 0.34, 0.26)
                                    .with_rotation(Quat::from_euler(EulerRot::XZY, 45.0_f32.to_radians(), 0.0, z_rot)),
                            );
                        }
                        // Side spikes
                        for side in [-1, 1] {
                            self.add_part(
                                commands, meshes, &hair_mat, root,
                                Shape::cone(0.09, 0.24, ShapeDetail::Medium),
                                Transform::from_xyz(side as f32 * 0.40, head_y + 0.28, 0.02)
                                    .with_rotation(Quat::from_euler(EulerRot::XZY, 0.0, 0.0, side as f32 * 45.0_f32.to_radians())),
                            );
                        }
                    }
                    HairStyle::BobFringe => {
                        // Fringe
                        self.add_part(
                            commands, meshes, &hair_mat, root,
                            Shape::cuboid(Vec3::new(0.58, 0.14, 0.18), 0.35, ShapeDetail::Medium),
                            Transform::from_xyz(0.0, head_y + 0.32, 0.28),
                        );
                        // Side locks
                        for side in [-1, 1] {
                            self.add_part(
                                commands, meshes, &hair_mat, root,
                                Shape::capsule(0.12, 0.46, ShapeDetail::Medium),
                                Transform::from_xyz(side as f32 * 0.43, head_y + 0.06, 0.04)
                                    .with_rotation(Quat::from_euler(EulerRot::ZXY, side as f32 * 5.0_f32.to_radians(), 0.0, 0.0)),
                            );
                        }
                    }
                    HairStyle::Ponytail => {
                        // Bangs
                        self.add_part(
                            commands, meshes, &hair_mat, root,
                            Shape::cuboid(Vec3::new(0.50, 0.12, 0.16), 0.35, ShapeDetail::Medium),
                            Transform::from_xyz(0.0, head_y + 0.34, 0.26),
                        );
                        // Ponytail tie (torus)
                        let tie_mat = materials.add(opaque(0.85, 0.2, 0.2));
                        self.add_part(
                            commands, meshes, &tie_mat, root,
                            Shape::torus(0.08, 0.03, 0.0, 2.0 * std::f32::consts::PI),
                            Transform::from_xyz(0.0, head_y + 0.34, -0.44)
                                .with_rotation(Quat::from_euler(EulerRot::XZY, 65.0_f32.to_radians(), 0.0, 0.0)),
                        );
                        // Ponytail
                        self.add_part(
                            commands, meshes, &hair_mat, root,
                            Shape::capsule(0.14, 0.54, ShapeDetail::Medium),
                            Transform::from_xyz(0.0, head_y + 0.14, -0.58)
                                .with_rotation(Quat::from_euler(EulerRot::XZY, -35.0_f32.to_radians(), 0.0, 0.0)),
                        );
                    }
                    HairStyle::TwinBuns => {
                        for side in [-1, 1] {
                            self.add_part(
                                commands, meshes, &hair_mat, root,
                                Shape::sphere(0.16, ShapeDetail::High),
                                Transform::from_xyz(side as f32 * 0.36, head_y + 0.42, -0.06),
                            );
                        }
                    }
                    HairStyle::Bald => {}
                }
            }
        }
    }

    // ─── Facial Hair ───

    fn build_facial_hair(
        &self,
        commands: &mut Commands,
        meshes: &mut Assets<bevy::render::mesh::Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        root: Entity,
    ) {
        match self.facial_hair {
            FacialHairStyle::None => {}
            FacialHairStyle::Stubble => {
                let stubble_mat = materials.add(transparent(
                    self.hair_dark[0] * 0.9,
                    self.hair_dark[1] * 0.9,
                    self.hair_dark[2] * 0.9,
                    0.4,
                ));
                self.add_part(
                    commands, meshes, &stubble_mat, root,
                    Shape::cuboid(Vec3::new(0.34, 0.18, 0.06), 0.5, ShapeDetail::Medium),
                    Transform::from_xyz(0.0, 0.80, 0.39),
                );
            }
            FacialHairStyle::ClassicMustache => {
                let f_mat = materials.add(opaque(
                    self.hair_dark[0], self.hair_dark[1], self.hair_dark[2],
                ));
                self.add_part(
                    commands, meshes, &f_mat, root,
                    Shape::ellipsoid(Vec3::new(0.20, 0.05, 0.05), ShapeDetail::Medium),
                    Transform::from_xyz(0.0, 0.88, 0.41),
                );
            }
            FacialHairStyle::Goatee => {
                let f_mat = materials.add(opaque(
                    self.hair_dark[0], self.hair_dark[1], self.hair_dark[2],
                ));
                self.add_part(
                    commands, meshes, &f_mat, root,
                    Shape::cone(0.08, 0.16, ShapeDetail::Medium),
                    Transform::from_xyz(0.0, 0.74, 0.41)
                        .with_rotation(Quat::from_euler(EulerRot::XZY, 15.0_f32.to_radians(), 0.0, 0.0)),
                );
            }
            FacialHairStyle::FullBeard => {
                let f_mat = materials.add(opaque(
                    self.hair_dark[0], self.hair_dark[1], self.hair_dark[2],
                ));
                self.add_part(
                    commands, meshes, &f_mat, root,
                    Shape::cuboid(Vec3::new(0.38, 0.24, 0.20), 0.4, ShapeDetail::High),
                    Transform::from_xyz(0.0, 0.78, 0.37),
                );
            }
        }
    }

    // ─── Glasses ───

    fn build_glasses(
        &self,
        commands: &mut Commands,
        meshes: &mut Assets<bevy::render::mesh::Mesh>,
        materials: &mut Assets<GradMeshMaterial>,
        root: Entity,
    ) {
        let g_col = [0.15, 0.15, 0.15];
        let frame_mat = materials.add(opaque(g_col[0], g_col[1], g_col[2]));
        let eye_spacing = 0.22;
        let eye_height = 1.05;
        let z_fwd = 0.43;

        match self.glasses {
            GlassesStyle::None => {}
            GlassesStyle::RoundFrames => {
                for side in [-1, 1] {
                    self.add_part(
                        commands, meshes, &frame_mat, root,
                        Shape::torus(0.11, 0.02, 0.0, 2.0 * std::f32::consts::PI),
                        Transform::from_xyz(side as f32 * eye_spacing, eye_height, z_fwd)
                            .with_rotation(Quat::from_euler(EulerRot::XZY, std::f32::consts::PI / 2.0, 0.0, 0.0)),
                    );
                }
                // Bridge
                self.add_part(
                    commands, meshes, &frame_mat, root,
                    Shape::cuboid(Vec3::new(eye_spacing * 1.2, 0.02, 0.02), 0.5, ShapeDetail::Medium),
                    Transform::from_xyz(0.0, eye_height, z_fwd),
                );
            }
            GlassesStyle::SquareFrames => {
                for side in [-1, 1] {
                    self.add_part(
                        commands, meshes, &frame_mat, root,
                        Shape::cuboid(Vec3::new(0.24, 0.18, 0.03), 0.2, ShapeDetail::Medium),
                        Transform::from_xyz(side as f32 * eye_spacing, eye_height, z_fwd),
                    );
                }
            }
            GlassesStyle::Sunglasses => {
                let lens_mat = materials.add(transparent(0.1, 0.1, 0.12, 0.92));
                for side in [-1, 1] {
                    self.add_part(
                        commands, meshes, &lens_mat, root,
                        Shape::cuboid(Vec3::new(0.28, 0.18, 0.04), 0.3, ShapeDetail::Medium),
                        Transform::from_xyz(side as f32 * eye_spacing, eye_height, z_fwd),
                    );
                }
            }
            GlassesStyle::Headband => {
                self.add_part(
                    commands, meshes, &frame_mat, root,
                    Shape::torus(0.44, 0.06, 0.0, 2.0 * std::f32::consts::PI),
                    Transform::from_xyz(0.0, eye_height + 0.22, 0.0)
                        .with_rotation(Quat::from_euler(EulerRot::XZY, 80.0_f32.to_radians(), 0.0, 0.0)),
                );
            }
        }
    }

    // ─── Helper ───

    fn add_part(
        &self,
        commands: &mut Commands,
        meshes: &mut Assets<bevy::render::mesh::Mesh>,
        material: &Handle<GradMeshMaterial>,
        parent: Entity,
        shape: Shape,
        transform: Transform,
    ) {
        let mesh_handle = meshes.add(shape.build());
        commands.entity(parent).with_children(|parent| {
            parent.spawn((
                Mesh3d(mesh_handle),
                MeshMaterial3d(material.clone()),
                transform,
            ));
        });
    }
}
