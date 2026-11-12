//! Headless ECS tests: plugin registration, the shared mesh cache and retained-slot replay.
//!
//! The app is Bevy's default plugins with the GPU backend switched off and no window, so
//! these tests run on CI without a display or an adapter.

use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::settings::WgpuSettings;
use bevy::window::ExitCondition;
use bevy::winit::WinitPlugin;
use bevy_gradmesh::{
    Deformation, GradMeshPlugin, InstancedStats, MeshCache, Shape, ShapeDetail,
};

/// Builds an app with `plugin` on top of the default plugins, without renderer or window.
fn headless_app(plugin: impl Plugin) -> App {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(RenderPlugin {
                render_creation: WgpuSettings {
                    backends: None,
                    ..default()
                }
                .into(),
                ..default()
            })
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .disable::<WinitPlugin>(),
    );
    app.add_plugins(plugin);
    app.finish();
    app.cleanup();
    app
}

fn assert_common_resources(app: &App) {
    assert!(app.world().contains_resource::<MeshCache>());
    assert!(app.world().contains_resource::<InstancedStats>());
    assert!(app.world().contains_resource::<Deformation>());
}

#[test]
fn default_plugin_registers_the_shared_resources() {
    let app = headless_app(GradMeshPlugin);
    assert_common_resources(&app);
    #[cfg(feature = "pbr")]
    assert!(
        app.world()
            .contains_resource::<bevy_gradmesh::GradMeshMaterialPool>()
    );
}

#[test]
fn mobile_plugin_registers_the_shared_resources_without_the_pbr_pool() {
    let app = headless_app(GradMeshPlugin::for_mobile());
    assert_common_resources(&app);
    #[cfg(feature = "pbr")]
    assert!(
        !app.world()
            .contains_resource::<bevy_gradmesh::GradMeshMaterialPool>(),
        "the PBR-free plugin must not register the material pool"
    );
}

#[test]
fn mesh_cache_builds_once_and_publishes_its_stats() {
    let mut app = headless_app(GradMeshPlugin::for_mobile());
    let shape = Shape::sphere(1.0, ShapeDetail::Low);

    {
        let mut cache = app.world_mut().resource_mut::<MeshCache>();
        cache.get_or_build(&shape);
        cache.get_or_build(&shape);
        cache.get_or_build(&Shape::sphere(2.0, ShapeDetail::Low));
    }
    app.update();

    let stats = app.world().resource::<InstancedStats>();
    assert_eq!(stats.cache_misses, 2, "two distinct shapes were built");
    assert_eq!(stats.cache_hits, 1, "the repeated shape was served from the cache");
    assert_eq!(stats.cached_meshes, 2);
    assert_eq!(stats.mesh_builds, 2);
    assert!(stats.cached_mesh_bytes > 0);

    // The per-frame build counter resets once it has been published.
    app.update();
    assert_eq!(app.world().resource::<InstancedStats>().mesh_builds, 0);
}

#[test]
fn mesh_cache_hands_out_one_asset_per_shape() {
    let mut app = headless_app(GradMeshPlugin::for_mobile());
    app.world_mut()
        .run_system_once(|mut cache: ResMut<MeshCache>, mut meshes: ResMut<Assets<Mesh>>| {
            let shape = Shape::cuboid(Vec3::ONE, 0.0, ShapeDetail::Low);
            let a = cache.get_or_build_handle(&shape, &mut meshes);
            let b = cache.get_or_build_handle(&shape, &mut meshes);
            assert_eq!(a, b);
        })
        .unwrap();
}

#[cfg(feature = "pbr")]
mod retained {
    use super::*;
    use bevy_gradmesh::{GradMeshHandle, InstancedBatchData, RetainedWorld};

    #[derive(Resource)]
    struct Retained {
        world: RetainedWorld,
        handles: Vec<GradMeshHandle>,
    }

    fn batch_entities(app: &mut App) -> usize {
        app.world_mut()
            .query::<&InstancedBatchData>()
            .iter(app.world())
            .count()
    }

    #[test]
    fn flush_replays_every_live_slot_into_shared_batches() {
        let mut app = headless_app(GradMeshPlugin);
        app.world_mut()
            .run_system_once(
                |mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>| {
                    let mut world = RetainedWorld::new();
                    let handles: Vec<_> = (0..3)
                        .map(|i| {
                            world.sphere(
                                &mut commands,
                                &mut meshes,
                                Vec3::X * i as f32,
                                0.5,
                                ShapeDetail::Low,
                                [1.0, 0.0, 0.0, 1.0],
                            )
                        })
                        .collect();
                    assert_eq!(world.len(), 3);
                    assert!(world.is_dirty());
                    commands.insert_resource(Retained { world, handles });
                },
            )
            .unwrap();

        app.world_mut()
            .run_system_once(
                |mut commands: Commands,
                 meshes: ResMut<Assets<Mesh>>,
                 mut retained: ResMut<Retained>| {
                    let spawned = retained.world.flush(&mut commands, meshes);
                    assert_eq!(spawned.len(), 1, "three equal spheres share one batch");
                    assert_eq!(retained.world.batch_count(), 1);
                    assert_eq!(retained.world.total_instances(), 3);
                    assert!(!retained.world.is_dirty());
                },
            )
            .unwrap();
        assert!(batch_entities(&mut app) >= 1);
    }

    #[test]
    fn removed_and_stale_handles_drop_out_of_the_next_flush() {
        let mut app = headless_app(GradMeshPlugin);
        app.world_mut()
            .run_system_once(
                |mut commands: Commands,
                 mut meshes: ResMut<Assets<Mesh>>| {
                    let mut world = RetainedWorld::new();
                    let handles: Vec<_> = (0..4)
                        .map(|i| {
                            world.sphere(
                                &mut commands,
                                &mut meshes,
                                Vec3::X * i as f32,
                                0.5,
                                ShapeDetail::Low,
                                [0.0, 1.0, 0.0, 1.0],
                            )
                        })
                        .collect();
                    commands.insert_resource(Retained { world, handles });
                },
            )
            .unwrap();

        app.world_mut()
            .run_system_once(
                |mut commands: Commands,
                 meshes: ResMut<Assets<Mesh>>,
                 mut retained: ResMut<Retained>| {
                    let removed = retained.handles[1];
                    let moved = retained.handles[2];
                    retained.world.remove(removed);
                    retained.world.set_position(moved, Vec3::new(9.0, 0.0, 0.0));
                    // Updating a removed handle is a harmless no-op, not a panic.
                    retained.world.set_position(removed, Vec3::ZERO);
                    retained.world.remove(removed);
                    assert!(retained.world.is_dirty());
                    assert_eq!(retained.world.len(), 3);

                    retained.world.flush(&mut commands, meshes);
                    assert_eq!(retained.world.total_instances(), 3);
                    assert_eq!(retained.world.batch_count(), 1);
                },
            )
            .unwrap();
    }

    #[test]
    fn a_reused_slot_does_not_revive_the_old_handle() {
        let mut app = headless_app(GradMeshPlugin);
        app.world_mut()
            .run_system_once(
                |mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>| {
                    let mut world = RetainedWorld::new();
                    let old = world.sphere(
                        &mut commands,
                        &mut meshes,
                        Vec3::ZERO,
                        0.5,
                        ShapeDetail::Low,
                        [1.0; 4],
                    );
                    world.remove(old);
                    let new = world.sphere(
                        &mut commands,
                        &mut meshes,
                        Vec3::ZERO,
                        0.5,
                        ShapeDetail::Low,
                        [1.0; 4],
                    );
                    assert_ne!(old, new, "a reused slot gets a new generation");
                    // The stale handle must not move the new shape: only one instance lives.
                    world.set_position(old, Vec3::splat(100.0));
                    assert_eq!(world.len(), 1);
                    world.remove(old);
                    assert_eq!(world.len(), 1, "removing a stale handle removes nothing");
                },
            )
            .unwrap();
    }
}
