//! World-space labels — positions UI text over 3D world positions.
//!
//! Shared across examples to avoid duplicating the `WorldLabel` component
//! and `position_labels` system.

use bevy::prelude::*;

/// Marks a UI text node that should follow a world-space position.
#[derive(Component)]
pub struct WorldLabel(pub Vec3);

/// System that projects world positions to screen space and updates label positions.
///
/// Query all `(Camera3d, GlobalTransform)` and all `(WorldLabel, Node)` entities.
/// Labels outside the viewport are hidden.
pub fn position_labels(
    camera: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    mut labels: Query<(&WorldLabel, &mut Node)>,
) {
    let Ok((camera, camera_transform)) = camera.single() else {
        return;
    };
    for (label, mut node) in &mut labels {
        if let Ok(viewport) = camera.world_to_viewport(camera_transform, label.0) {
            node.left = Val::Px(viewport.x - 60.0);
            node.top = Val::Px(viewport.y - 10.0);
            node.display = Display::Flex;
        } else {
            node.display = Display::None;
        }
    }
}

/// Spawn a world-space label entity.
///
/// # Arguments
/// * `commands` — Bevy Commands
/// * `text` — Label text
/// * `position` — World-space position to track
/// * `font_size` — Font size in pixels (default 14.0)
/// * `width` — Node width in pixels (default 120.0)
pub fn spawn_label(commands: &mut Commands, text: &str, position: Vec3) {
    spawn_label_with_size(commands, text, position, 14.0, 120.0);
}

/// Spawn a world-space label entity with custom size.
pub fn spawn_label_with_size(
    commands: &mut Commands,
    text: &str,
    position: Vec3,
    font_size: f32,
    width: f32,
) {
    commands.spawn((
        Text::new(text.to_string()),
        TextFont {
            font_size: FontSize::Px(font_size),
            ..default()
        },
        TextColor(Color::srgb(0.88, 0.91, 0.98)),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(0.0),
            top: Val::Px(0.0),
            width: Val::Px(width),
            height: Val::Px(20.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        WorldLabel(position),
    ));
}
