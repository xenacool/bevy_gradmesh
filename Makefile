
EXAMPLES = shapes styles character_creator immediate curves polyline pbr_effects kart_racer \
           deformation appearance retained standalone_pipeline advanced_shapes lod_demo culling_demo instanced

.PHONY: $(EXAMPLES) all-examples screenshots clean

all-examples:
	cargo build --examples
shapes:
	cargo run --example shapes

styles:
	cargo run --example styles

character:
	cargo run --example character_creator

immediate:
	cargo run --example immediate

curves:
	cargo run --example curves

polyline:
	cargo run --example polyline

pbr_effects:
	cargo run --example pbr_effects

kart_racer:
	cargo run --example kart_racer

deformation:
	cargo run --example deformation

appearance:
	cargo run --example appearance

retained:
	cargo run --example retained

standalone_pipeline:
	cargo run --example standalone_pipeline

advanced_shapes:
	cargo run --example advanced_shapes

lod_demo:
	cargo run --example lod_demo

culling_demo:
	cargo run --example culling_demo

instanced:
	cargo run --example instanced

SCREENSHOTS = screenshot_shapes.png screenshot_styles.png screenshot_character_creator.png screenshot_immediate.png \
               screenshot_curves.png screenshot_polyline.png screenshot_pbr_effects.png screenshot_kart_racer.png \
               screenshot_deformation.png screenshot_appearance.png screenshot_retained.png \
               screenshot_standalone_pipeline.png screenshot_advanced_shapes.png screenshot_lod_demo.png \
               screenshot_culling_demo.png screenshot_instanced.png

# Detect whether we need a virtual display for screenshot capture
# XVFB_RUN is set to "xvfb-run --auto-servernum --server-args='-screen 0 1280x720x24'" when no display is found
XVFB_RUN := $(shell if [ -z "$$DISPLAY" ] && [ -z "$$WAYLAND_DISPLAY" ] && [ -z "$$WAYLAND_SOCKET" ] && command -v xvfb-run >/dev/null 2>&1; then echo "xvfb-run --auto-servernum --server-args='-screen 0 1280x720x24'"; fi)

define capture_screenshot
	@echo "=== Capturing $(1) screenshot ==="
	$(XVFB_RUN) env DSD_SCREENSHOT=1 timeout 120 cargo run --example $(1) 2>&1 || true
	@if [ -f screenshot_$(1).png ]; then \
		if command -v pngcrush >/dev/null 2>&1; then \
			pngcrush -rem all -reduce screenshot_$(1).png screenshots/screenshot_$(1).png; \
		else \
			mv screenshot_$(1).png screenshots/; \
		fi \
	fi
endef

# Limit parallel build jobs to reduce memory usage during linking.
# Each Bevy example linker process can consume ~2.3 GB RAM; with 16 cores,
# unrestricted parallelism can exceed 30 GB. Capping at 4 jobs keeps it ~9 GB.
BUILD_JOBS ?= 4

screenshots:
	@echo "=== Building all examples (jobs=$(BUILD_JOBS)) ==="
	CARGO_BUILD_JOBS=$(BUILD_JOBS) cargo build --examples
	@mkdir -p screenshots
	@echo ""
	$(call capture_screenshot,shapes)
	@echo ""
	$(call capture_screenshot,styles)
	@echo ""
	$(call capture_screenshot,character_creator)
	@echo ""
	$(call capture_screenshot,immediate)
	@echo ""
	$(call capture_screenshot,curves)
	@echo ""
	$(call capture_screenshot,polyline)
	@echo ""
	$(call capture_screenshot,pbr_effects)
	@echo ""
	$(call capture_screenshot,kart_racer)
	@echo ""
	$(call capture_screenshot,deformation)
	@echo ""
	$(call capture_screenshot,appearance)
	@echo ""
	$(call capture_screenshot,retained)
	@echo ""
	$(call capture_screenshot,standalone_pipeline)
	@echo ""
	$(call capture_screenshot,advanced_shapes)
	@echo ""
	$(call capture_screenshot,lod_demo)
	@echo ""
	$(call capture_screenshot,culling_demo)
	@echo ""
	$(call capture_screenshot,instanced)
	@echo ""
	@echo "=== Screenshots saved to screenshots/ ==="
	@ls -la screenshots/

clean:
	cargo clean
