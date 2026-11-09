EXAMPLES = shapes styles mix_spaces curves pbr_effects arbitrary_shapes composed_shapes sdf_dash_gallery sweep terrain road character_creator immediate polyline deformation retained gradient_2d sdf_2d standalone instanced appearance

# Examples that change over time are recorded as animated WebP; the rest are still PNGs.
STILLS = shapes styles mix_spaces curves pbr_effects arbitrary_shapes composed_shapes sdf_dash_gallery sweep terrain road
# Animations are lossy and downscaled to keep each file well under a megabyte.
ANIMATION_WIDTH = 480
ANIMATION_QUALITY = 30
ANIMATIONS = character_creator immediate polyline deformation retained gradient_2d sdf_2d standalone instanced appearance

.PHONY: $(EXAMPLES) all-examples screenshots clean

all-examples:
	cargo build --examples

$(EXAMPLES):
	cargo run --example $@

# Detect whether we need a virtual display for screenshot capture
# XVFB_RUN is set to "xvfb-run --auto-servernum --server-args='-screen 0 1280x720x24'" when no display is found
XVFB_RUN := $(shell if [ -z "$$DISPLAY" ] && [ -z "$$WAYLAND_DISPLAY" ] && [ -z "$$WAYLAND_SOCKET" ] && command -v xvfb-run >/dev/null 2>&1; then echo "xvfb-run --auto-servernum --server-args='-screen 0 1280x720x24'"; fi)

# Arguments: example name. Still screenshot saved to screenshots/screenshot_<name>.png.
define capture_screenshot
	@echo "=== Capturing $(1) screenshot ==="
	$(XVFB_RUN) env DSD_SCREENSHOT=1 timeout 120 cargo run --example $(1) 2>&1 || true
	@if [ -f screenshot_$(1).png ]; then \
		if command -v pngcrush >/dev/null 2>&1; then \
			pngcrush -rem all -reduce screenshot_$(1).png screenshots/screenshot_$(1).png; \
			rm -f screenshot_$(1).png; \
		else \
			mv screenshot_$(1).png screenshots/; \
		fi \
	else \
		echo "Warning: no screenshot produced for $(1)"; \
	fi
endef

# Arguments: example name. Frames are captured on a fixed 30fps clock (one app tick is one
# frame), so they are encoded at 30fps and the animation plays back in real time. Frames are
# saved in the gitignored `frames/` directory and removed after encoding.
define capture_animation
	@echo "=== Capturing $(1) animation ==="
	@mkdir -p frames
	@rm -f frames/screenshot_$(1)_frame_*.png
	$(XVFB_RUN) env DSD_SCREENSHOT=1 timeout 600 cargo run --example $(1) 2>&1 || true
	@if ls frames/screenshot_$(1)_frame_*.png 1>/dev/null 2>&1; then \
		echo "Encoding frames to WebP..."; \
		ffmpeg -y -framerate 30 -i frames/screenshot_$(1)_frame_%04d.png \
			-vf "scale=$(ANIMATION_WIDTH):-2:flags=lanczos" \
			-vcodec libwebp -lossless 0 -quality $(ANIMATION_QUALITY) -compression_level 6 -loop 0 \
			screenshots/screenshot_$(1).webp 2>/dev/null && \
		echo "Saved screenshots/screenshot_$(1).webp" || \
		echo "Warning: ffmpeg WebP encoding failed for $(1)"; \
		rm -f frames/screenshot_$(1)_frame_*.png; \
	else \
		echo "Warning: no frames found for $(1)"; \
	fi
endef

# One target per example: `make screenshot-<name>` refreshes just that file.
$(addprefix screenshot-,$(STILLS)): screenshot-%:
	@mkdir -p screenshots
	$(call capture_screenshot,$*)

$(addprefix screenshot-,$(ANIMATIONS)): screenshot-%:
	@mkdir -p screenshots
	$(call capture_animation,$*)

screenshots:
	@echo "=== Building all examples ==="
	cargo build --examples
	@mkdir -p screenshots
	@echo ""
	$(call capture_screenshot,shapes)
	@echo ""
	$(call capture_screenshot,styles)
	@echo ""
	$(call capture_screenshot,mix_spaces)
	@echo ""
	$(call capture_screenshot,curves)
	@echo ""
	$(call capture_screenshot,pbr_effects)
	@echo ""
	$(call capture_screenshot,arbitrary_shapes)
	@echo ""
	$(call capture_screenshot,composed_shapes)
	@echo ""
	$(call capture_screenshot,sdf_dash_gallery)
	@echo ""
	$(call capture_screenshot,sweep)
	@echo ""
	$(call capture_screenshot,terrain)
	@echo ""
	$(call capture_screenshot,road)
	@echo ""
	$(call capture_animation,character_creator)
	@echo ""
	$(call capture_animation,immediate)
	@echo ""
	$(call capture_animation,polyline)
	@echo ""
	$(call capture_animation,deformation)
	@echo ""
	$(call capture_animation,retained)
	@echo ""
	$(call capture_animation,gradient_2d)
	@echo ""
	$(call capture_animation,sdf_2d)
	@echo ""
	$(call capture_animation,standalone)
	@echo ""
	$(call capture_animation,instanced)
	@echo ""
	$(call capture_animation,appearance)
	@echo ""
	@echo "=== Screenshots saved to screenshots/ ==="
	@ls -la screenshots/

clean:
	cargo clean
