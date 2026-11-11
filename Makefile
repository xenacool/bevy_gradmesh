EXAMPLES = shapes styles mix_spaces curves pbr_effects arbitrary_shapes composed_shapes sdf_dash_gallery sweep terrain road character_creator immediate polyline deformation retained gradient_2d sdf_2d standalone instanced appearance lerp lerp_2d

# Examples that change over time are recorded as animated GIF; the rest are still PNGs.
STILLS = shapes styles mix_spaces curves pbr_effects arbitrary_shapes composed_shapes sdf_dash_gallery sweep terrain road
# Animations are downscaled to keep each file well under a megabyte.
ANIMATION_WIDTH = 480
ANIMATIONS = character_creator immediate polyline deformation retained gradient_2d sdf_2d standalone instanced appearance lerp lerp_2d

# Examples that require the pbr feature (required-features in Cargo.toml).
# The capture macros add --features pbr for these so screenshot capture doesn't fail.
# Each gets a flag variable so exact matching works (findstring is substring-only).
pbr_example_character_creator = 1
pbr_example_immediate = 1
pbr_example_instanced = 1
pbr_example_pbr_effects = 1
pbr_example_shapes = 1
pbr_example_styles = 1
pbr_example_sweep = 1
pbr_example_terrain = 1
pbr_example_road = 1
pbr_example_composed_shapes = 1
pbr_example_appearance = 1
pbr_example_deformation = 1
pbr_example_retained = 1

.PHONY: $(EXAMPLES) all-examples screenshots clean

all-examples:
	cargo build --examples

$(EXAMPLES):
	cargo run --example $@

# Detect whether we need a virtual display for screenshot capture
# XVFB_RUN is set to "xvfb-run --auto-servernum --server-args='-screen 0 1280x720x24'" when no display is found
XVFB_RUN := $(shell if [ -z "$$DISPLAY" ] && [ -z "$$WAYLAND_DISPLAY" ] && [ -z "$$WAYLAND_SOCKET" ] && command -v xvfb-run >/dev/null 2>&1; then echo "xvfb-run --auto-servernum --server-args='-screen 0 1280x720x24'"; fi)

# Helper: returns "--features pbr" if pbr_example_$(1) is set, else empty.
# Uses per-example flag variables for exact matching (findstring is substring-only).
define pbr_flag
$(if $(pbr_example_$(1)),--features pbr,)
endef

# Arguments: example name. Still screenshot saved to screenshots/screenshot_<name>.png.
define capture_screenshot
	@echo "=== Capturing $(1) screenshot ==="
	$(XVFB_RUN) env DSD_SCREENSHOT=1 timeout 120 cargo run --example $(1) $(call pbr_flag,$(1)) 2>&1 || true
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
	$(XVFB_RUN) env DSD_SCREENSHOT=1 timeout 600 cargo run --example $(1) $(call pbr_flag,$(1)) 2>&1 || true
	@if ls frames/screenshot_$(1)_frame_*.png 1>/dev/null 2>&1; then \
		echo "Encoding frames to GIF..."; \
		ffmpeg -y -framerate 30 -i frames/screenshot_$(1)_frame_%04d.png \
			-vf "scale=$(ANIMATION_WIDTH):-2:flags=lanczos" \
			-vcodec gif -loop 0 \
			screenshots/screenshot_$(1).gif 2>/dev/null && \
		echo "Saved screenshots/screenshot_$(1).gif" || \
		echo "Warning: ffmpeg GIF encoding failed for $(1)"; \
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
	$(call capture_animation,lerp)
	@echo ""
	$(call capture_animation,lerp_2d)
	@echo ""
	@echo "=== Screenshots saved to screenshots/ ==="
	@ls -la screenshots/

clean:
	cargo clean

# Release target: sanity-checks all screenshots and packages them into a zip for GitHub Releases.
# Use this locally before uploading, and in CI to gate releases.
VERSION := $(shell grep '^version =' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')

release-zip: screenshots
	@echo "=== Validating screenshots ==="
	@for f in screenshots/*.png; do \
		if [ -f "$$f" ]; then \
			if ! file "$$f" | grep -q "PNG"; then \
				echo "ERROR: $$f is not a valid PNG"; exit 1; \
			fi; \
		fi; \
	done
	@for f in screenshots/*.gif; do \
		if [ -f "$$f" ]; then \
			if ! file "$$f" | grep -q "GIF"; then \
				echo "ERROR: $$f is not a valid GIF"; exit 1; \
			fi; \
		fi; \
	done
	@count=$$(ls screenshots/*.png screenshots/*.gif 2>/dev/null | wc -l); \
	echo "Found $$count screenshot files"
	@rm -f screenshots-v$(VERSION).zip
	@cd screenshots && zip -r ../screenshots-v$(VERSION).zip *.png *.gif 2>/dev/null || \
		(cd screenshots && zip ../screenshots-v$(VERSION).zip *.png *.gif)
	@echo "Created screenshots-v$(VERSION).zip ($$(du -h ../screenshots-v$(VERSION).zip | cut -f1))"

release: release-zip
	@echo ""
	@echo "Upload to GitHub with:"
	@echo "  gh release create v$(VERSION) screenshots-v$(VERSION).zip --title 'v$(VERSION)' --notes '...'"
	@echo ""
	@echo "Or amend an existing release:"
	@echo "  gh release upload v$(VERSION) screenshots-v$(VERSION).zip --clobber"
