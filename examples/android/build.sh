#!/usr/bin/env bash
# Build, package, sign and (optionally) install the Android test app.
#
#   examples/android/build.sh            # build + package -> target/gradmesh_android.apk
#   examples/android/build.sh install    # ... then adb install, launch, and screenshot
#
# Requirements: cargo-ndk, rustup target aarch64-linux-android, Android SDK + NDK.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK="${ANDROID_HOME:-$HOME/tools/android-sdk}"
export ANDROID_HOME="$SDK"
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$(ls -d "$SDK"/ndk/* | sort | tail -n1)}"
BUILD_TOOLS="${BUILD_TOOLS:-$(ls -d "$SDK"/build-tools/* | sort -V | tail -n1)}"
PLATFORM_JAR="$(ls -d "$SDK"/platforms/android-* | sort -V | tail -n1)/android.jar"
ADB="$SDK/platform-tools/adb"
KEYSTORE="${KEYSTORE:-$HOME/.android/debug.keystore}"
OUT="$HERE/target"
APK="$OUT/gradmesh_android.apk"
PACKAGE=org.gradmesh.standalone

cd "$HERE"
cargo ndk -t arm64-v8a -o "$OUT/jniLibs" build --release

rm -rf "$OUT/apk" && mkdir -p "$OUT/apk"
"$BUILD_TOOLS/aapt2" link \
    --manifest AndroidManifest.xml \
    -I "$PLATFORM_JAR" \
    --min-sdk-version 28 --target-sdk-version 33 \
    -o "$OUT/apk/unaligned-nolib.apk"

mkdir -p "$OUT/apk/lib/arm64-v8a"
cp "$OUT/jniLibs/arm64-v8a/libgradmesh_android.so" "$OUT/apk/lib/arm64-v8a/"
cp "$OUT/apk/unaligned-nolib.apk" "$OUT/apk/unaligned.apk"
(cd "$OUT/apk" && zip -q -r unaligned.apk lib)

"$BUILD_TOOLS/zipalign" -f -p 4 "$OUT/apk/unaligned.apk" "$OUT/apk/aligned.apk"
"$BUILD_TOOLS/apksigner" sign \
    --ks "$KEYSTORE" --ks-pass pass:android --key-pass pass:android \
    --out "$APK" "$OUT/apk/aligned.apk"
echo "Built $APK"

if [ "${1:-}" = "install" ]; then
    "$ADB" install -r "$APK"
    # A sleeping or locked phone captures as a black image.
    "$ADB" shell input keyevent KEYCODE_WAKEUP
    "$ADB" shell wm dismiss-keyguard
    "$ADB" shell am force-stop "$PACKAGE"
    "$ADB" logcat -c
    "$ADB" shell am start -n "$PACKAGE/android.app.NativeActivity"
    sleep "${WAIT:-15}"
    # `exec-out screencap` can return black on Vulkan surfaces; go through /sdcard.
    "$ADB" shell screencap -p /sdcard/gradmesh_screenshot.png
    "$ADB" pull /sdcard/gradmesh_screenshot.png "$OUT/device_screenshot.png"
    "$ADB" logcat -d > "$OUT/device_logcat.txt" || true
    echo "Screenshot: $OUT/device_screenshot.png  Logcat: $OUT/device_logcat.txt"
fi
