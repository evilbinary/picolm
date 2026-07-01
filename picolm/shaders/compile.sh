#!/bin/sh
# Compile GLSL compute shaders to SPIR-V
# Requires glslangValidator from Vulkan SDK
# Usage: GLSLANG=path/to/glslangValidator ./compile.sh
GLSLANG=${GLSLANG:-glslangValidator}
mkdir -p spv
$GLSLANG -V --target-env vulkan1.2 -o spv/q4_k.spv q4_k.comp
$GLSLANG -V --target-env vulkan1.2 -o spv/q6_k.spv q6_k.comp
$GLSLANG -V --target-env vulkan1.2 -o spv/q8_0.spv q8_0.comp
echo "Shaders compiled to spv/"
