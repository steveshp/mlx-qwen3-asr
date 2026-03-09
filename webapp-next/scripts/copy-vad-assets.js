#!/usr/bin/env node
/**
 * Copy ONNX runtime + Silero VAD assets from node_modules to public/
 * so the browser can load them at runtime.
 * Runs automatically via "postinstall" in package.json.
 */
const fs = require('fs');
const path = require('path');

const publicDir = path.join(__dirname, '..', 'public');

const assets = [
  // ONNX Runtime Web
  ...['ort.mjs', 'ort.min.mjs', 'ort.all.mjs', 'ort.all.min.mjs',
    'ort.all.bundle.min.mjs', 'ort.bundle.min.mjs',
    'ort.wasm.mjs', 'ort.wasm.min.mjs', 'ort.wasm.bundle.min.mjs',
    'ort.webgl.mjs', 'ort.webgl.min.mjs',
    'ort.webgpu.mjs', 'ort.webgpu.min.mjs', 'ort.webgpu.bundle.min.mjs',
    'ort.node.min.mjs',
    'ort.jspi.mjs', 'ort.jspi.min.mjs', 'ort.jspi.bundle.min.mjs',
    'ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs',
    'ort-wasm-simd-threaded.jsep.wasm', 'ort-wasm-simd-threaded.jsep.mjs',
    'ort-wasm-simd-threaded.jspi.wasm', 'ort-wasm-simd-threaded.jspi.mjs',
    'ort-wasm-simd-threaded.asyncify.wasm', 'ort-wasm-simd-threaded.asyncify.mjs',
  ].map(f => ({ src: `onnxruntime-web/dist/${f}`, dest: f })),

  // Silero VAD models + worklet
  { src: '@ricky0123/vad-web/dist/silero_vad_legacy.onnx', dest: 'silero_vad_legacy.onnx' },
  { src: '@ricky0123/vad-web/dist/silero_vad_v5.onnx', dest: 'silero_vad_v5.onnx' },
  { src: '@ricky0123/vad-web/dist/vad.worklet.bundle.min.js', dest: 'vad.worklet.bundle.min.js' },
];

let copied = 0;
let skipped = 0;

for (const { src, dest } of assets) {
  const srcPath = path.join(__dirname, '..', 'node_modules', src);
  const destPath = path.join(publicDir, dest);
  if (!fs.existsSync(srcPath)) {
    skipped++;
    continue;
  }
  fs.copyFileSync(srcPath, destPath);
  copied++;
}

console.log(`[copy-vad-assets] ${copied} copied, ${skipped} skipped`);
