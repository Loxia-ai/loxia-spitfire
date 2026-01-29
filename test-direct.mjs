#!/usr/bin/env node
/**
 * Direct test of WebGPU engine - bypasses chat demo entirely
 *
 * Usage:
 *   node test-direct.mjs [model-path] [--quantized]
 *
 * Options:
 *   --quantized  Keep weights in Q4_K/Q8_0 format (reduced VRAM mode)
 *                Without this flag, weights are dequantized to f32
 */

import { createWebGPUEngine } from './dist/engine/webgpu-engine.js';
import { resolve } from 'path';

// Parse command line args
const args = process.argv.slice(2);
const keepQuantized = args.includes('--quantized');
const modelArg = args.find(arg => !arg.startsWith('--'));
const MODEL_PATH = modelArg || resolve('../spitfire-chat-demo/models/qwen2.5-coder-3b-instruct.gguf');

async function main() {
  console.log('=== Direct WebGPU Engine Test ===\n');
  console.log('Model:', MODEL_PATH);
  console.log('Mode:', keepQuantized ? 'QUANTIZED (reduced VRAM)' : 'F32 (full precision)');

  const engine = createWebGPUEngine();

  try {
    console.log('\nInitializing WebGPU...');
    await engine.init();

    console.log('Loading model...');
    const loadStart = performance.now();
    await engine.loadModel(MODEL_PATH, {
      contextLength: 512,
      keepQuantized,  // Keep weights in Q4_K/Q8_0 format for reduced VRAM
    });
    const loadTime = ((performance.now() - loadStart) / 1000).toFixed(2);
    console.log(`Model loaded in ${loadTime}s.\n`);

    // Simple prompt
    const prompt = 'Write a short poem about coding.';
    console.log(`Prompt: "${prompt}"`);
    console.log('Response: ');

    // Stream the response with timing
    let tokenCount = 0;
    const genStart = performance.now();
    let firstTokenTime = null;

    for await (const token of engine.generateStream(prompt, {
      maxTokens: 100,
      temperature: 0.7,
      topK: 40,
    })) {
      if (firstTokenTime === null) {
        firstTokenTime = performance.now() - genStart;
      }
      process.stdout.write(token);
      tokenCount++;
    }

    const totalTime = performance.now() - genStart;
    const tokensPerSec = (tokenCount / (totalTime / 1000)).toFixed(2);

    console.log('\n');
    console.log('--- Performance ---');
    console.log(`Time to first token: ${firstTokenTime?.toFixed(0)}ms`);
    console.log(`Total tokens: ${tokenCount}`);
    console.log(`Total time: ${(totalTime / 1000).toFixed(2)}s`);
    console.log(`Speed: ${tokensPerSec} tokens/sec`);
    console.log(`Mode: ${keepQuantized ? 'QUANTIZED' : 'F32'}`);

  } catch (error) {
    console.error('\nError:', error);
  } finally {
    await engine.shutdown();
  }
}

main().catch(console.error);
