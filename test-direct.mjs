#!/usr/bin/env node
/**
 * Direct test of WebGPU engine - bypasses chat demo entirely
 */

import { createWebGPUEngine } from './dist/engine/webgpu-engine.js';
import { resolve } from 'path';

const MODEL_PATH = process.argv[2] || resolve('../spitfire-chat-demo/models/qwen2.5-coder-3b-instruct.gguf');

async function main() {
  console.log('=== Direct WebGPU Engine Test ===\n');
  console.log('Model:', MODEL_PATH);

  const engine = createWebGPUEngine();

  try {
    console.log('\nInitializing WebGPU...');
    await engine.init();

    console.log('Loading model...');
    await engine.loadModel(MODEL_PATH, { contextLength: 512 });
    console.log('Model loaded.\n');

    // Simple prompt
    const prompt = 'Hello';
    console.log(`Prompt: "${prompt}"`);
    console.log('Response: ');

    // Stream the response
    let tokenCount = 0;
    for await (const token of engine.generateStream(prompt, {
      maxTokens: 50,
      temperature: 0.7,
      topK: 40,
    })) {
      process.stdout.write(token);
      tokenCount++;
    }

    console.log(`\n\n[Generated ${tokenCount} tokens]`);

  } catch (error) {
    console.error('\nError:', error);
  } finally {
    await engine.shutdown();
  }
}

main().catch(console.error);
