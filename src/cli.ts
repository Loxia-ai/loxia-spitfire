#!/usr/bin/env node
/**
 * Spitfire CLI
 * Simple command-line interface for testing the module
 */

import { Spitfire, SpitfireServer, VERSION } from './index.js';
import type { Message } from './types/index.js';

const args = process.argv.slice(2);
const command = args[0];

function printHelp(): void {
  console.log(`
Spitfire v${VERSION} - Node.js LLM Runner

Usage: spitfire <command> [options]

Commands:
  serve [--port PORT]     Start the HTTP server (default port: 11434)
  list                    List installed models
  run <model> [prompt]    Run a model with optional prompt
  chat <model>            Start interactive chat with a model
  ps                      Show running models
  help                    Show this help message

Examples:
  spitfire serve --port 8080
  spitfire list
  spitfire run llama3.2 "Hello, world!"
  spitfire chat llama3.2
`);
}

async function serve(): Promise<void> {
  const portIndex = args.indexOf('--port');
  const port = portIndex !== -1 ? parseInt(args[portIndex + 1], 10) : 11434;

  console.log(`Starting Spitfire server on port ${port}...`);

  const server = new SpitfireServer({ port });

  // Handle events
  const scheduler = server.getScheduler();
  scheduler.on('modelLoading', ({ name }) => {
    console.log(`Loading model: ${name}`);
  });
  scheduler.on('modelLoaded', ({ name }) => {
    console.log(`Model loaded: ${name}`);
  });
  scheduler.on('modelUnloaded', ({ name }) => {
    console.log(`Model unloaded: ${name}`);
  });

  try {
    const address = await server.start();
    console.log(`Spitfire server running at ${address}`);
    console.log('Press Ctrl+C to stop');

    // Handle shutdown
    process.on('SIGINT', async () => {
      console.log('\nShutting down...');
      await server.stop();
      process.exit(0);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

async function list(): Promise<void> {
  const spitfire = new Spitfire();

  try {
    const response = await spitfire.list();

    if (response.models.length === 0) {
      console.log('No models installed');
      console.log('\nModels directory: ~/.ollama/models/');
      return;
    }

    console.log('Installed models:\n');
    for (const model of response.models) {
      const sizeGB = (model.size / (1024 * 1024 * 1024)).toFixed(2);
      console.log(`  ${model.name}`);
      console.log(`    Size: ${sizeGB} GB`);
      console.log(`    Modified: ${model.modifiedAt}`);
      console.log('');
    }
  } catch (error) {
    console.error('Error listing models:', error);
  } finally {
    await spitfire.shutdown();
  }
}

async function run(): Promise<void> {
  const model = args[1];
  const prompt = args.slice(2).join(' ') || 'Hello!';

  if (!model) {
    console.error('Error: Model name required');
    console.error('Usage: spitfire run <model> [prompt]');
    process.exit(1);
  }

  const spitfire = new Spitfire();

  try {
    console.log(`Loading ${model}...`);

    for await (const chunk of spitfire.generateStream({
      model,
      prompt,
    })) {
      process.stdout.write(chunk.response);

      if (chunk.done) {
        console.log('\n');
        const tokensPerSec = chunk.evalCount && chunk.evalDuration
          ? ((chunk.evalCount / chunk.evalDuration) * 1_000_000_000).toFixed(2)
          : 'N/A';
        console.log(`Tokens: ${chunk.evalCount || 'N/A'}, Speed: ${tokensPerSec} tok/s`);
      }
    }
  } catch (error) {
    console.error('Error:', error instanceof Error ? error.message : error);
  } finally {
    await spitfire.shutdown();
  }
}

async function chat(): Promise<void> {
  const model = args[1];

  if (!model) {
    console.error('Error: Model name required');
    console.error('Usage: spitfire chat <model>');
    process.exit(1);
  }

  const readline = await import('readline');
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const spitfire = new Spitfire();
  const messages: Message[] = [];

  console.log(`Starting chat with ${model}...`);
  console.log('Type "exit" or press Ctrl+C to quit\n');

  const askQuestion = (): void => {
    rl.question('You: ', async (input) => {
      const trimmed = input.trim();

      if (trimmed.toLowerCase() === 'exit' || trimmed.toLowerCase() === 'quit') {
        console.log('Goodbye!');
        await spitfire.shutdown();
        rl.close();
        return;
      }

      if (!trimmed) {
        askQuestion();
        return;
      }

      messages.push({ role: 'user', content: trimmed });

      process.stdout.write('Assistant: ');

      try {
        let fullResponse = '';

        for await (const chunk of spitfire.chatStream({
          model,
          messages,
        })) {
          process.stdout.write(chunk.message.content);
          fullResponse += chunk.message.content;
        }

        console.log('\n');
        messages.push({ role: 'assistant', content: fullResponse });
      } catch (error) {
        console.error('\nError:', error instanceof Error ? error.message : error);
      }

      askQuestion();
    });
  };

  // Handle Ctrl+C
  rl.on('close', async () => {
    console.log('\nGoodbye!');
    await spitfire.shutdown();
    process.exit(0);
  });

  askQuestion();
}

async function ps(): Promise<void> {
  const spitfire = new Spitfire();

  try {
    const response = spitfire.ps();

    if (response.models.length === 0) {
      console.log('No models currently running');
      return;
    }

    console.log('Running models:\n');
    for (const model of response.models) {
      console.log(`  ${model.name}`);
      console.log(`    Expires: ${model.expiresAt}`);
      console.log('');
    }
  } finally {
    await spitfire.shutdown();
  }
}

async function main(): Promise<void> {
  switch (command) {
    case 'serve':
      await serve();
      break;
    case 'list':
    case 'ls':
      await list();
      break;
    case 'run':
    case 'generate':
      await run();
      break;
    case 'chat':
      await chat();
      break;
    case 'ps':
      await ps();
      break;
    case 'help':
    case '--help':
    case '-h':
    case undefined:
      printHelp();
      break;
    default:
      console.error(`Unknown command: ${command}`);
      printHelp();
      process.exit(1);
  }
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
