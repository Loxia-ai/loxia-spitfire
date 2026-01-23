/**
 * Integration tests for Spitfire
 */

import { Spitfire, SpitfireServer, VERSION } from '../src/index.js';
import { ModelManager } from '../src/model/index.js';
import { RunnerManager } from '../src/runner/index.js';
import { Scheduler } from '../src/scheduler/index.js';
import {
  parseDuration,
  formatDuration,
  formatBytes,
} from '../src/utils/index.js';
import {
  parseModelName,
  formatModelName,
} from '../src/types/model.js';
import type {
  GenerateRequest,
  ChatRequest,
  EmbedRequest,
  Message,
} from '../src/types/index.js';

describe('Module Exports', () => {
  it('should export VERSION', () => {
    expect(VERSION).toBe('0.1.0');
  });

  it('should export Spitfire class', () => {
    expect(Spitfire).toBeDefined();
    expect(typeof Spitfire).toBe('function');
  });

  it('should export SpitfireServer class', () => {
    expect(SpitfireServer).toBeDefined();
    expect(typeof SpitfireServer).toBe('function');
  });

  it('should export ModelManager class', () => {
    expect(ModelManager).toBeDefined();
    expect(typeof ModelManager).toBe('function');
  });

  it('should export RunnerManager class', () => {
    expect(RunnerManager).toBeDefined();
    expect(typeof RunnerManager).toBe('function');
  });

  it('should export Scheduler class', () => {
    expect(Scheduler).toBeDefined();
    expect(typeof Scheduler).toBe('function');
  });
});

describe('Spitfire Class', () => {
  let spitfire: Spitfire;

  beforeEach(() => {
    spitfire = new Spitfire();
  });

  afterEach(async () => {
    await spitfire.shutdown();
  });

  it('should instantiate without options', () => {
    expect(spitfire).toBeInstanceOf(Spitfire);
  });

  it('should instantiate with options', () => {
    const sf = new Spitfire({
      maxLoadedModels: 2,
      defaultKeepAlive: '10m',
    });
    expect(sf).toBeInstanceOf(Spitfire);
    sf.shutdown();
  });

  it('should have list method', async () => {
    const result = await spitfire.list();
    expect(result).toHaveProperty('models');
    expect(Array.isArray(result.models)).toBe(true);
  });

  it('should have ps method', () => {
    const result = spitfire.ps();
    expect(result).toHaveProperty('models');
    expect(Array.isArray(result.models)).toBe(true);
  });

  it('should have exists method', async () => {
    const exists = await spitfire.exists('nonexistent-model');
    expect(typeof exists).toBe('boolean');
    expect(exists).toBe(false);
  });

  it('should have show method', async () => {
    const info = await spitfire.show('nonexistent-model');
    expect(info).toBeNull();
  });

  it('should have generate method', () => {
    expect(typeof spitfire.generate).toBe('function');
  });

  it('should have generateStream method', () => {
    expect(typeof spitfire.generateStream).toBe('function');
  });

  it('should have chat method', () => {
    expect(typeof spitfire.chat).toBe('function');
  });

  it('should have chatStream method', () => {
    expect(typeof spitfire.chatStream).toBe('function');
  });

  it('should have embed method', () => {
    expect(typeof spitfire.embed).toBe('function');
  });

  it('should have preload method', () => {
    expect(typeof spitfire.preload).toBe('function');
  });

  it('should have unload method', () => {
    expect(typeof spitfire.unload).toBe('function');
  });

  it('should have shutdown method', () => {
    expect(typeof spitfire.shutdown).toBe('function');
  });
});

describe('SpitfireServer Class', () => {
  it('should instantiate without options', () => {
    const server = new SpitfireServer();
    expect(server).toBeInstanceOf(SpitfireServer);
  });

  it('should instantiate with options', () => {
    const server = new SpitfireServer({
      host: '0.0.0.0',
      port: 8080,
      cors: true,
    });
    expect(server).toBeInstanceOf(SpitfireServer);
  });

  it('should have start method', () => {
    const server = new SpitfireServer();
    expect(typeof server.start).toBe('function');
  });

  it('should have stop method', () => {
    const server = new SpitfireServer();
    expect(typeof server.stop).toBe('function');
  });

  it('should have getScheduler method', () => {
    const server = new SpitfireServer();
    expect(typeof server.getScheduler).toBe('function');
  });

  it('getScheduler should return Scheduler instance', () => {
    const server = new SpitfireServer();
    const scheduler = server.getScheduler();
    expect(scheduler).toBeInstanceOf(Scheduler);
  });
});

describe('Scheduler Class', () => {
  let scheduler: Scheduler;

  beforeEach(() => {
    scheduler = new Scheduler();
  });

  afterEach(async () => {
    await scheduler.shutdown();
  });

  it('should instantiate without options', () => {
    expect(scheduler).toBeInstanceOf(Scheduler);
  });

  it('should instantiate with options', () => {
    const s = new Scheduler({
      maxLoadedModels: 3,
      defaultKeepAlive: '15m',
      loadTimeout: 60000,
    });
    expect(s).toBeInstanceOf(Scheduler);
    s.shutdown();
  });

  it('should have getRunner method', () => {
    expect(typeof scheduler.getRunner).toBe('function');
  });

  it('should have unloadModel method', () => {
    expect(typeof scheduler.unloadModel).toBe('function');
  });

  it('should have getLoadedModels method', () => {
    const models = scheduler.getLoadedModels();
    expect(Array.isArray(models)).toBe(true);
  });

  it('should have isLoaded method', () => {
    const loaded = scheduler.isLoaded('test-model');
    expect(typeof loaded).toBe('boolean');
    expect(loaded).toBe(false);
  });

  it('should have getModelManager method', () => {
    const mm = scheduler.getModelManager();
    expect(mm).toBeInstanceOf(ModelManager);
  });

  it('should have getRunnerManager method', () => {
    const rm = scheduler.getRunnerManager();
    expect(rm).toBeInstanceOf(RunnerManager);
  });
});

describe('ModelManager Class', () => {
  let modelManager: ModelManager;

  beforeEach(() => {
    modelManager = new ModelManager();
  });

  it('should instantiate without options', () => {
    expect(modelManager).toBeInstanceOf(ModelManager);
  });

  it('should have list method', async () => {
    const models = await modelManager.list();
    expect(Array.isArray(models)).toBe(true);
  });

  it('should have load method', () => {
    expect(typeof modelManager.load).toBe('function');
  });

  it('should have show method', () => {
    expect(typeof modelManager.show).toBe('function');
  });

  it('should have exists method', async () => {
    const exists = await modelManager.exists('nonexistent');
    expect(exists).toBe(false);
  });

  it('should have getModelsPath method', () => {
    const path = modelManager.getModelsPath();
    expect(typeof path).toBe('string');
    expect(path.length).toBeGreaterThan(0);
  });
});

describe('RunnerManager Class', () => {
  let runnerManager: RunnerManager;

  beforeEach(() => {
    runnerManager = new RunnerManager();
  });

  afterEach(async () => {
    await runnerManager.killAll();
  });

  it('should instantiate without options', () => {
    expect(runnerManager).toBeInstanceOf(RunnerManager);
  });

  it('should have spawn method', () => {
    expect(typeof runnerManager.spawn).toBe('function');
  });

  it('should have load method', () => {
    expect(typeof runnerManager.load).toBe('function');
  });

  it('should have kill method', () => {
    expect(typeof runnerManager.kill).toBe('function');
  });

  it('should have killAll method', () => {
    expect(typeof runnerManager.killAll).toBe('function');
  });

  it('should have findRunner method', () => {
    expect(typeof runnerManager.findRunner).toBe('function');
  });
});

describe('Utility Functions', () => {
  describe('parseModelName', () => {
    it('should handle complex model names', () => {
      const result = parseModelName('registry.example.com/namespace/model:tag');
      expect(result.registry).toBe('registry.example.com');
      expect(result.namespace).toBe('namespace');
      expect(result.model).toBe('model');
      expect(result.tag).toBe('tag');
    });

    it('should handle model with tag', () => {
      const result = parseModelName('model:v1');
      expect(result.model).toBe('model');
      expect(result.tag).toBe('v1');
    });
  });

  describe('formatModelName', () => {
    it('should format with all components', () => {
      const formatted = formatModelName({
        registry: 'custom.registry',
        namespace: 'myns',
        model: 'mymodel',
        tag: 'v1',
      });
      expect(formatted).toBe('custom.registry/myns/mymodel:v1');
    });
  });

  describe('parseDuration', () => {
    it('should parse simple durations', () => {
      expect(parseDuration('2h')).toBe(7200000);
      expect(parseDuration('500ms')).toBe(500);
      expect(parseDuration('30m')).toBe(1800000);
    });

    it('should handle zero and negative', () => {
      expect(parseDuration('0')).toBe(0);
      expect(parseDuration('-1')).toBe(Number.MAX_SAFE_INTEGER); // Never expire
    });
  });

  describe('formatDuration', () => {
    it('should format various durations', () => {
      expect(formatDuration(5400000)).toBe('1.5h');
      expect(formatDuration(90000)).toBe('1.5m');
      expect(formatDuration(500)).toBe('500ms');
    });
  });

  describe('formatBytes', () => {
    it('should format large values', () => {
      expect(formatBytes(1024 * 1024 * 1024 * 10)).toBe('10.0 GB');
      expect(formatBytes(1024 * 1024 * 500)).toBe('500.0 MB');
    });
  });
});

describe('Type Compatibility', () => {
  it('GenerateRequest should accept all options', () => {
    const request: GenerateRequest = {
      model: 'llama3.2',
      prompt: 'Hello',
      images: ['base64data'],
      stream: true,
      keepAlive: '5m',
      options: {
        temperature: 0.7,
        topK: 40,
        topP: 0.9,
        seed: 42,
        numPredict: 100,
        repeatPenalty: 1.1,
        stop: ['<|end|>'],
      },
    };
    expect(request.model).toBe('llama3.2');
  });

  it('ChatRequest should accept all options', () => {
    const request: ChatRequest = {
      model: 'llama3.2',
      messages: [
        { role: 'system', content: 'You are helpful' },
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi!' },
        { role: 'user', content: 'How are you?' },
      ],
      stream: false,
      keepAlive: '10m',
      options: {
        temperature: 0.8,
      },
    };
    expect(request.messages.length).toBe(4);
  });

  it('EmbedRequest should accept array input', () => {
    const request: EmbedRequest = {
      model: 'nomic-embed-text',
      input: ['First text', 'Second text', 'Third text'],
      keepAlive: '5m',
    };
    expect(Array.isArray(request.input)).toBe(true);
  });

  it('Message should support images', () => {
    const message: Message = {
      role: 'user',
      content: 'What is in this image?',
      images: ['base64encodedimage'],
    };
    expect(message.images?.length).toBe(1);
  });
});
