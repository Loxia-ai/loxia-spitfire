/**
 * Type tests - verify TypeScript definitions work correctly
 */

import {
  parseModelName,
  formatModelName,
  DEFAULT_OPTIONS,
  parseDuration,
  formatDuration,
  formatBytes,
  type GenerateRequest,
  type ChatRequest,
  type Message,
  type ModelName,
} from '../src/index.js';

describe('Model Name Parsing', () => {
  test('should parse simple model name', () => {
    const result = parseModelName('llama3.2');
    expect(result).toEqual({
      registry: 'registry.ollama.ai',
      namespace: 'library',
      model: 'llama3.2',
      tag: 'latest',
    });
  });

  test('should parse model name with tag', () => {
    const result = parseModelName('llama3.2:8b');
    expect(result).toEqual({
      registry: 'registry.ollama.ai',
      namespace: 'library',
      model: 'llama3.2',
      tag: '8b',
    });
  });

  test('should parse model name with namespace', () => {
    const result = parseModelName('myuser/mymodel:v1');
    expect(result).toEqual({
      registry: 'registry.ollama.ai',
      namespace: 'myuser',
      model: 'mymodel',
      tag: 'v1',
    });
  });

  test('should parse full model name with registry', () => {
    const result = parseModelName('my.registry.com/namespace/model:tag');
    expect(result).toEqual({
      registry: 'my.registry.com',
      namespace: 'namespace',
      model: 'model',
      tag: 'tag',
    });
  });
});

describe('Model Name Formatting', () => {
  test('should format simple model name', () => {
    const name: ModelName = {
      registry: 'registry.ollama.ai',
      namespace: 'library',
      model: 'llama3.2',
      tag: 'latest',
    };
    expect(formatModelName(name)).toBe('llama3.2');
  });

  test('should format model name with custom namespace', () => {
    const name: ModelName = {
      registry: 'registry.ollama.ai',
      namespace: 'myuser',
      model: 'mymodel',
      tag: 'latest',
    };
    expect(formatModelName(name)).toBe('myuser/mymodel');
  });

  test('should format model name with tag', () => {
    const name: ModelName = {
      registry: 'registry.ollama.ai',
      namespace: 'library',
      model: 'llama3.2',
      tag: '8b',
    };
    expect(formatModelName(name)).toBe('llama3.2:8b');
  });
});

describe('Duration Parsing', () => {
  test('should parse number as seconds', () => {
    expect(parseDuration(60)).toBe(60000);
  });

  test('should parse string with seconds', () => {
    expect(parseDuration('30s')).toBe(30000);
  });

  test('should parse string with minutes', () => {
    expect(parseDuration('5m')).toBe(300000);
  });

  test('should parse string with hours', () => {
    expect(parseDuration('1h')).toBe(3600000);
  });

  test('should handle negative duration as never expire', () => {
    expect(parseDuration(-1)).toBe(Number.MAX_SAFE_INTEGER);
    expect(parseDuration('-1')).toBe(Number.MAX_SAFE_INTEGER);
  });

  test('should use default for undefined', () => {
    expect(parseDuration(undefined)).toBe(300000); // 5 minutes
  });
});

describe('Duration Formatting', () => {
  test('should format milliseconds', () => {
    expect(formatDuration(500)).toBe('500ms');
  });

  test('should format seconds', () => {
    expect(formatDuration(5000)).toBe('5.0s');
  });

  test('should format minutes', () => {
    expect(formatDuration(120000)).toBe('2.0m');
  });

  test('should format hours', () => {
    expect(formatDuration(7200000)).toBe('2.0h');
  });
});

describe('Bytes Formatting', () => {
  test('should format bytes', () => {
    expect(formatBytes(500)).toBe('500.0 B');
  });

  test('should format kilobytes', () => {
    expect(formatBytes(2048)).toBe('2.0 KB');
  });

  test('should format megabytes', () => {
    expect(formatBytes(5242880)).toBe('5.0 MB');
  });

  test('should format gigabytes', () => {
    expect(formatBytes(5368709120)).toBe('5.0 GB');
  });
});

describe('Default Options', () => {
  test('should have reasonable defaults', () => {
    expect(DEFAULT_OPTIONS.temperature).toBe(0.8);
    expect(DEFAULT_OPTIONS.topK).toBe(40);
    expect(DEFAULT_OPTIONS.topP).toBe(0.9);
    expect(DEFAULT_OPTIONS.numCtx).toBe(2048);
  });
});

describe('Type Definitions', () => {
  test('GenerateRequest type should be valid', () => {
    const request: GenerateRequest = {
      model: 'llama3.2',
      prompt: 'Hello, world!',
      stream: true,
      options: {
        temperature: 0.7,
      },
    };
    expect(request.model).toBe('llama3.2');
    expect(request.prompt).toBe('Hello, world!');
  });

  test('ChatRequest type should be valid', () => {
    const message: Message = {
      role: 'user',
      content: 'Hi there!',
    };

    const request: ChatRequest = {
      model: 'llama3.2',
      messages: [message],
      stream: false,
    };

    expect(request.model).toBe('llama3.2');
    expect(request.messages).toHaveLength(1);
    expect(request.messages[0].role).toBe('user');
  });

  test('Message type should support all roles', () => {
    const messages: Message[] = [
      { role: 'system', content: 'You are helpful' },
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi!' },
      { role: 'tool', content: 'Tool result', toolCallId: '123' },
    ];
    expect(messages).toHaveLength(4);
  });
});
