/**
 * Tests for Speculative Decoding (N-gram Draft Cache)
 */

import {
  NgramDraftCache,
  DEFAULT_SPECULATIVE_OPTIONS,
} from '../src/engine/speculative.js';

describe('NgramDraftCache', () => {
  describe('Basic Functionality', () => {
    test('should create cache with default options', () => {
      const cache = new NgramDraftCache();
      const stats = cache.getStats();
      expect(stats.cacheSize).toBe(0);
      expect(stats.attempts).toBe(0);
    });

    test('should create cache with custom n-gram size', () => {
      const cache = new NgramDraftCache(4);
      // Lookup should fail with less than 4 tokens
      const drafts = cache.lookup([1, 2, 3], 10);
      expect(drafts.length).toBe(0);
    });

    test('should return empty for insufficient context', () => {
      const cache = new NgramDraftCache(3);
      const drafts = cache.lookup([1, 2], 10);
      expect(drafts.length).toBe(0);
    });
  });

  describe('Cache Update and Lookup', () => {
    test('should store and retrieve continuations', () => {
      const cache = new NgramDraftCache(3);

      // Simulate generation: [1, 2, 3, 4, 5, 6]
      const allTokens = [1, 2, 3, 4, 5, 6];
      const promptLen = 2; // First 2 tokens are prompt

      cache.update(allTokens, promptLen);

      // Looking up context [1, 2, 3] should return [4, 5, 6]
      const drafts = cache.lookup([1, 2, 3], 10);
      expect(drafts).toEqual([4, 5, 6]);
    });

    test('should limit draft length to maxLen', () => {
      const cache = new NgramDraftCache(3);

      const allTokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      cache.update(allTokens, 2);

      // Request only 3 tokens
      const drafts = cache.lookup([1, 2, 3], 3);
      expect(drafts.length).toBe(3);
      expect(drafts).toEqual([4, 5, 6]);
    });

    test('should prefer longer continuations', () => {
      const cache = new NgramDraftCache(3);

      // First update: short continuation
      cache.update([1, 2, 3, 4, 5], 2);

      // Second update: longer continuation for same context
      cache.update([1, 2, 3, 4, 5, 6, 7, 8], 2);

      const drafts = cache.lookup([1, 2, 3], 10);
      expect(drafts.length).toBe(5); // Should get [4, 5, 6, 7, 8]
    });

    test('should handle cache miss', () => {
      const cache = new NgramDraftCache(3);

      cache.update([1, 2, 3, 4, 5], 2);

      // Different context
      const drafts = cache.lookup([10, 11, 12], 10);
      expect(drafts.length).toBe(0);

      const stats = cache.getStats();
      expect(stats.cacheMisses).toBeGreaterThan(0);
    });
  });

  describe('Prompt Seeding', () => {
    test('should seed cache from prompt tokens', () => {
      const cache = new NgramDraftCache(3);

      // Seed with a prompt
      const prompt = [100, 101, 102, 103, 104, 105, 106];
      cache.seedFromPrompt(prompt);

      // Should find continuation for context in prompt
      const drafts = cache.lookup([100, 101, 102], 10);
      expect(drafts.length).toBeGreaterThan(0);
      expect(drafts[0]).toBe(103);
    });

    test('should not seed from short prompts', () => {
      const cache = new NgramDraftCache(3);

      // Too short to seed (need n+1 tokens minimum)
      cache.seedFromPrompt([1, 2, 3]);

      const stats = cache.getStats();
      expect(stats.cacheSize).toBe(0);
    });

    test('should extract multiple n-grams from prompt', () => {
      const cache = new NgramDraftCache(2);

      const prompt = [1, 2, 3, 4, 5];
      cache.seedFromPrompt(prompt);

      // Should find patterns for multiple contexts
      expect(cache.lookup([1, 2], 1)).toEqual([3]);
      expect(cache.lookup([2, 3], 1)).toEqual([4]);
      expect(cache.lookup([3, 4], 1)).toEqual([5]);
    });
  });

  describe('Repeat Pattern Detection', () => {
    test('should detect simple repeating pattern', () => {
      const cache = new NgramDraftCache(3);

      // Pattern: [1, 2] repeating
      const tokens = [1, 2, 1, 2, 1, 2, 1, 2];
      const drafts = cache.detectRepeatPattern(tokens, 6);

      expect(drafts.length).toBe(6);
      // Should continue the pattern
      expect(drafts[0]).toBe(1);
      expect(drafts[1]).toBe(2);
    });

    test('should detect longer repeating patterns', () => {
      const cache = new NgramDraftCache(3);

      // Pattern: [1, 2, 3] repeating
      const tokens = [1, 2, 3, 1, 2, 3, 1, 2, 3];
      const drafts = cache.detectRepeatPattern(tokens, 9);

      expect(drafts.length).toBe(9);
      expect(drafts.slice(0, 3)).toEqual([1, 2, 3]);
    });

    test('should return empty for non-repeating sequences', () => {
      const cache = new NgramDraftCache(3);

      const tokens = [1, 2, 3, 4, 5, 6, 7, 8];
      const drafts = cache.detectRepeatPattern(tokens, 4);

      expect(drafts.length).toBe(0);
    });

    test('should return empty for short sequences', () => {
      const cache = new NgramDraftCache(3);

      const drafts = cache.detectRepeatPattern([1, 2, 3], 4);
      expect(drafts.length).toBe(0);
    });
  });

  describe('getDrafts (Combined Strategy)', () => {
    test('should prefer n-gram cache over repeat detection', () => {
      const cache = new NgramDraftCache(3);

      // Set up n-gram cache with specific continuation
      cache.update([1, 2, 3, 10, 11, 12], 0);

      // Input has repeat pattern but also n-gram match
      const tokens = [5, 5, 1, 2, 3]; // Last 3 match cached n-gram
      const drafts = cache.getDrafts(tokens, 3);

      // Should return n-gram result, not repeat pattern
      expect(drafts).toEqual([10, 11, 12]);
    });

    test('should fall back to repeat detection when no n-gram match', () => {
      const cache = new NgramDraftCache(3);

      // No n-gram cache entries
      // But tokens have repeating pattern
      const tokens = [7, 8, 7, 8, 7, 8];
      const drafts = cache.getDrafts(tokens, 4);

      expect(drafts.length).toBe(4);
      expect(drafts[0]).toBe(7);
      expect(drafts[1]).toBe(8);
    });

    test('should return empty when no patterns found', () => {
      const cache = new NgramDraftCache(3);

      // Random non-repeating tokens, no cache
      const tokens = [1, 3, 7, 2, 9, 4];
      const drafts = cache.getDrafts(tokens, 4);

      expect(drafts.length).toBe(0);

      const stats = cache.getStats();
      expect(stats.cacheMisses).toBeGreaterThan(0);
    });
  });

  describe('Statistics Tracking', () => {
    test('should track cache hits', () => {
      const cache = new NgramDraftCache(3);

      cache.update([1, 2, 3, 4, 5], 0);
      cache.lookup([1, 2, 3], 3);
      cache.lookup([1, 2, 3], 3);

      const stats = cache.getStats();
      expect(stats.cacheHits).toBe(2);
    });

    test('should track cache misses', () => {
      const cache = new NgramDraftCache(3);

      cache.lookup([1, 2, 3], 3);
      cache.lookup([4, 5, 6], 3);

      const stats = cache.getStats();
      expect(stats.cacheMisses).toBe(2);
    });

    test('should track proposed tokens', () => {
      const cache = new NgramDraftCache(3);

      cache.update([1, 2, 3, 4, 5, 6, 7], 0);
      cache.lookup([1, 2, 3], 3); // Proposes 3 tokens
      cache.lookup([1, 2, 3], 2); // Proposes 2 tokens

      const stats = cache.getStats();
      expect(stats.proposedTokens).toBe(5);
    });

    test('should track accepted tokens', () => {
      const cache = new NgramDraftCache(3);

      cache.recordAccepted(5);
      cache.recordAccepted(3);

      const stats = cache.getStats();
      expect(stats.acceptedTokens).toBe(8);
    });

    test('should calculate hit rate correctly', () => {
      const cache = new NgramDraftCache(3);

      cache.update([1, 2, 3, 4, 5], 0);
      cache.lookup([1, 2, 3], 3); // Hit
      cache.lookup([7, 8, 9], 3); // Miss
      cache.lookup([1, 2, 3], 3); // Hit
      cache.lookup([10, 11, 12], 3); // Miss

      const stats = cache.getStats();
      expect(stats.hitRate).toBe(0.5); // 2 hits / 4 attempts
    });

    test('should calculate acceptance rate correctly', () => {
      const cache = new NgramDraftCache(3);

      cache.update([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0);
      cache.lookup([1, 2, 3], 5); // Proposes 5
      cache.recordAccepted(3); // Only 3 accepted

      const stats = cache.getStats();
      expect(stats.acceptanceRate).toBe(0.6); // 3/5
    });
  });

  describe('Cache Management', () => {
    test('should clear cache and stats', () => {
      const cache = new NgramDraftCache(3);

      cache.update([1, 2, 3, 4, 5], 0);
      cache.lookup([1, 2, 3], 3);
      cache.recordAccepted(2);

      cache.clear();

      const stats = cache.getStats();
      expect(stats.cacheSize).toBe(0);
      expect(stats.attempts).toBe(0);
      expect(stats.cacheHits).toBe(0);
      expect(stats.acceptedTokens).toBe(0);
    });

    test('should evict old entries when cache is full', () => {
      const maxEntries = 5;
      const cache = new NgramDraftCache(2, maxEntries);

      // Add more entries than max
      for (let i = 0; i < 10; i++) {
        cache.update([i, i + 1, i + 2, i + 3], 0);
      }

      const stats = cache.getStats();
      expect(stats.cacheSize).toBeLessThanOrEqual(maxEntries);
    });
  });

  describe('Multiple Branch Lookup', () => {
    test('should return multiple branches with different contexts', () => {
      const cache = new NgramDraftCache(3);

      // Set up cache with different continuations
      cache.update([1, 2, 3, 10, 11], 0);
      cache.update([2, 3, 4, 20, 21], 0);
      cache.update([3, 4, 5, 30, 31], 0);

      // Lookup with context that matches multiple patterns
      const branches = cache.lookupMultiple([1, 2, 3], 2, 3);

      expect(branches.length).toBeGreaterThan(0);
    });

    test('should avoid duplicate branches', () => {
      const cache = new NgramDraftCache(2);

      // Same continuation for different context lengths
      cache.update([1, 2, 3, 4, 5], 0);

      const branches = cache.lookupMultiple([1, 2], 3, 5);

      // Should deduplicate
      const uniqueBranches = new Set(branches.map(b => b.join(',')));
      expect(uniqueBranches.size).toBe(branches.length);
    });
  });

  describe('Edge Cases', () => {
    test('should handle empty token array', () => {
      const cache = new NgramDraftCache(3);

      const drafts = cache.lookup([], 10);
      expect(drafts.length).toBe(0);

      const getDraftsResult = cache.getDrafts([], 10);
      expect(getDraftsResult.length).toBe(0);
    });

    test('should handle maxLen = 0', () => {
      const cache = new NgramDraftCache(3);
      cache.update([1, 2, 3, 4, 5], 0);

      const drafts = cache.lookup([1, 2, 3], 0);
      expect(drafts.length).toBe(0);
    });

    test('should handle single token context', () => {
      const cache = new NgramDraftCache(1);

      cache.update([5, 10, 15, 20], 0);

      const drafts = cache.lookup([5], 3);
      expect(drafts).toEqual([10, 15, 20]);
    });

    test('should handle very long continuations', () => {
      const cache = new NgramDraftCache(3, 1000, 10); // maxContinuationLen = 10

      const longSequence = Array.from({ length: 100 }, (_, i) => i);
      cache.update(longSequence, 0);

      // Should truncate to maxContinuationLen
      const drafts = cache.lookup([0, 1, 2], 100);
      expect(drafts.length).toBeLessThanOrEqual(10);
    });
  });
});

describe('DEFAULT_SPECULATIVE_OPTIONS', () => {
  test('should have sensible defaults', () => {
    expect(DEFAULT_SPECULATIVE_OPTIONS.enabled).toBe(true);
    expect(DEFAULT_SPECULATIVE_OPTIONS.ngramSize).toBeGreaterThan(0);
    expect(DEFAULT_SPECULATIVE_OPTIONS.maxDraftTokens).toBeGreaterThan(0);
  });
});
