/**
 * Speculative Decoding Support
 * N-gram based draft prediction for faster token generation
 */

export interface SpeculativeOptions {
  /** Enable speculative decoding */
  enabled: boolean;
  /** N-gram size for context matching (default: 3) */
  ngramSize: number;
  /** Maximum draft tokens to speculate per iteration (default: 16) */
  maxDraftTokens: number;
  /** Number of parallel branches to explore (default: 8) */
  numBranches: number;
  /** Depth of primary speculation branch */
  treeDepth: number;
  /** Number of alternative candidates at first level */
  numAlternatives: number;
}

export const DEFAULT_SPECULATIVE_OPTIONS: SpeculativeOptions = {
  enabled: true,
  ngramSize: 3,
  maxDraftTokens: 64,
  numBranches: 8,
  treeDepth: 16,       // Primary branch goes 16 deep
  numAlternatives: 4,  // Plus 4 alternatives at level 1
  // Total tokens per speculation: 16 + 4 = 20 tokens processed in parallel
};

export interface SpeculativeStats {
  /** Total speculative attempts */
  attempts: number;
  /** Total tokens accepted via speculation */
  acceptedTokens: number;
  /** Total draft tokens proposed */
  proposedTokens: number;
  /** N-gram cache hits */
  cacheHits: number;
  /** N-gram cache misses */
  cacheMisses: number;
}

/**
 * N-gram based draft cache for speculative decoding
 *
 * Stores observed token continuations during generation.
 * When generating, looks up the current context to predict likely continuations.
 */
export class NgramDraftCache {
  private cache: Map<string, number[]>;
  private n: number;
  private maxEntries: number;
  private maxContinuationLen: number;
  private stats: SpeculativeStats;

  constructor(
    ngramSize: number = 3,
    maxEntries: number = 10000,
    maxContinuationLen: number = 256  // Match maxDraftTokens
  ) {
    this.cache = new Map();
    this.n = ngramSize;
    this.maxEntries = maxEntries;
    this.maxContinuationLen = maxContinuationLen;
    this.stats = {
      attempts: 0,
      acceptedTokens: 0,
      proposedTokens: 0,
      cacheHits: 0,
      cacheMisses: 0,
    };
  }

  /**
   * Generate cache key from last N tokens
   */
  private makeKey(tokens: number[]): string {
    const context = tokens.slice(-this.n);
    return context.join(',');
  }

  /**
   * Look up potential continuation for current context
   * @param tokens - All tokens generated so far
   * @param maxLen - Maximum draft length to return
   * @returns Draft tokens if found, empty array otherwise
   */
  lookup(tokens: number[], maxLen: number): number[] {
    if (tokens.length < this.n) {
      return [];
    }

    this.stats.attempts++;
    const key = this.makeKey(tokens);
    const continuation = this.cache.get(key);

    if (continuation && continuation.length > 0) {
      this.stats.cacheHits++;
      const draft = continuation.slice(0, maxLen);
      this.stats.proposedTokens += draft.length;
      return draft;
    }

    this.stats.cacheMisses++;
    return [];
  }

  /**
   * Update cache with newly generated tokens
   * Called after tokens are accepted/generated
   *
   * @param allTokens - Full sequence including new tokens
   * @param promptLen - Length of the original prompt (to avoid caching prompt patterns)
   */
  update(allTokens: number[], promptLen: number): void {
    // Only cache patterns from generated text, not the prompt
    const generatedStart = Math.max(promptLen, this.n);

    // For each position in generated text, store the continuation
    for (let i = generatedStart; i < allTokens.length - 1; i++) {
      // Context: tokens [i-n, i)
      const contextStart = i - this.n;
      if (contextStart < 0) continue;

      const context = allTokens.slice(contextStart, i);
      const key = context.join(',');

      // Continuation: tokens [i, min(i + maxLen, end)]
      const continuation = allTokens.slice(i, Math.min(i + this.maxContinuationLen, allTokens.length));

      // Store or update continuation
      // Prefer longer continuations
      const existing = this.cache.get(key);
      if (!existing || continuation.length > existing.length) {
        this.cache.set(key, continuation);
      }
    }

    // Evict old entries if cache is too large (simple FIFO)
    if (this.cache.size > this.maxEntries) {
      const keysToDelete = Array.from(this.cache.keys()).slice(0, this.cache.size - this.maxEntries);
      for (const key of keysToDelete) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Look up multiple possible continuations by varying context
   * Returns different branches for parallel speculation
   * @param tokens - All tokens generated so far
   * @param maxLen - Maximum draft length per branch
   * @param numBranches - Maximum number of branches to return
   * @returns Array of draft token arrays (different possible continuations)
   */
  lookupMultiple(tokens: number[], maxLen: number, numBranches: number): number[][] {
    const branches: number[][] = [];
    const seen = new Set<string>();

    // Try different context sizes (longer = more specific, shorter = more general)
    for (let contextSize = this.n; contextSize >= 1 && branches.length < numBranches; contextSize--) {
      if (tokens.length < contextSize) continue;

      const context = tokens.slice(-contextSize);
      const key = context.join(',');

      const continuation = this.cache.get(key);
      if (continuation && continuation.length > 0) {
        const draft = continuation.slice(0, maxLen);
        const draftKey = draft.join(',');

        // Avoid duplicate branches
        if (!seen.has(draftKey)) {
          seen.add(draftKey);
          branches.push(draft);
          this.stats.cacheHits++;
          this.stats.proposedTokens += draft.length;
        }
      }
    }

    // Also try looking up with the last token only (most general)
    if (tokens.length >= 1 && branches.length < numBranches) {
      const lastToken = tokens[tokens.length - 1];
      const key = String(lastToken);
      const continuation = this.cache.get(key);
      if (continuation && continuation.length > 0) {
        const draft = continuation.slice(0, maxLen);
        const draftKey = draft.join(',');
        if (!seen.has(draftKey)) {
          seen.add(draftKey);
          branches.push(draft);
          this.stats.cacheHits++;
          this.stats.proposedTokens += draft.length;
        }
      }
    }

    this.stats.attempts++;
    if (branches.length === 0) {
      this.stats.cacheMisses++;
    }

    return branches;
  }

  /**
   * Record accepted tokens for statistics
   */
  recordAccepted(count: number): void {
    this.stats.acceptedTokens += count;
  }

  /**
   * Clear the cache (on new conversation)
   */
  clear(): void {
    this.cache.clear();
    this.stats = {
      attempts: 0,
      acceptedTokens: 0,
      proposedTokens: 0,
      cacheHits: 0,
      cacheMisses: 0,
    };
  }

  /**
   * Seed the cache with patterns from the prompt
   * This helps speculation when the model continues prompt patterns
   * (e.g., code completion, template following)
   *
   * @param promptTokens - Tokenized prompt
   */
  seedFromPrompt(promptTokens: number[]): void {
    if (promptTokens.length < this.n + 1) return;

    // Extract n-gram patterns from prompt
    for (let i = this.n; i < promptTokens.length; i++) {
      const context = promptTokens.slice(i - this.n, i);
      const key = context.join(',');

      // Store continuation from this position
      const continuation = promptTokens.slice(i, Math.min(i + this.maxContinuationLen, promptTokens.length));
      if (continuation.length > 0) {
        const existing = this.cache.get(key);
        if (!existing || continuation.length > existing.length) {
          this.cache.set(key, continuation);
        }
      }
    }
  }

  /**
   * Detect repeating patterns and propose continuation
   * This works even without prior n-gram history
   * @param tokens - Recent tokens to check for patterns
   * @param maxLen - Maximum draft length to return
   * @returns Draft tokens if pattern found, empty array otherwise
   */
  detectRepeatPattern(tokens: number[], maxLen: number): number[] {
    if (tokens.length < 4) return [];

    // Try to find repeating patterns of various lengths
    for (let patternLen = 2; patternLen <= Math.min(16, tokens.length / 2); patternLen++) {
      const pattern = tokens.slice(-patternLen);
      const beforePattern = tokens.slice(-patternLen * 2, -patternLen);

      // Check if the pattern repeats
      let matches = true;
      for (let i = 0; i < patternLen && matches; i++) {
        if (pattern[i] !== beforePattern[i]) {
          matches = false;
        }
      }

      if (matches) {
        // Pattern is repeating! Propose continuation
        const draft = [];
        for (let i = 0; i < maxLen; i++) {
          draft.push(pattern[i % patternLen]);
        }
        // Note: stats are tracked by getDrafts, not here
        return draft;
      }
    }

    return [];
  }

  /**
   * Get drafts using all available methods
   * @param tokens - All tokens so far
   * @param maxLen - Max draft length
   * @returns Draft tokens from best available source
   */
  getDrafts(tokens: number[], maxLen: number): number[] {
    this.stats.attempts++;

    // First try n-gram cache (highest quality)
    if (tokens.length >= this.n) {
      const key = this.makeKey(tokens);
      const continuation = this.cache.get(key);
      if (continuation && continuation.length > 0) {
        this.stats.cacheHits++;
        const draft = continuation.slice(0, maxLen);
        this.stats.proposedTokens += draft.length;
        return draft;
      }
    }

    // Fall back to repeat pattern detection (only for actual repeating patterns)
    const repeatDraft = this.detectRepeatPattern(tokens, maxLen);
    if (repeatDraft.length > 0) {
      this.stats.cacheHits++;  // Count repeat detection as a "hit"
      this.stats.proposedTokens += repeatDraft.length;
      return repeatDraft;
    }

    // No aggressive fallback - low acceptance rate drafts waste GPU cycles
    // The forward pass cost scales with draft length, so bad drafts are worse than no drafts
    this.stats.cacheMisses++;
    return [];
  }

  /**
   * Get cache statistics
   */
  getStats(): SpeculativeStats & { cacheSize: number; hitRate: number; acceptanceRate: number } {
    const hitRate = this.stats.attempts > 0
      ? this.stats.cacheHits / this.stats.attempts
      : 0;
    const acceptanceRate = this.stats.proposedTokens > 0
      ? this.stats.acceptedTokens / this.stats.proposedTokens
      : 0;

    return {
      ...this.stats,
      cacheSize: this.cache.size,
      hitRate,
      acceptanceRate,
    };
  }
}
