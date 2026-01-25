/**
 * Tokenizer for GGUF models
 * Implements BPE tokenization compatible with Llama/Qwen models
 */

import type { GGUFFile } from '../types/model.js';
import { GGUF_KEYS } from '../types/model.js';

export interface TokenizerConfig {
  bosTokenId?: number;
  eosTokenId?: number;
  padTokenId?: number;
  unkTokenId?: number;
}

/**
 * BPE Tokenizer for GGUF models
 */
export class Tokenizer {
  private vocab: Map<string, number> = new Map();
  private reverseVocab: Map<number, string> = new Map();
  private merges: Map<string, number> = new Map();

  readonly bosTokenId: number;
  readonly eosTokenId: number;
  readonly padTokenId: number;
  readonly unkTokenId: number;
  readonly vocabSize: number;

  constructor(
    vocab: string[],
    merges: string[] | null,
    config: TokenizerConfig = {}
  ) {
    // Build vocabulary maps
    for (let i = 0; i < vocab.length; i++) {
      this.vocab.set(vocab[i], i);
      this.reverseVocab.set(i, vocab[i]);
    }
    this.vocabSize = vocab.length;

    // Build merges map with priority (earlier = higher priority)
    if (merges) {
      for (let i = 0; i < merges.length; i++) {
        this.merges.set(merges[i], i);
      }
    }

    // Set special tokens
    this.bosTokenId = config.bosTokenId ?? this.findSpecialToken(['<|im_start|>', '<s>', '<bos>']) ?? 1;
    this.eosTokenId = config.eosTokenId ?? this.findSpecialToken(['<|im_end|>', '</s>', '<eos>', '<|endoftext|>']) ?? 2;
    this.padTokenId = config.padTokenId ?? this.findSpecialToken(['<|pad|>', '<pad>']) ?? 0;
    this.unkTokenId = config.unkTokenId ?? this.findSpecialToken(['<|unk|>', '<unk>']) ?? 0;
  }

  private findSpecialToken(candidates: string[]): number | undefined {
    for (const candidate of candidates) {
      const id = this.vocab.get(candidate);
      if (id !== undefined) {
        return id;
      }
    }
    return undefined;
  }

  /**
   * Load tokenizer from GGUF file metadata
   */
  static fromGGUF(gguf: GGUFFile): Tokenizer {
    const metadata = gguf.metadata;

    // Get vocabulary
    let vocab = metadata[GGUF_KEYS.TOKENIZER_LIST];
    if (typeof vocab === 'string' && vocab.startsWith('[array of')) {
      // Vocabulary was skipped due to size, need to re-parse
      throw new Error(
        'Vocabulary was not loaded from GGUF. Increase maxArraySize when parsing.'
      );
    }
    if (!Array.isArray(vocab)) {
      throw new Error('No vocabulary found in GGUF metadata');
    }

    // Get merges (may not exist for all tokenizers)
    let merges = metadata[GGUF_KEYS.TOKENIZER_MERGES];
    if (typeof merges === 'string' && merges.startsWith('[array of')) {
      merges = null; // Merges were skipped
    }

    // Get special token IDs
    const config: TokenizerConfig = {
      bosTokenId: metadata[GGUF_KEYS.TOKENIZER_BOS_ID] as number | undefined,
      eosTokenId: metadata[GGUF_KEYS.TOKENIZER_EOS_ID] as number | undefined,
      padTokenId: metadata[GGUF_KEYS.TOKENIZER_PAD_ID] as number | undefined,
      unkTokenId: metadata[GGUF_KEYS.TOKENIZER_UNK_ID] as number | undefined,
    };

    return new Tokenizer(vocab as string[], merges as string[] | null, config);
  }

  /**
   * Encode text to token IDs
   */
  encode(text: string): number[] {
    if (!text) {
      return [];
    }

    // Pre-tokenize into words/pieces
    const pieces = this.preTokenize(text);

    // Apply BPE to each piece
    const tokens: number[] = [];
    for (const piece of pieces) {
      const pieceTokens = this.bpeEncode(piece);
      tokens.push(...pieceTokens);
    }

    return tokens;
  }

  /**
   * Pre-tokenize text into pieces
   * This handles special tokens and splits on whitespace/punctuation
   */
  private preTokenize(text: string): string[] {
    const pieces: string[] = [];

    // Check for special tokens first
    const specialTokenPattern = this.buildSpecialTokenPattern();
    if (specialTokenPattern) {
      // When split() uses a capturing group, matched tokens are included in the result
      // e.g., "<|im_start|>user".split(/(<|im_start|>)/) => ["", "<|im_start|>", "user"]
      const parts = text.split(specialTokenPattern);

      for (const part of parts) {
        if (!part) continue; // Skip empty strings

        // Check if this part is a special token (it will be in vocab as-is)
        if (this.vocab.has(part)) {
          pieces.push(part);
        } else {
          // Regular text - split by whitespace/unicode categories
          pieces.push(...this.splitText(part));
        }
      }
    } else {
      pieces.push(...this.splitText(text));
    }

    return pieces.filter(p => p.length > 0);
  }

  private buildSpecialTokenPattern(): RegExp | null {
    const specialTokens: string[] = [];

    // Add known special tokens to pattern
    const candidates = [
      '<|im_start|>', '<|im_end|>', '<|endoftext|>',
      '<s>', '</s>', '<pad>', '<unk>',
      '<|system|>', '<|user|>', '<|assistant|>',
    ];

    for (const token of candidates) {
      if (this.vocab.has(token)) {
        specialTokens.push(this.escapeRegex(token));
      }
    }

    if (specialTokens.length === 0) {
      return null;
    }

    return new RegExp(`(${specialTokens.join('|')})`, 'g');
  }

  private escapeRegex(str: string): string {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  /**
   * Split text into pieces based on whitespace and unicode categories
   * Similar to GPT-2/Qwen tokenization
   */
  private splitText(text: string): string[] {
    const pieces: string[] = [];

    // Pattern similar to GPT-2 but simplified:
    // Split on whitespace boundaries while preserving the whitespace
    // Also split numbers and punctuation
    const pattern = /(\s+|\d+|[^\s\d]+)/g;
    let match;

    while ((match = pattern.exec(text)) !== null) {
      let piece = match[1];

      // For leading spaces on words, encode space as part of the token
      // Many tokenizers use special space characters (like Ġ for GPT-2)
      // Qwen uses direct UTF-8 encoding
      if (pieces.length > 0 || !piece.startsWith(' ')) {
        pieces.push(piece);
      } else if (piece.startsWith(' ')) {
        // Leading whitespace
        pieces.push(piece);
      }
    }

    return pieces;
  }

  /**
   * Apply BPE encoding to a single piece
   */
  private bpeEncode(piece: string): number[] {
    // Check if the whole piece is in vocabulary (common for short tokens)
    const wholeId = this.vocab.get(piece);
    if (wholeId !== undefined) {
      return [wholeId];
    }

    // Try with space prefix (Ġ style for GPT-2 derivatives)
    const spacePrefixed = '▁' + piece; // LOWER ONE EIGHTH BLOCK
    const spacePrefixedId = this.vocab.get(spacePrefixed);
    if (spacePrefixedId !== undefined) {
      return [spacePrefixedId];
    }

    // Convert to characters (handling UTF-8 properly)
    let symbols = Array.from(piece);

    // If no merges, fall back to character-level
    if (this.merges.size === 0) {
      return this.encodeCharacters(symbols);
    }

    // Apply BPE merges
    while (symbols.length > 1) {
      // Find the best merge (lowest priority number = highest priority)
      let bestMerge: string | null = null;
      let bestPriority = Infinity;
      let bestIdx = -1;

      for (let i = 0; i < symbols.length - 1; i++) {
        const pair = symbols[i] + ' ' + symbols[i + 1];
        const priority = this.merges.get(pair);
        if (priority !== undefined && priority < bestPriority) {
          bestMerge = pair;
          bestPriority = priority;
          bestIdx = i;
        }
      }

      if (bestMerge === null) {
        break; // No more merges possible
      }

      // Apply the merge
      const merged = symbols[bestIdx] + symbols[bestIdx + 1];
      symbols = [
        ...symbols.slice(0, bestIdx),
        merged,
        ...symbols.slice(bestIdx + 2),
      ];
    }

    // Convert symbols to token IDs
    return this.encodeCharacters(symbols);
  }

  // Cache for GPT-2 byte mappings (computed once)
  private static gpt2ByteToChar: Map<number, string> | null = null;
  private static gpt2CharToByte: Map<string, number> | null = null;

  /**
   * Initialize GPT-2 byte encoding tables (lazy initialization)
   */
  private static initGPT2Tables(): void {
    if (Tokenizer.gpt2ByteToChar !== null) return;

    // GPT-2 bytes_to_unicode mapping
    const bs = [
      ...Array.from({ length: 94 }, (_, i) => 33 + i),   // 33-126 (printable ASCII)
      ...Array.from({ length: 12 }, (_, i) => 161 + i),  // 161-172
      ...Array.from({ length: 82 }, (_, i) => 174 + i),  // 174-255
    ];

    const cs = [...bs];
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) {
        bs.push(b);
        cs.push(256 + n);
        n++;
      }
    }

    Tokenizer.gpt2ByteToChar = new Map<number, string>();
    Tokenizer.gpt2CharToByte = new Map<string, number>();

    for (let i = 0; i < bs.length; i++) {
      const char = String.fromCharCode(cs[i]);
      Tokenizer.gpt2ByteToChar.set(bs[i], char);
      Tokenizer.gpt2CharToByte.set(char, bs[i]);
    }
  }

  /**
   * GPT-2 style byte-to-unicode mapping
   * Maps bytes to unicode characters to avoid whitespace/control char issues
   */
  private byteToUnicode(byte: number): string {
    Tokenizer.initGPT2Tables();
    return Tokenizer.gpt2ByteToChar!.get(byte) || String.fromCharCode(byte);
  }

  /**
   * Decode GPT-2 style unicode characters back to bytes
   * Converts Ġ -> space, Ċ -> newline, etc.
   */
  private decodeGPT2Bytes(text: string): string {
    Tokenizer.initGPT2Tables();

    const result: number[] = [];
    for (const char of text) {
      const byte = Tokenizer.gpt2CharToByte!.get(char);
      if (byte !== undefined) {
        result.push(byte);
      } else {
        // Regular character, encode as UTF-8
        const encoded = new TextEncoder().encode(char);
        result.push(...encoded);
      }
    }

    return new TextDecoder().decode(new Uint8Array(result));
  }

  /**
   * Encode individual characters/symbols to token IDs
   */
  private encodeCharacters(symbols: string[]): number[] {
    const tokens: number[] = [];

    for (const symbol of symbols) {
      let id = this.vocab.get(symbol);

      // Try common alternatives
      if (id === undefined) {
        // Try with space prefix (SentencePiece style)
        id = this.vocab.get('▁' + symbol);
      }

      if (id === undefined) {
        // Try GPT-2 byte-level encoding
        const bytes = new TextEncoder().encode(symbol);
        for (const byte of bytes) {
          // First try GPT-2 style unicode char (e.g., Ċ for newline, Ġ for space)
          const gpt2Char = this.byteToUnicode(byte);
          const gpt2Id = this.vocab.get(gpt2Char);
          if (gpt2Id !== undefined) {
            tokens.push(gpt2Id);
            continue;
          }

          // Then try <0xNN> format
          const byteToken = `<0x${byte.toString(16).toUpperCase().padStart(2, '0')}>`;
          const byteId = this.vocab.get(byteToken);
          if (byteId !== undefined) {
            tokens.push(byteId);
          } else {
            // Fall back to unknown token
            tokens.push(this.unkTokenId);
          }
        }
        continue;
      }

      tokens.push(id);
    }

    return tokens;
  }

  /**
   * Decode token IDs to text
   */
  decode(tokens: number[]): string {
    const pieces: string[] = [];

    for (const token of tokens) {
      const piece = this.reverseVocab.get(token);
      if (piece !== undefined) {
        pieces.push(piece);
      }
    }

    let text = pieces.join('');

    // Handle SentencePiece space encoding (▁)
    text = text.replace(/▁/g, ' ');

    // Handle <0xNN> byte tokens
    text = text.replace(/<0x([0-9A-Fa-f]{2})>/g, (_, hex) => {
      return String.fromCharCode(parseInt(hex, 16));
    });

    // Handle GPT-2 byte-level encoding (Ġ -> space, Ċ -> newline, etc.)
    text = this.decodeGPT2Bytes(text);

    return text;
  }

  /**
   * Decode a single token ID to text
   */
  decodeToken(token: number): string {
    const piece = this.reverseVocab.get(token);
    if (piece === undefined) {
      return '';
    }

    let text = piece;

    // Handle SentencePiece space encoding (▁)
    text = text.replace(/▁/g, ' ');

    // Handle <0xNN> byte tokens
    text = text.replace(/<0x([0-9A-Fa-f]{2})>/g, (_, hex) => {
      return String.fromCharCode(parseInt(hex, 16));
    });

    // Handle GPT-2 byte-level encoding (Ġ -> space, Ċ -> newline, etc.)
    text = this.decodeGPT2Bytes(text);

    return text;
  }

  /**
   * Get token string for a given ID (raw, no decoding)
   */
  getToken(id: number): string | undefined {
    return this.reverseVocab.get(id);
  }

  /**
   * Get ID for a given token string
   */
  getTokenId(token: string): number | undefined {
    return this.vocab.get(token);
  }
}

/**
 * Parse a GGUF file with full tokenizer data (no array size limit)
 */
export async function parseGGUFWithTokenizer(filePath: string): Promise<GGUFFile> {
  const { parseGGUF } = await import('../model/gguf.js');
  // Parse with a very large maxArraySize to get the vocabulary
  return parseGGUF(filePath, -1); // -1 means no limit
}
