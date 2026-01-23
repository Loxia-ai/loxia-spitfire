/**
 * Spitfire WASM Bindings
 * Exposes llama.cpp functionality to JavaScript via Emscripten
 */

#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#include <string>
#include <vector>
#include <memory>
#include <cstring>

#include "llama.h"

// Global state
static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static std::vector<llama_token> g_tokens;

// Callback function pointer for streaming
typedef void (*stream_callback_t)(const char* text, int is_done);
static stream_callback_t g_stream_callback = nullptr;

extern "C" {

/**
 * Initialize llama backend
 */
EMSCRIPTEN_KEEPALIVE
int llama_init() {
    llama_backend_init();
    return 0;
}

/**
 * Load a model from a file path
 * Returns 0 on success, -1 on failure
 */
EMSCRIPTEN_KEEPALIVE
int llama_load_model(const char* model_path, int n_ctx, int n_batch, int n_threads) {
    // Free existing model if any
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }

    // Model params
    llama_model_params model_params = llama_model_default_params();

    // Load model
    g_model = llama_model_load_from_file(model_path, model_params);
    if (!g_model) {
        return -1;
    }

    // Context params
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx > 0 ? n_ctx : 2048;
    ctx_params.n_batch = n_batch > 0 ? n_batch : 512;
    ctx_params.n_threads = n_threads > 0 ? n_threads : 4;

    // Create context
    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (!g_ctx) {
        llama_model_free(g_model);
        g_model = nullptr;
        return -1;
    }

    return 0;
}

/**
 * Tokenize a string
 * Returns number of tokens, -1 on failure
 */
EMSCRIPTEN_KEEPALIVE
int llama_tokenize_text(const char* text, int* tokens_out, int max_tokens, bool add_bos) {
    if (!g_model || !text || !tokens_out) {
        return -1;
    }

    std::vector<llama_token> tokens(max_tokens);
    int n_tokens = llama_tokenize(
        g_model,
        text,
        strlen(text),
        tokens.data(),
        max_tokens,
        add_bos,
        true  // special tokens
    );

    if (n_tokens < 0) {
        return n_tokens;
    }

    for (int i = 0; i < n_tokens && i < max_tokens; i++) {
        tokens_out[i] = tokens[i];
    }

    return n_tokens;
}

/**
 * Detokenize tokens to string
 * Returns the text
 */
EMSCRIPTEN_KEEPALIVE
const char* llama_detokenize(int* tokens, int n_tokens) {
    static std::string result;
    result.clear();

    if (!g_model || !tokens || n_tokens <= 0) {
        return "";
    }

    for (int i = 0; i < n_tokens; i++) {
        char buf[256];
        int n = llama_token_to_piece(g_model, tokens[i], buf, sizeof(buf), 0, true);
        if (n > 0) {
            result.append(buf, n);
        }
    }

    return result.c_str();
}

/**
 * Generate text completion
 * Calls the stream callback for each token
 * Returns 0 on success, -1 on failure
 */
EMSCRIPTEN_KEEPALIVE
int llama_generate(
    const char* prompt,
    int max_tokens,
    float temperature,
    float top_p,
    int top_k,
    float repeat_penalty,
    stream_callback_t callback
) {
    if (!g_model || !g_ctx || !prompt) {
        return -1;
    }

    g_stream_callback = callback;

    // Tokenize prompt
    std::vector<llama_token> tokens(llama_n_ctx(g_ctx));
    int n_prompt_tokens = llama_tokenize(
        g_model,
        prompt,
        strlen(prompt),
        tokens.data(),
        tokens.size(),
        true,  // add_bos
        true   // special
    );

    if (n_prompt_tokens < 0) {
        return -1;
    }
    tokens.resize(n_prompt_tokens);

    // Clear KV cache
    llama_kv_cache_clear(g_ctx);

    // Process prompt
    llama_batch batch = llama_batch_init(512, 0, 1);

    for (int i = 0; i < n_prompt_tokens; i++) {
        llama_batch_add(batch, tokens[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(g_ctx, batch) != 0) {
        llama_batch_free(batch);
        return -1;
    }

    // Setup sampler
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));

    // Generate tokens
    int n_generated = 0;
    int n_cur = n_prompt_tokens;

    while (n_generated < max_tokens) {
        // Sample next token
        llama_token new_token = llama_sampler_sample(sampler, g_ctx, -1);

        // Check for end of generation
        if (llama_token_is_eog(g_model, new_token)) {
            if (callback) {
                callback("", 1);  // Signal done
            }
            break;
        }

        // Convert token to text
        char buf[256];
        int n = llama_token_to_piece(g_model, new_token, buf, sizeof(buf), 0, true);

        if (n > 0) {
            buf[n] = '\0';
            if (callback) {
                callback(buf, 0);  // Stream token
            }
        }

        // Prepare next batch
        llama_batch_clear(batch);
        llama_batch_add(batch, new_token, n_cur, {0}, true);

        if (llama_decode(g_ctx, batch) != 0) {
            break;
        }

        n_cur++;
        n_generated++;
    }

    // Signal completion if not already done
    if (callback && n_generated >= max_tokens) {
        callback("", 1);
    }

    llama_sampler_free(sampler);
    llama_batch_free(batch);

    return n_generated;
}

/**
 * Get embeddings for text
 * Returns embedding dimension, -1 on failure
 */
EMSCRIPTEN_KEEPALIVE
int llama_embedding(const char* text, float* embedding_out, int max_dim) {
    if (!g_model || !g_ctx || !text || !embedding_out) {
        return -1;
    }

    // Tokenize
    std::vector<llama_token> tokens(llama_n_ctx(g_ctx));
    int n_tokens = llama_tokenize(
        g_model,
        text,
        strlen(text),
        tokens.data(),
        tokens.size(),
        true,
        true
    );

    if (n_tokens < 0) {
        return -1;
    }
    tokens.resize(n_tokens);

    // Clear KV cache
    llama_kv_cache_clear(g_ctx);

    // Create batch
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], i, {0}, i == n_tokens - 1);
    }

    if (llama_decode(g_ctx, batch) != 0) {
        llama_batch_free(batch);
        return -1;
    }

    // Get embeddings
    int n_embd = llama_model_n_embd(g_model);
    float* embeddings = llama_get_embeddings(g_ctx);

    if (!embeddings) {
        llama_batch_free(batch);
        return -1;
    }

    int n_copy = (n_embd < max_dim) ? n_embd : max_dim;
    memcpy(embedding_out, embeddings, n_copy * sizeof(float));

    llama_batch_free(batch);
    return n_embd;
}

/**
 * Free resources
 */
EMSCRIPTEN_KEEPALIVE
void llama_free_resources() {
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    llama_backend_free();
}

/**
 * Get model info
 */
EMSCRIPTEN_KEEPALIVE
int llama_get_n_ctx() {
    return g_ctx ? llama_n_ctx(g_ctx) : 0;
}

EMSCRIPTEN_KEEPALIVE
int llama_get_n_embd() {
    return g_model ? llama_model_n_embd(g_model) : 0;
}

EMSCRIPTEN_KEEPALIVE
int llama_get_n_vocab() {
    return g_model ? llama_model_n_vocab(g_model) : 0;
}

} // extern "C"
