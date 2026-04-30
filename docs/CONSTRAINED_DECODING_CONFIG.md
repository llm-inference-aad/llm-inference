# Environment Configuration for Constrained Decoding

# Enable/Disable constrained decoding support
# This is automatically enabled if vLLM is available
# CONSTRAINED_DECODING_ENABLED=true

# Default constraint type for batch requests (optional)
# If set, constrains all requests unless overridden per-request
# Values: 'grammar', 'json', 'regex', or empty (unconstrained)
# DEFAULT_CONSTRAINT_TYPE=json

# Default JSON schema for all requests (if DEFAULT_CONSTRAINT_TYPE=json)
# This is a stringified JSON that will be parsed by the server
# Example for structured output:
# DEFAULT_JSON_SCHEMA={"type":"object","properties":{"output":{"type":"string"}}}

# Constraint validation timeout (seconds)
# Some constraints can be slow to validate; set a timeout to avoid hanging requests
# CONSTRAINT_VALIDATION_TIMEOUT=30

# Enable constraint enforcement logging
# Set to 'true' to log all constraint-related decisions
# CONSTRAINT_LOGGING_ENABLED=false

# Maximum grammar constraint complexity
# Prevents overly complex constraints that could cause server slowdown
# CONSTRAINT_MAX_GRAMMAR_SIZE=5000

# --- Optional: speculative decoding on top of constraints (vLLM only) ---
# ENABLE_SPECULATIVE_DECODING=true
# VLLM_SPECULATIVE_MODEL=/path/to/draft/model
# VLLM_NUM_SPECULATIVE_TOKENS=4
# VLLM_SPECULATIVE_METHOD=draft_model
#
# Adaptive speculation (recommended): if true and method is unset,
# server selects vLLM suffix speculation to adapt speculation depth
# dynamically instead of a fixed draft length.
# VLLM_ADAPTIVE_SPECULATION=true
# VLLM_SUFFIX_DECODING_MAX_TREE_DEPTH=24
# VLLM_SUFFIX_DECODING_MAX_SPEC_FACTOR=1.0
# VLLM_SUFFIX_DECODING_MIN_TOKEN_PROB=0.1
#
# For ngram speculation (no draft model), use:
# VLLM_SPECULATIVE_METHOD=ngram
# VLLM_SPECULATIVE_PROMPT_LOOKUP_MIN=2
# VLLM_SPECULATIVE_PROMPT_LOOKUP_MAX=5
#
# Advanced: raw JSON object merged into vLLM speculative_config
# VLLM_SPECULATIVE_CONFIG={"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}

# Acceptance-oriented tuning for constrained + speculative decoding:
# SPECULATIVE_CONSTRAINED_TEMPERATURE_CAP=0.25
# SPECULATIVE_CONSTRAINED_TOP_P_CAP=0.9
# ENABLE_PROFILED_BATCH_GROUPING=true
