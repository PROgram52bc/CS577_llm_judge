#!/usr/bin/env bash
#
# Template for configuring environment variables required by CS577 LLM Judge.
#
# Usage:
#   cp env.example.sh env.local.sh
#   # edit env.local.sh to add your real API keys
#   source env.local.sh
#
# You can keep your personalized copy outside the repository to avoid
# accidentally committing secrets.

# OpenAI-compatible endpoints use this variable when --llm-backend=openai.
export OPENAI_API_KEY="your-openai-api-key"

# Purdue RCAC GenAI endpoint key for --llm-backend=rcac.
export RCAC_GENAI_API_KEY="your-rcac-genai-api-key"
