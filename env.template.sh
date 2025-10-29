#!/usr/bin/env bash
# Copy this file to a location outside the repository (e.g., env.local.sh)
# and run `source /path/to/env.local.sh` to load the API credentials.
# The default constructors for the API clients will read these values when
# instantiating OpenAI and RCAC GenAI backends.

export OPENAI_API_KEY="replace-with-your-openai-api-key"
export RCAC_GENAI_API_KEY="replace-with-your-rcac-api-key"
