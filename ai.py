from openai import OpenAI

import keys

client = OpenAI(
    organization="org-6MzLQ9ZEmqSZHH6L1DZhFP4k",
    project="proj_j5qzsNihWPBmUVrCKrQumQTk",
    api_key=keys.OPENAI_API_KEY,
)
