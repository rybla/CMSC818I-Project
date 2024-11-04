# Notes

## Overview

- LLM agent
  - gets a piece of code as input
  - use the LLM to generate a list of possible vulnerabilities
  - has access to _StackOverflow_
  - goal: find the likely bugs in the code (and potentially suggest fixes)
- benchmark
  - open database of code vulnerabilities with examples
  - compare pre- and post- augmentation of base model (gpt-4-turbo)
  - test for true positives on examples of vulnerabilities
  - test for true negatives on the post-fix vulnerability examples

## Stackexchange API

throttle: don't send more than 30 requests per minute