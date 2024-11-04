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

## Agent Workflow

1. get piece of possibly-buggy code
2. generate a list of possible bugs in the code using  the LLM
3. for each possible bug:
   1. generate a query to StackOverflow that is related to the bug using the LLM
   2. send the query to StackOverflow, and recieve a list of related questions and answers
   3. using that context, determine what the bug is in the code
   4. maybe: chain-of-thought (or something) using the LLM to refine what the
      identified bug is (or if there is one in the first place)
4. done
5. maybe: suggest fixes for bugs

## Future Work

- use SemGrep as static analyzer instead of calling StackOverflow

## Stackexchange API

throttle: don't send more than 30 requests per minute