# Data Card

## Task

The primary task is semantic direct prompt-injection attempt detection. A positive input contains an attempt to override, bypass, replace, or manipulate higher-priority instructions or hidden context, regardless of whether an attack would succeed in a downstream application.

## Included Dataset Preparation

The preparation script downloads public datasets at reproduction time:

- `hendzh/PromptShield`: primary semantic benchmark.
- `deepset/prompt-injections`: external semantic benchmark.
- `leolee99/NotInject`: benign hard-negative stress set.
- `Lakera/gandalf_ignore_instructions`: attack-only recall stress set.

Downloaded dataset files are not included in this repository.

## Excluded Corpus Type

Outcome-labeled corpora are not used as primary semantic ground truth when labels reflect downstream attack success rather than prompt-injection attempt semantics.

## Privacy Controls

Dataset audit JSONs omit prompt excerpts by default. Use `--include-examples` only for local private inspection.
