# DSL Conversion Plan

This file tracks the phased migration of examples in this repository to the newest `MLX::DSL` APIs from `../../mlx-ruby/master`.

## Status Legend

- `[ ]` not started
- `[-]` in progress
- `[x]` completed

## Phases

- [x] Phase 0: Runtime alignment and baseline verification
  - Use `../../mlx-ruby/master/lib` as runtime source.
  - Validate CPU test execution path.
  - Ensure BERT weights/config exist locally for parity test execution.
- [x] Phase 1: Low-risk trainer adoption
  - Targets: `mnist/main.rb`, `clip/linear_probe.rb`, `normalizing_flow/main.rb`
  - Replace manual loops with `model.trainer(...).fit_report(...)` where practical.
  - Introduce `MLX::DSL::Data` pipelines for train batching/shuffle ergonomics.
- [x] Phase 2: Validation/checkpoint ergonomics
  - Targets: `speechcommands/main.rb`, `cvae/main.rb`, `gcn/main.rb`
  - Replace ad-hoc best-checkpoint logic with DSL monitor/checkpoint flows.
  - Adopt `artifact_policy`, monitor/patience/min_delta, and lifecycle hooks.
- [x] Phase 3: Dataflow and split-plan reuse
  - Targets: train/validation/test examples with repetitive fit kwargs.
  - Introduce `MLX::DSL.splits` and `register_dataflow` / `use_dataflow`.
- [x] Phase 4: Advanced examples partial DSL adoption
  - Targets: `cifar/main.rb`, `transformer_lm/main.rb`, `lora/lora.rb`, `flux/dreambooth.rb`
  - Keep custom inner loops where needed; move orchestration/reporting/checkpointing to DSL.
- [x] Phase 5: Model declaration ergonomics
  - Expand `MLX::DSL::Model` macro/builder usage in examples where it reduces boilerplate.
- [x] Phase 6: Experiment and artifact standardization
  - Introduce `MLX::DSL.experiment` and run-bundle/resume conventions in selected examples.
- [x] Phase 7: Docs and test refresh
  - Update example docs to reflect DSL-first flows.
  - Update tests to validate new DSL-driven behavior.

## Execution Log

- 2026-02-14: Created migration tracker and marked Phase 0 complete.
- 2026-02-14: Completed Phase 1 migration for `mnist/main.rb`, `clip/linear_probe.rb`, and `normalizing_flow/main.rb` with `MLX::DSL::Trainer` + `MLX::DSL::Data` flows.
- 2026-02-14: Completed Phase 2 migration for `speechcommands/main.rb`, `cvae/main.rb`, and `gcn/main.rb` using DSL monitor/patience/checkpoint policies.
- 2026-02-14: Completed Phase 3 by introducing `MLX::DSL.splits` + `register_dataflow`/`use_dataflow` in `mnist/main.rb`, `normalizing_flow/main.rb`, `speechcommands/main.rb`, and `gcn/main.rb`.
- 2026-02-14: Completed Phase 4 partial DSL adoption in `cifar/main.rb`, `transformer_lm/main.rb`, `lora/lora.rb`, and `flux/dreambooth.rb`.
- 2026-02-14: Completed Phase 5 by migrating selected model declarations to `MLX::DSL::Model` macros (`mnist/main.rb`, `normalizing_flow/flows.rb`, `clip/linear_probe.rb`).
- 2026-02-14: Completed Phase 6 with `MLX::DSL.experiment` adoption in `mnist/main.rb` and run-bundle/resume artifact-policy conventions in `mnist/main.rb` and `gcn/main.rb`.
- 2026-02-14: Completed Phase 7 docs/test refresh across migrated examples; tests now assert DSL helper availability in updated models.
