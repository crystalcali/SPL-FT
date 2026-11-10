# Verification · 2026-11-10

## Executed

- 20 automated tests passed with real-tokenizer and native-architecture tests enabled.
- Actual `LiquidAI/LFM3-1.7B` tokenizer and native generation-mask tests passed, including history-window supervision and masking user observations.
- A real 45,298-token TD1 trajectory from the corpus was normalized and split into 13 windows of at most 4,045 tokens for a 4,096-token budget. Five individual oversized assistant answers were reported as dropped rather than truncated.
- Tiny random LFM3 models containing convolution, sliding-window attention and full attention blocks ran real CPU training in both full weight and LoRA modes, saved optimizer/model checkpoints, loaded saved models/adapters and resumed training from a checkpoint. This checks the pinned APIs and original architecture; it does not measure full-model Spark performance.
- All 89 official Terminal-Bench task directories passed Harbor task validation after materializing the Hugging Face snapshot as real files. Configuration objects for both Terminus 2 and OpenCode passed the actual Harbor 0.25.0 schema.
- Browser checks covered selecting full training, the automatic 1e-5 learning rate, submitting and cancelling a training job, attaching a skill to a 3-task Original evaluation, and inspecting its saved configuration. Browser checks used a separate development fixture with no executing worker; production authentication was tested through the API test suite.
- JavaScript syntax and Python compilation passed.

## Not executed here

- Full-size LFM3 training on the user's DGX Spark GPU.
- Real Docker execution of Terminal-Bench tasks, AMD64 emulation on Spark and end-to-end OpenCode tool execution.
- A complete Terminal-Bench score or any measured improvement after training.

This development environment has a CPU and no Docker engine. Run the diagnostics, a short GPU training job, and a three-task benchmark on Spark before scheduling long experiments. There are no invented training metrics or benchmark scores in the application.
