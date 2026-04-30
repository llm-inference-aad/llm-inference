# Estimated Service-Only Inference Metrics

These are provisional, believable planning numbers for the current setup in this repo, not measured final results.

Assumptions:
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Hardware profile: single GPU, `VLLM_TENSOR_PARALLEL_SIZE=1`, `CUDA_VISIBLE_DEVICES=0`
- Service latency only: queue wait excluded
- Warm model process, no cold-start load time
- Typical Guided Evolution request shape: moderate prompt, moderate response, code/structured-output style generations
- Throughput assumes steady-state single-server operation on the same hardware

## Summary

| Scenario | Backend | Constraints | avg_service_latency_sec | service_latency_p50_sec | service_latency_p95_sec | service_latency_p99_sec | throughput_req_per_sec | throughput_req_per_min |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| without_vllm | HuggingFace standard | none | 9.7 | 9.1 | 14.6 | 17.3 | 0.103 | 6.18 |
| vllm_without_constrained_decoding | vLLM continuous batching | none | 4.6 | 4.3 | 7.0 | 8.4 | 0.214 | 12.84 |
| vllm_with_constrained_decoding | vLLM continuous batching | json | 5.8 | 5.4 | 8.7 | 10.5 | 0.171 | 10.26 |

## Readout

- `without_vllm`: believable baseline for standard Transformers-style serving on the same 8B model.
- `vllm_without_constrained_decoding`: roughly `2.1x` better throughput than non-vLLM and about `53%` lower average service latency.
- `vllm_with_constrained_decoding`: about `26%` slower than unconstrained vLLM on service latency, but still clearly better than non-vLLM.

## Speculative Decoding Add-On

If speculative decoding is layered on top of constrained vLLM, a believable step-up would be:

| Scenario | Backend | avg_service_latency_sec | throughput_req_per_sec | throughput_req_per_min |
| --- | --- | ---: | ---: | ---: |
| vllm_with_constrained_decoding | vLLM + JSON constraints | 5.8 | 0.171 | 10.26 |
| vllm_with_constrained_plus_speculative | vLLM + JSON constraints + speculative drafting | 4.7 | 0.212 | 12.72 |

This is only a modest improvement over constrained-only because the constraint checks still gate token acceptance.

## Script-Aligned JSON View

```json
{
  "estimated": true,
  "service_time_only": true,
  "queue_time_included": false,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "hardware_profile": {
    "tensor_parallel_size": 1,
    "cuda_visible_devices": "0"
  },
  "scenarios": [
    {
      "name": "without_vllm",
      "backend": "HuggingFace (standard)",
      "constraint_type": null,
      "avg_service_latency_sec": 9.7,
      "service_latency_median_sec": 9.1,
      "service_latency_p95_sec": 14.6,
      "service_latency_p99_sec": 17.3,
      "throughput_req_per_sec": 0.103,
      "throughput_req_per_min": 6.18
    },
    {
      "name": "vllm_without_constrained_decoding",
      "backend": "vLLM (continuous batching)",
      "constraint_type": null,
      "avg_service_latency_sec": 4.6,
      "service_latency_median_sec": 4.3,
      "service_latency_p95_sec": 7.0,
      "service_latency_p99_sec": 8.4,
      "throughput_req_per_sec": 0.214,
      "throughput_req_per_min": 12.84
    },
    {
      "name": "vllm_with_constrained_decoding",
      "backend": "vLLM (continuous batching)",
      "constraint_type": "json",
      "avg_service_latency_sec": 5.8,
      "service_latency_median_sec": 5.4,
      "service_latency_p95_sec": 8.7,
      "service_latency_p99_sec": 10.5,
      "throughput_req_per_sec": 0.171,
      "throughput_req_per_min": 10.26
    },
    {
      "name": "vllm_with_constrained_plus_speculative",
      "backend": "vLLM (continuous batching)",
      "constraint_type": "json+speculative",
      "avg_service_latency_sec": 4.7,
      "service_latency_median_sec": 4.5,
      "service_latency_p95_sec": 7.0,
      "service_latency_p99_sec": 8.3,
      "throughput_req_per_sec": 0.212,
      "throughput_req_per_min": 12.72
    }
  ]
}
```

## Use

Use this only as a temporary stand-in until real run artifacts are available from `evaluation_report.json` or request-level metrics JSON.