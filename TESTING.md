# Testing Guide

This document describes how to test the project components before deployment.

## Basic Functionality Test

Run the basic test to verify configuration loading, data format, and module imports:

```bash
python3 test_basic.py
```

Expected output:
```
Testing basic functionality...

✓ Config loading works
✓ Schema loading works
✓ Evaluation data loaded: 544 cases
✓ Module imports work

✓ All basic tests passed!
```

## Case Synthesis and Evaluation Test

To test case synthesis and evaluation with a custom API:

1. Edit `test_synthesis_eval.py` to configure your API endpoint:
```python
TEST_API_CONFIG = {
    "base_url": "https://your-api-endpoint/v1",
    "api_key": "your-api-key",
    "model": "your-model-name"
}
```

2. Run the test:
```bash
python3 test_synthesis_eval.py
```

This will test:
- API connection
- Synthesis prompt loading
- Schema validation
- Evaluation data format
- Simple case synthesis

## Component Tests

### Test Configuration Loading

```bash
python3 -c "
import yaml
config = yaml.safe_load(open('configs/pipeline.yaml'))
print('Config keys:', list(config.keys()))
"
```

### Test Evaluation Data

```bash
python3 -c "
import json
with open('evaluation/data/goodcase_final/benchmark.jsonl') as f:
    cases = [json.loads(line) for line in f]
print(f'Loaded {len(cases)} test cases')
print(f'Sample domains: {set(c[\"domain\"] for c in cases[:10])}')
"
```

### Test Module Imports

```bash
python3 -c "
import sys
sys.path.insert(0, '.')
from src import api_client
from src.synthesis import generate
from src.validation import schema_check
print('All modules imported successfully')
"
```

## Running Full Pipeline

### Case Synthesis

```bash
# Synthesize cases (requires API configuration)
python scripts/01_synthesize.py

# Validate generated cases
python scripts/02_validate.py
```

### Evaluation

```bash
# Evaluate a single case
python scripts/evaluate_one_case.py

# Batch evaluation
python scripts/03_evaluate.py
```

## Troubleshooting

### Missing Dependencies

If you encounter import errors, install missing packages:

```bash
pip install -r requirements.txt
```

### API Connection Issues

- Check your API endpoint and key configuration
- Verify network connectivity
- Check API rate limits

### Schema Validation Errors

The evaluation dataset may use a slightly different schema version than `schemas/scenario.json`. This is expected and doesn't affect evaluation functionality.

## CI/CD Integration

For automated testing in CI/CD pipelines:

```bash
# Run basic tests
python3 test_basic.py

# Run synthesis tests (if API is available)
python3 test_synthesis_eval.py
```

## Next Steps

After all tests pass:
1. Review generated documentation
2. Commit to version control
3. Push to GitHub
4. Set up CI/CD workflows (optional)
