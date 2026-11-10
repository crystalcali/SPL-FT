"""Check the entire production recipe against the pinned Transformers API."""
import inspect
import pytest
from spl_ft.config import Experiment, presets
from spl_ft.training import training_argument_kwargs


@pytest.mark.parametrize('preset', ['terminal', 'long_horizon', 'reasoning'])
@pytest.mark.parametrize('method', ['lora', 'full'])
def test_complete_production_arguments_match_transformers_api(preset, method, tmp_path):
    from transformers import TrainingArguments
    config = Experiment(**{**presets()[preset], 'method': method})
    values = training_argument_kwargs(config, tmp_path)
    inspect.signature(TrainingArguments).bind(**values)
    assert TrainingArguments.get_warmup_steps(type('Args', (), values)(), 100) == 5
    assert values['output_dir'] == str(tmp_path / 'checkpoints')
