"""Read actual nested Parquet through the production preparation path."""
import json
import pytest


def test_direct_parquet_prepare_preserves_nested_rows_and_holdouts(tmp_path, monkeypatch):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    import huggingface_hub
    from spl_ft.config import Experiment
    from spl_ft.data import prepare, normalize, parquet_rows

    rows = [{'conversations': [
        {'role': 'user', 'content': f'Task {i}'},
        {'role': 'assistant', 'content': 'Reasoning and result. ' * (100 + i)},
        {'role': 'user', 'content': 'Terminal observation. ' * 30},
        {'role': 'assistant', 'content': f'Checked task {i}'}]} for i in range(51)]
    source = tmp_path / 'nested.parquet'
    pq.write_table(pa.Table.from_pylist(rows), source, row_group_size=len(rows))
    assert list(parquet_rows(source)) == rows
    opened = []

    class FileSystem:
        def open(self, path, mode):
            assert '@fixture-revision/dataset_adapters/nested.parquet' in path
            opened.append(path)
            return source.open(mode)

    monkeypatch.setattr(huggingface_hub, 'HfFileSystem', FileSystem)
    monkeypatch.setattr(huggingface_hub.HfApi, 'list_repo_files', lambda *a, **k: ['dataset_adapters/nested.parquet'])
    monkeypatch.setattr(huggingface_hub.HfApi, 'dataset_info', lambda *a, **k: pytest.fail('Pinned source should not resolve a new revision'))
    config = Experiment(train_size=8, val_size=3, test_size=4,
                        dataset_revision='fixture-revision', td1_configs=['dataset_adapters'])
    first = prepare(config, tmp_path / 'workspace', emit=lambda *_: None)
    manifest = json.loads((first / 'manifest.json').read_text())
    assert manifest['scanned'] == 51 and manifest['invalid'] == 0
    assert len(manifest['split_ids']['train']) == 8
    second = prepare(config.model_copy(update={'train_size': 15}), tmp_path / 'workspace', emit=lambda *_: None)
    larger = json.loads((second / 'manifest.json').read_text())
    assert larger['split_ids']['test'] == manifest['split_ids']['test']
    assert larger['split_ids']['validation'] == manifest['split_ids']['validation']
    assert len(opened) == 2


def test_shard_order_and_revision_are_explicit(tmp_path, monkeypatch):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    import huggingface_hub
    from spl_ft.config import TD2
    from spl_ft.data import source_rows
    opened = []
    for name, value in [('train-a.parquet', 'a'), ('train-b.parquet', 'b')]:
        pq.write_table(pa.table({'value': [value]}), tmp_path / name)

    class FileSystem:
        def open(self, path, mode):
            opened.append(path)
            return (tmp_path / path.rsplit('/', 1)[-1]).open(mode)

    monkeypatch.setattr(huggingface_hub, 'HfFileSystem', FileSystem)
    rows = list(source_rows(TD2, 'default', 'revision-123',
                ['README.md', 'data/train-b.parquet', 'data/train-a.parquet'], emit=lambda *_: None))
    assert rows == [{'value': 'a'}, {'value': 'b'}]
    assert all('@revision-123/' in path for path in opened)
    with pytest.raises(ValueError, match='No Parquet shards'):
        list(source_rows(TD2, 'default', 'revision-123', [], emit=lambda *_: None))
