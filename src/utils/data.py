from datasets import load_dataset, Dataset
from pathlib import Path

def load_dataset_from_hub_or_local(dataset_id: str, **kwargs) -> Dataset:
    """Load a dataset from the Hugging Face Hub or from local files."""
    dataset_id_as_path = Path(dataset_id)
    
    # Check if this is a local JSONL file
    if dataset_id_as_path.exists() and dataset_id_as_path.is_file() and dataset_id_as_path.suffix == '.jsonl':
        # Check if we need to load split files
        split = kwargs.pop('split', None)
        if split:
            # Try to load split-specific file
            split_file = dataset_id_as_path.parent / f"{dataset_id_as_path.stem}_{split}.jsonl"
            if split_file.exists():
                # Load the split-specific file
                dataset = load_dataset('json', data_files=str(split_file), split='train', **kwargs)
            else:
                # Load the original file with the requested split
                # This assumes the file has been pre-split or will be split by the caller
                dataset = load_dataset('json', data_files=str(dataset_id_as_path), split='train', **kwargs)
        else:
            # Load without specific split
            dataset = load_dataset('json', data_files=str(dataset_id_as_path), **kwargs)
    else:
        # Load from Hugging Face Hub
        dataset = load_dataset(dataset_id, **kwargs)
    
    return dataset