import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    config.dataroot = "/tmp/test_data"
    config.checkpoints_dir = "/tmp/test_checkpoints"
    config.model = "test_model"
    config.dataset_mode = "unaligned"
    config.gpu_ids = []
    config.isTrain = False
    config.phase = "test"
    config.num_threads = 0
    config.batch_size = 1
    config.serial_batches = True
    config.no_flip = True
    config.display_id = -1
    config.preprocess = "resize_and_crop"
    config.load_size = 256
    config.crop_size = 256
    config.max_dataset_size = float("inf")
    return config


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor for testing."""
    import torch
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    dataset = Mock()
    dataset.__len__ = Mock(return_value=10)
    dataset.__getitem__ = Mock(return_value={
        'A': sample_image_tensor(),
        'B': sample_image_tensor(),
        'A_paths': ['/path/to/imageA.jpg'],
        'B_paths': ['/path/to/imageB.jpg']
    })
    return dataset


@pytest.fixture
def mock_dataloader(mock_dataset):
    """Create a mock dataloader for testing."""
    dataloader = Mock()
    dataloader.dataset = mock_dataset
    dataloader.__iter__ = Mock(return_value=iter([mock_dataset[0]]))
    return dataloader


@pytest.fixture
def cleanup_gpu():
    """Cleanup GPU memory after tests."""
    yield
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def mock_visdom():
    """Mock visdom for visualization tests."""
    with patch('visdom.Visdom') as mock:
        instance = Mock()
        mock.return_value = instance
        instance.check_connection.return_value = True
        yield instance


@pytest.fixture
def sample_options():
    """Create sample command line options."""
    class Options:
        def __init__(self):
            self.dataroot = './datasets/test'
            self.name = 'test_experiment'
            self.model = 'cycle_gan'
            self.checkpoints_dir = './checkpoints'
            self.results_dir = './results'
            self.gpu_ids = []
            self.isTrain = False
            self.phase = 'test'
            self.batch_size = 1
            self.serial_batches = True
            self.num_threads = 0
            self.load_size = 256
            self.crop_size = 256
            self.max_dataset_size = float("inf")
            self.preprocess = 'resize_and_crop'
            self.no_flip = True
            self.display_id = -1
            self.verbose = False
    
    return Options()


@pytest.fixture(autouse=True)
def reset_imports():
    """Reset certain imports to avoid state pollution between tests."""
    yield
    import sys
    modules_to_reset = [m for m in sys.modules.keys() if m.startswith(('models.', 'data.', 'options.', 'util.'))]
    for module in modules_to_reset:
        sys.modules.pop(module, None)


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured_output)
    yield captured_output


@pytest.fixture
def mock_model_class():
    """Create a mock model class for testing."""
    model = Mock()
    model.setup = Mock()
    model.eval = Mock()
    model.test = Mock()
    model.get_current_visuals = Mock(return_value={'real_A': sample_image_tensor(), 'fake_B': sample_image_tensor()})
    model.get_image_paths = Mock(return_value=['/path/to/image.jpg'])
    return model