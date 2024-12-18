import pytest
from unittest.mock import MagicMock, patch

import sys, os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import parse_args, main, Model

def test_parse_args(monkeypatch):
    monkeypatch.setattr('sys.argv', ['model.py', '--train', 'data/train.jsonl.gz', '--test', 'data/test.jsonl.gz'])
    args = parse_args()
    assert args.train == 'data/train.jsonl.gz'
    assert args.test == 'data/test.jsonl.gz'

def test_model_evaluate():
    # Create a mock model
    model = Model()
    model.evaluate = MagicMock()
    model.load = MagicMock()

    # Mock test data
    test_data = "mock_test_data"
    model_path = "mock_model_path"

    # Call the evaluate method
    model.evaluate(test_data)
    model.evaluate.assert_called_once_with(test_data)

    # Call the load method and re-evaluate
    model.load(model_path)
    model.load.assert_called_once_with(model_path)
    model.evaluate(test_data)
    assert model.evaluate.call_count == 2

@patch('model.Model')
def test_main(mock_model_class, monkeypatch):
    # Mock the model instance
    mock_model = mock_model_class.return_value
    mock_model.evaluate = MagicMock()
    mock_model.load = MagicMock()

    # Mock the arguments
    monkeypatch.setattr('sys.argv', ['model.py', '--train', 'data/train.jsonl.gz', '--test', 'data/test.jsonl.gz'])

    # Call the main function
    main(parse_args())

    # Verify the model methods were called
    mock_model.evaluate.assert_called()
    mock_model.load.assert_called()
