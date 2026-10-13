import unittest
from unittest.mock import patch

import torch

from scripts.mnist_sudoku import get_dataloader


class SudokuSeedTests(unittest.TestCase):
    def loaders(self, seed=None, pct=0.5):
        ids = torch.arange(10000)
        with patch('scripts.mnist_sudoku.load', return_value=None), patch(
            'scripts.mnist_sudoku.process_inputs', return_value=(ids, ids + 10000, ids + 20000)
        ):
            return get_dataloader(pct, seed=seed)

    def test_seed_reproduces_split_and_keeps_tensors_aligned(self):
        train, test = self.loaders(seed=0)
        again, again_test = self.loaders(seed=0)
        other, other_test = self.loaders(seed=1)
        for loader, repeat, changed in ((train, again, other), (test, again_test, other_test)):
            images, masks, labels = loader.dataset.tensors
            self.assertTrue(torch.equal(images, repeat.dataset.tensors[0]))
            self.assertFalse(torch.equal(images, changed.dataset.tensors[0]))
            self.assertTrue(torch.equal(masks, images + 20000))
            self.assertTrue(torch.equal(labels, images + 10000))
        self.assertEqual(len(train.dataset), 5000)
        self.assertEqual(len(test.dataset), 1000)
        self.assertTrue(set(train.dataset.tensors[0].tolist()).isdisjoint(test.dataset.tensors[0].tolist()))

    def test_pretraining_excludes_test_set(self):
        pretrain, _ = self.loaders(seed=0, pct=0.9)
        train, test = self.loaders(seed=0)
        pretrain_ids = set(pretrain.dataset.tensors[0].tolist())
        self.assertTrue(set(train.dataset.tensors[0].tolist()).issubset(pretrain_ids))
        self.assertTrue(pretrain_ids.isdisjoint(test.dataset.tensors[0].tolist()))

    def test_no_seed_preserves_original_slices(self):
        train, test = self.loaders()
        self.assertTrue(torch.equal(train.dataset.tensors[0], torch.arange(5000)))
        self.assertTrue(torch.equal(test.dataset.tensors[0], torch.arange(9000, 10000)))


if __name__ == '__main__':
    unittest.main()
