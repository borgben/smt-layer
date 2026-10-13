"""Controlled core baseline and K=1..8 Sudoku experiment; run with python -m scripts.marco_sweep."""
import copy
import csv
from pathlib import Path
import random
import time

import click
import numpy as np
import torch
import z3

from scripts.mnist_sudoku import (
    MNISTSudokuSolver, get_dataloader, pretrain, train_epoch, test_epoch,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    z3.set_param('smt.random_seed', seed)
    z3.set_param('sat.random_seed', seed)


def run_sweep(output, seed=0, pct=0.5, pretrain_epochs=30, epochs=5,
              batch_size=1, lr=1.0, clip_norm=0.1, pretrained_checkpoint=None):
    seed_all(seed)
    pretrain_load, _ = get_dataloader(0.9, batch_size=256, seed=seed)
    train_load, test_load = get_dataloader(pct, batch_size=batch_size, seed=seed)
    model = MNISTSudokuSolver(size=9).cuda()
    if pretrained_checkpoint is not None:
        model.load_state_dict(torch.load(pretrained_checkpoint, map_location='cpu', weights_only=True))
        checkpoint = Path(pretrained_checkpoint)
    else:
        if pretrain_epochs:
            optimizer = torch.optim.Adam([
                {'params': model.extractor.parameters(), 'lr': 1.e-3},
                {'params': model.decoder.parameters(), 'lr': 1.e-3},
            ])
            pretrain(model, optimizer, pretrain_load, pretrain_epochs, clip_norm=None)
        checkpoint = Path(output).with_suffix('.pretrained.pt')
        torch.save(model.state_dict(), checkpoint)
    initial_state = copy.deepcopy(model.state_dict())
    del model

    fields = ['mode', 'K', 'seed', 'pct', 'pretrained_checkpoint', 'pretrain_epochs',
              'epochs', 'batch_size', 'lr', 'clip_norm', 'train_accuracy',
              'train_loss', 'test_accuracy', 'test_loss', 'training_seconds',
              'marco_enumeration_seconds', 'muses_found', 'mean_mus_size',
              'mean_union_size', 'marco_calls', 'fallbacks']
    with open(output, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()
        for k in [None, *range(1, 9)]:
            mode = 'core' if k is None else 'marco_union'
            seed_all(seed)
            model = MNISTSudokuSolver(size=9, marco_union_backward=k is not None,
                                      marco_max_muses=k if k is not None else 8).cuda()
            model.load_state_dict(initial_state)
            # Construction consumes random numbers; reset again so each K sees
            # identical loader order and random batch splitting during training.
            seed_all(seed)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, lr, epochs=epochs, steps_per_epoch=len(train_load),
                pct_start=1. / float(epochs))
            training_seconds = 0.0
            # Fixed epoch budget: no accuracy-triggered early stopping.
            for epoch in range(1, epochs + 1):
                torch.cuda.synchronize()
                started = time.perf_counter()
                train_acc, train_loss, _ = train_epoch(
                    epoch, model, optimizer, train_load, clip_norm=clip_norm,
                    sched=sched, do_maxsat_forward=False)
                torch.cuda.synchronize()
                training_seconds += time.perf_counter() - started
                test_acc, test_loss = test_epoch(epoch, model, test_load)
            stats = model.sat.marco_stats
            row = dict(mode=mode, K=k, seed=seed, pct=pct, pretrained_checkpoint=str(checkpoint),
                       pretrain_epochs=pretrain_epochs if pretrained_checkpoint is None else '',
                       epochs=epochs, batch_size=batch_size, lr=lr, clip_norm=clip_norm,
                       train_accuracy=float(train_acc), train_loss=float(train_loss),
                       test_accuracy=float(test_acc), test_loss=float(test_loss),
                       training_seconds=training_seconds,
                       marco_enumeration_seconds=stats['enumeration_seconds'],
                       muses_found=stats['muses_found'],
                       mean_mus_size=stats['mus_size'] / stats['muses_found'] if stats['muses_found'] else 0.,
                       mean_union_size=stats['union_size'] / stats['calls'] if stats['calls'] else 0.,
                       marco_calls=stats['calls'], fallbacks=stats['fallbacks'])
            writer.writerow(row)
            handle.flush()
            click.echo('{} K={} completed: {}'.format(mode, k, row))
            del model, optimizer, sched


@click.command()
@click.option('--output', default='marco-sweep.csv', type=click.Path(path_type=Path), show_default=True)
@click.option('--seed', default=0, type=click.IntRange(0, 2**32-1), show_default=True)
@click.option('--pct', default=0.5, type=click.FloatRange(0., 0.9, min_open=True), show_default=True)
@click.option('--pretrain_epochs', default=30, type=click.IntRange(min=0), show_default=True)
@click.option('--epochs', default=5, type=click.IntRange(min=2), show_default=True)
@click.option('--batch_size', default=1, type=click.IntRange(min=1), show_default=True)
@click.option('--lr', default=1.0, type=click.FloatRange(min=0., min_open=True), show_default=True)
@click.option('--clip_norm', default=0.1, type=click.FloatRange(min=0., min_open=True), show_default=True)
@click.option('--pretrained_checkpoint', type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help='Existing full MNISTSudokuSolver state_dict; otherwise pretrain once.')
def main(**kwargs):
    if not torch.cuda.is_available():
        raise click.ClickException('Sudoku training requires CUDA; no CUDA device is available.')
    # Fail before expensive pretraining if the optional MARCO dependency is absent.
    from smtlayer.marco_union import enumerate_muses
    a = z3.Bool('sweep_dependency_check')
    list(enumerate_muses([z3.Not(a)], [a], 1))
    run_sweep(**kwargs)


if __name__ == '__main__':
    main()
