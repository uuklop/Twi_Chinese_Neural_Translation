# encoding: utf-8
from __future__ import unicode_literals, print_function

import json
import os
import io
import itertools
import numpy as np
import random
from time import time
import torch
import pickle
import shutil
import math

from torch.utils.tensorboard import SummaryWriter

import metrics as evaluator
import model as net
import optimizer as optim
import utils
from config import get_train_args


# ---------------------------------------------------------------------------
# Replacements for torchtext.data.iterator / torchtext.data.utils
# (torchtext is broken on newer PyTorch / NumPy builds)
# ---------------------------------------------------------------------------
def _batch(data_iter, batch_size, batch_size_fn=None):
    """Yield successive mini-batches respecting an optional word-count budget."""
    if batch_size_fn is None:
        batch_size_fn = lambda new, count, sofar: count
    minibatch, size_so_far = [], 0
    for ex in data_iter:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
    if minibatch:
        yield minibatch


def _pool(data_list, batch_size, key, batch_size_fn=None, random_shuffler=None):
    """Sort within large chunks, batch, then shuffle batches (torchtext pool)."""
    if random_shuffler is None:
        random_shuffler = lambda x: random.sample(x, len(x))
    for chunk in _batch(data_list, batch_size * 100, batch_size_fn):
        batches = list(_batch(sorted(chunk, key=key), batch_size, batch_size_fn))
        for b in random_shuffler(batches):
            yield b


def _interleave_keys(a, b):
    """Combine two length keys so similar (src_len, tgt_len) pairs sort together."""
    def interleave(vals):
        bits = [format(v, '016b') for v in vals]
        return ''.join(x for pair in zip(*bits) for x in pair)
    return int(interleave([a, b]), base=2)
# ---------------------------------------------------------------------------


def save_checkpoint(state, is_best, model_path_, best_model_path_):
    torch.save(state, model_path_)
    if is_best:
        shutil.copyfile(model_path_, best_model_path_)


def batch_size_func(new, count, sofar):
    # return sofar + len(new[0]) + len(new[1])
    return sofar + (2 * max(len(new[0]), len(new[1])))


def save_output(hypotheses, vocab, outf):
    # Save the Hypothesis to output file
    with io.open(outf, 'w') as fp:
        for sent in hypotheses:
            words = [vocab[y] for y in sent]
            fp.write(' '.join(words) + '\n')


def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % n_params)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
    print('encoder: ', enc)
    print('decoder: ', dec)


def report_func(epoch, batch, num_batches, start_time, report_stats,
                report_every):
    """
    This is the user-defined batch-level training progress
    report function.
    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % report_every == -1 % report_every:
        report_stats.output(epoch, batch + 1, num_batches, start_time)
        report_stats = utils.Statistics()

    return report_stats


class CalculateBleu(object):
    def __init__(self, model, test_data, key, batch=50, max_length=50,
                 beam_size=1, alpha=0.6, max_sent=None):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = -1
        self.max_length = max_length
        self.beam_size = beam_size
        self.alpha = alpha
        self.max_sent = max_sent

    def __call__(self):
        self.model.eval()
        references = []
        hypotheses = []
        for i in range(0, len(self.test_data), self.batch):
            sources, targets = zip(*self.test_data[i:i + self.batch])
            references.extend(t.tolist() for t in targets)
            if self.beam_size > 1:
                ys = self.model.translate(sources,
                                          self.max_length,
                                          beam=self.beam_size,
                                          alpha=self.alpha)
            else:
                ys = [y.tolist() for y in
                      self.model.translate(sources,
                                           self.max_length,
                                           beam=False)]
            hypotheses.extend(ys)

            if self.max_sent is not None and \
                    ((i + 1) > self.max_sent):
                break

            # Log Progress
            if self.max_sent is not None:
                den = self.max_sent
            else:
                den = len(self.test_data)
            print("> Completed: [ %d / %d ]" % (i, den), end='\r')

        bleu = evaluator.BLEUEvaluator().evaluate(references, hypotheses)
        print('BLEU:', bleu.score_str())
        print('')
        return bleu.bleu, hypotheses


def _decode_ids(ids, id2w, sp=None):
    """Convert a list of token IDs to a readable string.

    sp (SentencePieceProcessor | None):
        If given, apply SPM decode (Twi BPE output).
        If None, join adjacent Chinese characters and leave other tokens space-separated.
    """
    special = {0, 1, 2, 3}  # PAD EOS UNK BOS
    tokens = [id2w.get(i, '<unk>') for i in ids if i not in special]
    if sp is not None:
        return sp.decode(tokens)
    # join adjacent Chinese characters
    joined = []
    for tok in tokens:
        if (joined and tok
                and utils._is_chinese_char(tok[0])
                and utils._is_chinese_char(joined[-1][-1])):
            joined[-1] += tok
        else:
            joined.append(tok)
    return ''.join(joined) if all(utils._is_chinese_char(c)
                                  for t in joined for c in t if t) \
        else ' '.join(joined)


def _tb_sample_translations(writer, model, samples_fwd, samples_rev,
                             id2w, sp, step, beam_size=4):
    """Translate a small fixed set of val pairs and log them to TensorBoard Text."""
    model.eval()
    rows_fwd, rows_rev = [], []

    for src_ids, tgt_ids in samples_fwd:
        with torch.no_grad():
            hyp = model.translate([src_ids], max_length=100,
                                  beam=beam_size, alpha=0.6)[0]
        src_str = _decode_ids(src_ids.tolist(), id2w, sp=sp)   # BPE Twi → raw
        ref_str = _decode_ids(tgt_ids.tolist(), id2w, sp=None) # char Chinese
        hyp_str = _decode_ids(hyp,              id2w, sp=None)
        rows_fwd.append(f"| {src_str} | {ref_str} | {hyp_str} |")

    for src_ids, tgt_ids in samples_rev:
        with torch.no_grad():
            hyp = model.translate([src_ids], max_length=100,
                                  beam=beam_size, alpha=0.6)[0]
        src_str = _decode_ids(src_ids.tolist(), id2w, sp=None) # char Chinese
        ref_str = _decode_ids(tgt_ids.tolist(), id2w, sp=sp)   # BPE Twi → raw
        hyp_str = _decode_ids(hyp,              id2w, sp=sp)
        rows_rev.append(f"| {src_str} | {ref_str} | {hyp_str} |")

    header = "| Source | Reference | Hypothesis |\n|--------|-----------|------------|"
    if rows_fwd:
        writer.add_text('Translations/Twi_to_Chinese',
                        header + '\n' + '\n'.join(rows_fwd), step)
    if rows_rev:
        writer.add_text('Translations/Chinese_to_Twi',
                        header + '\n' + '\n'.join(rows_rev), step)


def _tb_log_histograms(writer, model, step):
    """Log weight and gradient histograms for every named parameter."""
    for name, param in model.named_parameters():
        safe = name.replace('.', '/')
        writer.add_histogram(f'Weights/{safe}', param.data.float(), step)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{safe}', param.grad.data.float(), step)


def _append_metrics(path, record):
    """Append one JSON record (one eval step) to a JSONL metrics file."""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')


def main():
    best_score = 0
    args = get_train_args()
    print(json.dumps(args.__dict__, indent=4))
    utils.set_device(args.gpu)

    # Automatic Mixed Precision — enabled whenever a CUDA GPU is in use
    use_amp = args.gpu >= 0 and torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Reading the int indexed text dataset
    train_data = np.load(os.path.join(args.input, args.data + ".train.npy"), allow_pickle=True)
    train_data = train_data.tolist()
    dev_data = np.load(os.path.join(args.input, args.data + ".valid.npy"), allow_pickle=True)
    dev_data = dev_data.tolist()
    test_data = np.load(os.path.join(args.input, args.data + ".test.npy"), allow_pickle=True)
    test_data = test_data.tolist()

    # Reading the vocab file
    with open(os.path.join(args.input, args.data + '.vocab.pickle'),
              'rb') as f:
        id2w = pickle.load(f)

    args.id2w = id2w
    args.n_vocab = len(id2w)

    # Define Model
    model = net.Transformer(args)

    tally_parameters(model)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    print(model)

    optimizer = optim.TransformerAdamTrainer(model, args)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.model_file):
            print("=> loading checkpoint '{}'".format(args.model_file))
            checkpoint = torch.load(args.model_file, weights_only=False)
            args.start_epoch = checkpoint['epoch']
            best_score = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.steps = checkpoint.get('optimizer_steps', 0)
            if use_amp and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})".
                  format(args.model_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.model_file))

    # Reverse val/test by swapping (src, tgt) — no extra files needed
    dev_data_rev  = [(t, s) for s, t in dev_data]
    test_data_rev = [(t, s) for s, t in test_data]

    src_data, trg_data = list(zip(*train_data))
    total_src_words = len(list(itertools.chain.from_iterable(src_data)))
    total_trg_words = len(list(itertools.chain.from_iterable(trg_data)))
    iter_per_epoch = (total_src_words + total_trg_words) // args.wbatchsize
    print('Approximate number of iter/epoch =', iter_per_epoch)
    time_s = time()

    # ── TensorBoard ────────────────────────────────────────────────────────────
    tb_dir = os.path.join(args.out, 'runs')
    writer = SummaryWriter(log_dir=tb_dir)
    print(f'TensorBoard logs → {tb_dir}')
    print(f'  Launch with:  tensorboard --logdir {tb_dir}')

    # Log hyperparameters once (visible in TensorBoard HPARAMS tab)
    hparam_dict = {
        'layers':          args.layers,
        'multi_heads':     args.multi_heads,
        'n_units':         args.n_units,
        'n_hidden':        args.n_units * 4,
        'dropout':         args.dropout,
        'label_smoothing': args.label_smoothing,
        'wbatchsize':      args.wbatchsize,
        'batchsize':       args.batchsize,
        'warmup_steps':    args.warmup_steps,
        'epoch':           args.epoch,
        'beam_size':       args.beam_size,
        'tied':            int(args.tied),
    }
    writer.add_hparams(hparam_dict, metric_dict={'BLEU/avg': 0.0},
                       run_name='hparams')

    # Fixed val samples for live translation snapshots (5 fwd + 5 rev)
    _n_samples = min(5, len(dev_data))
    tb_samples_fwd = dev_data[:_n_samples]          # (twi_ids, chi_ids)
    tb_samples_rev = dev_data_rev[:_n_samples]      # (chi_ids, twi_ids)

    # Load SPM for decoding Twi BPE output in sample translations
    _spm_path = getattr(args, 'spm_model', None)
    _sp_for_tb = None
    if _spm_path and os.path.exists(_spm_path):
        import sentencepiece as _spm_mod
        _sp_for_tb = _spm_mod.SentencePieceProcessor()
        _sp_for_tb.load(_spm_path)

    # Metrics log (JSONL — one record per eval step, safe to resume/append)
    metrics_path = os.path.join(args.out, 'metrics.jsonl')

    # Checkpoint ring buffer for averaging (last K checkpoints → averaged model)
    CKPT_RING_SIZE = 8
    ckpt_ring = []      # paths of recent eval-step checkpoints

    global_steps = 0
    for epoch in range(args.start_epoch, args.epoch):
        random.shuffle(train_data)
        train_iter = _pool(train_data,
                           args.wbatchsize,
                           key=lambda x: _interleave_keys(len(x[0]), len(x[1])),
                           batch_size_fn=batch_size_func)
        report_stats = utils.Statistics()
        train_stats = utils.Statistics()
        valid_stats = utils.Statistics()

        if args.debug:
            grad_norm = 0.
        for num_steps, train_batch in enumerate(train_iter):
            global_steps += 1
            model.train()
            optimizer.zero_grad()
            src_iter = list(zip(*train_batch))[0]
            src_words = len(list(itertools.chain.from_iterable(src_iter)))
            report_stats.n_src_words += src_words
            train_stats.n_src_words += src_words
            in_arrays = utils.seq2seq_pad_concat_convert(train_batch, -1)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss, stat = model(*in_arrays)
            scaler.scale(loss).backward()
            if args.debug:
                # unscale before measuring norm so the value is comparable
                scaler.unscale_(optimizer.optimizer)
                norm = utils.grad_norm(model.parameters())
                grad_norm += norm
                if global_steps % args.report_every == 0:
                    print("> Gradient Norm: %1.4f" % (grad_norm / (num_steps + 1)))
            grad_norm = optimizer.step(scaler=scaler)

            report_stats.update(stat)
            train_stats.update(stat)

            # ── per-step TensorBoard scalars ─────────────────────────────
            writer.add_scalar('Step/loss',      stat.loss / max(stat.n_words, 1), global_steps)
            writer.add_scalar('Step/ppl',       stat.ppl(),                        global_steps)
            writer.add_scalar('Step/grad_norm', grad_norm,                         global_steps)
            writer.add_scalar('Optimizer/learning_rate',
                              optimizer.optimizer.param_groups[0]['lr'],           global_steps)
            report_stats = report_func(epoch, num_steps, iter_per_epoch,
                                       time_s, report_stats, args.report_every)

            if (global_steps + 1) % args.eval_steps == 0:
                dev_iter = _pool(dev_data,
                                 args.wbatchsize,
                                 key=lambda x: _interleave_keys(len(x[0]), len(x[1])),
                                 batch_size_fn=batch_size_func)

                model.eval()
                for dev_batch in dev_iter:
                    in_arrays = utils.seq2seq_pad_concat_convert(dev_batch, -1)
                    with torch.no_grad():
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            _, stat = model(*in_arrays)
                    valid_stats.update(stat)

                print('Train perplexity: %g' % train_stats.ppl())
                print('Train accuracy: %g' % train_stats.accuracy())

                print('Validation perplexity: %g' % valid_stats.ppl())
                print('Validation accuracy: %g' % valid_stats.accuracy())

                print('Twi -> Chi BLEU:')
                bleu_fwd, _ = CalculateBleu(model,
                                            dev_data,
                                            'Dev Bleu (Twi->Chi)',
                                            batch=args.batchsize // 4,
                                            beam_size=args.beam_size,
                                            alpha=args.alpha,
                                            max_sent=args.max_sent_eval)()
                print('Chi -> Twi BLEU:')
                bleu_rev, _ = CalculateBleu(model,
                                            dev_data_rev,
                                            'Dev Bleu (Chi->Twi)',
                                            batch=args.batchsize // 4,
                                            beam_size=args.beam_size,
                                            alpha=args.alpha,
                                            max_sent=args.max_sent_eval)()
                bleu_fwd = bleu_fwd or 0.0
                bleu_rev = bleu_rev or 0.0
                avg_bleu = (bleu_fwd + bleu_rev) / 2.0
                print('Avg BLEU: %.4f' % (avg_bleu * 100))

                # ── log metrics to JSONL ──────────────────────────────────
                current_lr = optimizer.optimizer.param_groups[0]['lr']
                _append_metrics(metrics_path, {
                    'step':      global_steps,
                    'epoch':     epoch + 1,
                    'lr':        float(current_lr),
                    'train_ppl': float(train_stats.ppl()),
                    'train_acc': float(train_stats.accuracy()),
                    'val_ppl':   float(valid_stats.ppl()),
                    'val_acc':   float(valid_stats.accuracy()),
                    'bleu_fwd':  float(bleu_fwd * 100),
                    'bleu_rev':  float(bleu_rev * 100),
                    'avg_bleu':  float(avg_bleu * 100),
                })

                # ── TensorBoard eval-step logging ─────────────────────────
                writer.add_scalars('Perplexity', {
                    'train': train_stats.ppl(),
                    'val':   valid_stats.ppl(),
                }, global_steps)
                writer.add_scalars('Accuracy', {
                    'train': train_stats.accuracy(),
                    'val':   valid_stats.accuracy(),
                }, global_steps)
                writer.add_scalar('Train/loss',
                                  train_stats.loss / max(train_stats.n_words, 1),
                                  global_steps)
                writer.add_scalar('Val/loss',
                                  valid_stats.loss / max(valid_stats.n_words, 1),
                                  global_steps)
                writer.add_scalars('BLEU', {
                    'twi_to_chinese': bleu_fwd * 100,
                    'chinese_to_twi': bleu_rev * 100,
                    'average':        avg_bleu * 100,
                }, global_steps)
                writer.add_scalar('Optimizer/learning_rate', current_lr, global_steps)

                # Weight + gradient histograms
                _tb_log_histograms(writer, model, global_steps)

                # Live sample translations
                _tb_sample_translations(writer, model,
                                        tb_samples_fwd, tb_samples_rev,
                                        id2w, _sp_for_tb, global_steps,
                                        beam_size=min(args.beam_size, 4))

                writer.flush()

                if args.metric == "bleu":
                    score = avg_bleu
                elif args.metric == "accuracy":
                    score = valid_stats.accuracy()

                is_best = score > best_score
                best_score = max(score, best_score)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_score': best_score,
                    'optimizer': optimizer.state_dict(),
                    'optimizer_steps': optimizer.steps,
                    'scaler': scaler.state_dict() if use_amp else None,
                    'opts': args,
                }, is_best,
                    args.model_file,
                    args.best_model_file)

                # Ring checkpoint for weight averaging (keeps last CKPT_RING_SIZE)
                ring_path = os.path.join(args.out, f'ring_{global_steps}.ckpt')
                torch.save({'state_dict': model.state_dict()}, ring_path)
                ckpt_ring.append(ring_path)
                if len(ckpt_ring) > CKPT_RING_SIZE:
                    old = ckpt_ring.pop(0)
                    try:
                        os.remove(old)
                    except OSError:
                        pass

    # ── Checkpoint averaging ───────────────────────────────────────────────────
    # Average the last CKPT_RING_SIZE checkpoints instead of using a single best.
    # Equivalent to free ensembling; consistently adds +0.5–2 BLEU.
    if len(ckpt_ring) >= 2:
        print(f'Averaging {len(ckpt_ring)} recent checkpoints for final eval...')
        avg_sd = None
        for ring_path in ckpt_ring:
            sd = torch.load(ring_path, weights_only=False)['state_dict']
            if avg_sd is None:
                avg_sd = {k: v.float().clone() for k, v in sd.items()}
            else:
                for k in avg_sd:
                    avg_sd[k] += sd[k].float()
        n = float(len(ckpt_ring))
        ref_sd = torch.load(args.best_model_file, weights_only=False)['state_dict']
        avg_sd = {k: (avg_sd[k] / n).to(ref_sd[k].dtype) for k in avg_sd}
        model.load_state_dict(avg_sd)
        # Clean up ring files
        for ring_path in ckpt_ring:
            try:
                os.remove(ring_path)
            except OSError:
                pass
        print('Averaged checkpoint loaded.')
    else:
        checkpoint = torch.load(args.best_model_file, weights_only=False)
        print("=> loaded checkpoint '{}' (epoch {}, best score {})".
              format(args.best_model_file,
                     checkpoint['epoch'],
                     checkpoint['best_score']))
        model.load_state_dict(checkpoint['state_dict'])

    spm_model = getattr(args, 'spm_model', None)

    print('Dev Set BLEU Score (Twi -> Chi)')
    _, dev_hyp = CalculateBleu(model,
                               dev_data,
                               'Dev Bleu (Twi->Chi)',
                               batch=args.batchsize // 4,
                               beam_size=args.beam_size,
                               alpha=args.alpha)()
    save_output(dev_hyp, id2w, args.dev_hyp)
    utils.post_process_output(args.dev_hyp, spm_path=None)          # join Chi chars

    print('Dev Set BLEU Score (Chi -> Twi)')
    _, dev_hyp_rev = CalculateBleu(model,
                                   dev_data_rev,
                                   'Dev Bleu (Chi->Twi)',
                                   batch=args.batchsize // 4,
                                   beam_size=args.beam_size,
                                   alpha=args.alpha)()
    save_output(dev_hyp_rev, id2w, args.dev_hyp_rev)
    utils.post_process_output(args.dev_hyp_rev, spm_path=spm_model) # decode Twi BPE

    print('Test Set BLEU Score (Twi -> Chi)')
    _, test_hyp = CalculateBleu(model,
                                test_data,
                                'Test Bleu (Twi->Chi)',
                                batch=args.batchsize // 4,
                                beam_size=args.beam_size,
                                alpha=args.alpha)()
    save_output(test_hyp, id2w, args.test_hyp)
    utils.post_process_output(args.test_hyp, spm_path=None)          # join Chi chars

    print('Test Set BLEU Score (Chi -> Twi)')
    _, test_hyp_rev = CalculateBleu(model,
                                    test_data_rev,
                                    'Test Bleu (Chi->Twi)',
                                    batch=args.batchsize // 4,
                                    beam_size=args.beam_size,
                                    alpha=args.alpha)()
    save_output(test_hyp_rev, id2w, args.test_hyp_rev)
    utils.post_process_output(args.test_hyp_rev, spm_path=spm_model) # decode Twi BPE

    writer.close()


if __name__ == '__main__':
    main()
