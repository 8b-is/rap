"""Exercise retrieval definitions without importing model/API/benchmark startup.

Uses NumPy and PyTorch from the benchmark dependencies; embeddings are synthetic.
"""
import ast
import contextlib
import io
from pathlib import Path
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None


def cosine(a, b):
    a = torch.as_tensor(a, dtype=torch.float32).reshape(-1, 2)
    b = torch.as_tensor(b, dtype=torch.float32).reshape(-1, 2)
    return torch.nn.functional.normalize(a, dim=1) @ torch.nn.functional.normalize(b, dim=1).T


class Encoder:
    def __init__(self, vectors=None):
        self.vectors = vectors or {}
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        return np.array([self.vectors.get(t, [1.0, 0.0]) for t in texts], dtype=np.float32)


def load_retrieval(benchmark, encoder=None, params=None):
    path = Path(__file__).resolve().parents[1] / benchmark / 'main.py'
    source = ast.parse(path.read_text(), filename=str(path))
    functions = [n for n in source.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('generate_embeddings', 'generate_examples')]
    if len(functions) != 2:
        raise AssertionError('retrieval definitions not found')
    ns = dict(np=np, torch=torch, cos_sim=cosine,
              model_embedding=encoder or Encoder(), config={'params': params or {}})
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), ns)
    return ns


@unittest.skipUnless(torch is not None, 'requires benchmark NumPy/PyTorch dependencies')
class TestBenchmarkRetrieval(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_empty_memory_does_not_call_encoder(self):
        for benchmark in ('alfworld', 'webshop'):
            with self.subTest(benchmark=benchmark):
                encoder = Encoder()
                ns = load_retrieval(benchmark, encoder)
                result = ns['generate_embeddings']([])
                self.assertEqual(result, {} if benchmark == 'alfworld' else ([], {}))
                self.assertEqual(encoder.calls, [])
                if benchmark == 'alfworld':
                    self.assertEqual(ns['generate_examples']({}, [], {}, for_plan=True), [])
                    self.assertEqual(ns['generate_examples']({}, [], {}), [])
                else:
                    self.assertEqual(ns['generate_examples']({}, [], [], {}, reasoning='keep'), ('', 'keep'))
                self.assertEqual(encoder.calls, [])

    def test_webshop_filters_can_remove_all_memories(self):
        for params, memory in (({}, [{'Reward': 0.0}]),
                               ({'success': True}, [{'Reward': 0.5, 'Success': False}])):
            with self.subTest(params=params):
                encoder = Encoder()
                ns = load_retrieval('webshop', encoder, params)
                self.assertEqual(ns['generate_embeddings'](memory), ([], {}))
                self.assertEqual(encoder.calls, [])

    def test_alfworld_uses_available_examples_in_both_paths(self):
        ns = load_retrieval('alfworld')
        memory = [dict(Task='task', Category='category', Plan='plan', Actions=['act', 'obs'])]
        embeddings = ns['generate_embeddings'](memory)
        query = dict(Task='task', Category='category', Plan='plan')
        for for_plan in (False, True):
            with self.subTest(for_plan=for_plan):
                hits = ns['generate_examples'](query, memory, embeddings, k=8, for_plan=for_plan, act_len=2, key='act')
                self.assertEqual(len(hits), 1)
                self.assertIn('task', hits[0])
                self.assertEqual(ns['generate_examples'](query, memory, embeddings, k=0, for_plan=for_plan), [])

    def test_webshop_uses_available_examples_in_both_paths(self):
        ns = load_retrieval('webshop')
        memory = [dict(Id='one', Instruction='task', Reward=1.0, Category='cat', Query='q', Actions=['task\n', 'act', 'obs'])]
        memory, embeddings = ns['generate_embeddings'](memory)
        for act_len in (0, 2):
            with self.subTest(act_len=act_len):
                prompt, _ = ns['generate_examples']({'instruction': 'task'}, [], memory, embeddings,
                                                   reasoning='act', k=8, act_len=act_len)
                self.assertIn('task', prompt)
                self.assertEqual(ns['generate_examples']({'instruction': 'task'}, [], memory, embeddings, k=0), ('', ''))

    def test_webshop_intra_task_scores_the_matched_observation(self):
        ns = load_retrieval('webshop', Encoder({'query_obs': [0.0, 1.0]}), {'success': True, 'intra_task': True})
        memory = [dict(Id=name, Actions=[name + '\n', 'act', 'obs']) for name in ('A', 'B')]
        embeddings = {'Instruction': np.array([[1., 0.], [1., 0.]]),
                      'Actions': [np.array([[1., 0.], [-1., 0.]]), np.array([[-1., 0.], [0., 1.]])]}
        actions = ['Action: reset', 'Observation: initial', 'Action: click', 'Observation: query_obs']
        prompt, _ = ns['generate_examples']({'instruction': 'task'}, actions, memory, embeddings,
                                           reasoning='query', k=1, act_len=1, use_act_obs=True)
        self.assertIn('B\n', prompt)
        self.assertNotIn('A\n', prompt)
        actions[-2] = 'Action: think about task'
        prompt, _ = ns['generate_examples']({'instruction': 'task'}, actions, memory, embeddings,
                                           reasoning='query', k=1, act_len=1, use_act_obs=True)
        self.assertIn('A\n', prompt)


if __name__ == '__main__':
    unittest.main()
