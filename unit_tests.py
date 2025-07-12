import unittest
import torch
from torch import nn
from network.mlp import MLP

class TestMLP(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_vectors = 5
        self.input_size = 16
        self.hidden_size = 32
        self.output_size = 8

        self.model = MLP(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        )

    def test_output_shape(self):
        x = torch.randn(self.batch_size, self.num_vectors, self.input_size)
        output = self.model(x)

        expected_shape = (self.batch_size, self.num_vectors, self.output_size)
        self.assertEqual(output.shape, expected_shape)

    def test_output_type(self):
        x = torch.randn(self.batch_size, self.num_vectors, self.input_size)
        output = self.model(x)

        self.assertIsInstance(output, torch.Tensor)

    def test_requires_grad(self):
        x = torch.randn(self.batch_size, self.num_vectors, self.input_size, requires_grad=True)
        output = self.model(x)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

    def test_forward_consistency(self):
        """Ensure the forward pass is deterministic with fixed weights and input."""
        torch.manual_seed(42)
        model = MLP(self.input_size, self.output_size, self.hidden_size)
        model.eval()  # Disable dropout/batchnorm if added later

        x = torch.ones(self.batch_size, self.num_vectors, self.input_size)
        output1 = model(x)
        output2 = model(x)

        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
