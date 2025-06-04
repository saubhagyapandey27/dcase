import torch
import torch.nn as nn


class CutInputIntoSegmentsWrapper(nn.Module):
    def __init__(self, model, max_input_length, segment_length, hop_size):
        """
        Args:
            model (nn.Module): The PyTorch model to wrap.
            max_input_length (int): Maximum length of input the model can handle.
            segment_length (int): Length of each segment if input exceeds max_input_length.
            hop_size (int): Hop size for overlapping segmentation.
        """
        super().__init__()
        self.model = model
        self.max_input_length = max_input_length
        self.segment_length = segment_length
        self.hop_size = hop_size

    def forward(self, x):
        """Processes the input audio through the model, handling segmentation if needed."""
        batch_size, input_length = x.shape

        if input_length <= self.max_input_length:
            return self.model(x).unsqueeze(1)  # Add segment dimension

        # Split into overlapping segments
        segments = []
        indices = list(range(0, input_length - self.segment_length + 1, self.hop_size))
        for i in indices:
            segments.append(x[:, i:i + self.segment_length])

        segments = torch.stack(segments)  # Shape: (num_segments, batch_size, segment_length)
        outputs = self.model(segments.reshape(-1, self.segment_length))  # Process each segment
        outputs = outputs.view(len(indices), batch_size, -1).permute(1, 0, 2)   # Reshape back to (batch, num_segments , embedding_dim)

        # Return segments separately
        return outputs
