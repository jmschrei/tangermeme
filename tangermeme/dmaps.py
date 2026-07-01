from tangermeme.utils import one_hot_encode, _device

import numpy as np

import torch
import torch.nn as nn

def calculate_saliency_map(
        model: nn.Module, 
        input_tensor: torch.Tensor, 
        device) -> np.ndarray:
    """
    Calculates the saliency map (gradient of output w.r.t. input) for a given input.
    """
    model.eval()
    model.to(device)
    
    input_tensor.requires_grad_()
    
    # Forward pass
    prediction = model(input_tensor.to(device)) 
    
    # Backward pass to get gradients
    # We use a gradient of `1` for the scalar output to kickstart backpropagation.
    prediction.backward(torch.tensor([[1.0]]).to(device))
    
    # The gradients are our raw saliency map
    saliency = input_tensor.grad.data.abs()
    
    # To get a single importance score per position, we multiply the gradients
    # by the input tensor (to only keep gradients for the actual nucleotide present)
    # and then sum across the nucleotide dimension (dim=1).
    positional_saliency = (saliency * input_tensor).sum(dim=1).squeeze(0)
    
    return positional_saliency.detach().cpu().numpy()


def dependency_map(
        model: nn.Module,
        sequence: str
) -> np.ndarray:
    
    
    seq_len = len(sequence)
    device = _device()

    DNA_ALPHABET = 'ACGT'
    DNA_TO_INT = {char: i for i, char in enumerate(DNA_ALPHABET)}

    original_input = one_hot_encode(sequence)
    base_saliency = calculate_saliency_map(model, original_input.clone(), device=device)

    dependency_map = np.zeros((seq_len, seq_len))

    for i in range(seq_len):
        if (i + 1) % 50 == 0:
            print(f"  Processing mutation at position {i+1}/{seq_len}...")
            
        # original_base_idx = DNA_TO_INT[sequence[i]]
        current_char = sequence[i]
        
        # Store the changes for all possible mutations at this position
        saliency_changes = []
        
        # Step 3: Iterate through all possible alternate bases at position `i`
        for mut_type in ['A', 'T', 'C', 'G', 'N']:
            if mut_type == current_char:
                continue
            
            mutated_input = original_input.clone()
            
            if mut_type == 'N':
                # Representing deletion as a zero vector
                mutated_input[0, :, i] = 0 
            else:
                # Standard mutation to one of the 4 bases
                mut_base_idx = DNA_TO_INT[mut_type]
                mutated_input[0, :, i] = 0
                mutated_input[0, mut_base_idx, i] = 1
            
            # Step 4 & 5: Calculate difference
            mutated_saliency = calculate_saliency_map(model, mutated_input, device)
            saliency_diff = mutated_saliency - base_saliency
            saliency_changes.append(saliency_diff)
            
        # Step 6: Aggregate the effects of all mutations at position `i`
        # We take the mean of the absolute differences. You could also use max or other metrics.
        # print(len(saliency_changes))
        if saliency_changes:
            # print(np.array(saliency_changes).shape)
            avg_abs_change = np.mean([np.abs(c) for c in saliency_changes], axis=0)
            # print(avg_abs_change.shape)
            dependency_map[:, i] = avg_abs_change

    print("Dependency map generation complete.")
    return dependency_map