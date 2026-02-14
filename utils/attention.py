"""
Attention processing utilities for MedGemma cross-attention visualization.
"""
import numpy as np
import torch
from PIL import Image
import io
import base64
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def extract_cross_attention(
    attentions: Tuple,
    token_idx: int = -1,
    layer_idx: int = -1,
    aggregate_heads: bool = True
) -> torch.Tensor:
    """
    Extract attention weights for a specific generated token.
    
    Args:
        attentions: Tuple of attention tensors (one per generated token)
        token_idx: Which generated token to extract (-1 = last)
        layer_idx: Which layer to extract from (-1 = last layer)
        aggregate_heads: Whether to average across attention heads
        
    Returns:
        Attention weights tensor (seq_len,) or (num_heads, seq_len)
    """
    if not attentions or len(attentions) == 0:
        raise ValueError("No attention data available")
    
    # Get attention for specified token
    token_attentions = attentions[token_idx]  # Tuple of layers for this token
    
    # Get specified layer
    layer_attention = token_attentions[layer_idx]  # Shape: (batch, num_heads, seq_len, seq_len)
    
    # Get the last row (attention FROM the newly generated token TO all previous tokens)
    # This shows what the model was looking at when generating this token
    last_row_attention = layer_attention[0, :, -1, :]  # Shape: (num_heads, seq_len)
    
    if aggregate_heads:
        # Average across heads: (num_heads, seq_len) -> (seq_len,)
        attention = last_row_attention.mean(dim=0)
    else:
        # Keep heads: (num_heads, seq_len)
        attention = last_row_attention
    
    return attention


def attention_to_spatial_grid(
    attention_weights: torch.Tensor,
    grid_size: int = 16
) -> np.ndarray:
    """
    Convert attention weights to spatial grid.
    For MedGemma, the early sequence tokens are image patch embeddings.
    
    Args:
        attention_weights: Attention weights (seq_len,)
        grid_size: Size of spatial grid (default 16x16)
        
    Returns:
        2D spatial attention map (grid_size, grid_size)
    """
    # Convert to numpy (handle BFloat16)
    if isinstance(attention_weights, torch.Tensor):
        # Convert BFloat16 -> Float32 -> numpy
        attention_weights = attention_weights.float().cpu().numpy()
    
    # Extract just the image tokens (first N tokens, where N = grid_size^2)
    num_image_tokens = grid_size * grid_size
    
    if len(attention_weights) < num_image_tokens:
        logger.warning(f"Sequence length {len(attention_weights)} < expected image tokens {num_image_tokens}")
        # Pad with zeros
        padding = num_image_tokens - len(attention_weights)
        attention_weights = np.pad(attention_weights, (0, padding), mode='constant')
    
    # Take first N tokens as image tokens
    image_attention = attention_weights[:num_image_tokens]
    
    # Reshape to 2D grid
    spatial_grid = image_attention.reshape(grid_size, grid_size)
    
    return spatial_grid


def upsample_attention_map(
    attention_grid: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: str = 'bicubic'
) -> np.ndarray:
    """
    Upsample attention map to match image dimensions.
    
    Args:
        attention_grid: NxN attention grid
        target_size: Target image size (height, width)
        interpolation: Interpolation method
        
    Returns:
        Upsampled attention map
    """
    from scipy.ndimage import zoom
    
    h, w = target_size
    grid_h, grid_w = attention_grid.shape
    
    # Calculate zoom factors
    zoom_h = h / grid_h
    zoom_w = w / grid_w
    
    # Upsample using scipy zoom
    if interpolation == 'bicubic':
        order = 3
    elif interpolation == 'bilinear':
        order = 1
    else:
        order = 0  # nearest
    
    upsampled = zoom(attention_grid, (zoom_h, zoom_w), order=order)
    
    return upsampled


def generate_heatmap_image(
    attention_map: np.ndarray,
    colormap: str = 'jet',
    normalize: bool = True
) -> str:
    """
    Generate heatmap image from attention map.
    
    Args:
        attention_map: 2D attention map
        colormap: Matplotlib colormap name
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Base64 encoded PNG image
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Normalize
    if normalize:
        min_val = attention_map.min()
        max_val = attention_map.max()
        if max_val > min_val:
            attention_map = (attention_map - min_val) / (max_val - min_val)
        else:
            attention_map = np.zeros_like(attention_map)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(attention_map)
    
    # Convert to PIL Image (0-255)
    img_array = (colored[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return img_base64


def process_input_token_attentions(
    attentions: Tuple,
    input_text: str,
    image_size: Tuple[int, int],
    num_image_tokens: int = 256,
    layer_idx: int = -1
) -> List[dict]:
    """
    Process overall attention pattern.
    Shows where the model looked during generation (averaged across multiple steps).
    
    Args:
        attentions: Model attention outputs (tuple of tuples)
        input_text: The user's input question/prompt (unused, kept for API compatibility)
        image_size: Original image size (height, width)
        num_image_tokens: Number of image tokens (for grid size calculation)
        layer_idx: Which layer to use
        
    Returns:
        List with single {token, heatmap} dict for overall attention
    """
    results = []
    
    # Calculate grid size from number of image tokens
    grid_size = int(np.sqrt(num_image_tokens))
    
    logger.info(f"Processing overall attention pattern, attentions length: {len(attentions)}")
    
    # Average attention across several generation steps for stable overall pattern
    num_steps = min(10, len(attentions))
    attention_sum = None
    count = 0
    
    try:
        for step_idx in range(num_steps):
            if step_idx >= len(attentions):
                break
            
            # Get attention for this generation step
            step_attention = extract_cross_attention(
                attentions,
                token_idx=step_idx,
                layer_idx=layer_idx,
                aggregate_heads=True
            )
            
            # Only take image tokens (first grid_size^2 elements)
            num_img_tokens = grid_size * grid_size
            if len(step_attention) >= num_img_tokens:
                img_attention = step_attention[:num_img_tokens]
                
                if attention_sum is None:
                    attention_sum = img_attention.clone()
                else:
                    attention_sum = attention_sum + img_attention
                count += 1
        
        if attention_sum is None or count == 0:
            logger.warning("No valid attention data to process")
            return []
        
        # Average the accumulated attention
        attention_avg = attention_sum / count
        
        logger.info(f"Overall attention shape: {attention_avg.shape}")
        
        # Convert to spatial grid
        spatial_grid = attention_to_spatial_grid(attention_avg, grid_size=grid_size)
        
        # Upsample to image size
        upsampled = upsample_attention_map(spatial_grid, image_size)
        
        # Generate heatmap image
        heatmap_base64 = generate_heatmap_image(upsampled, colormap='hot')
        
        results.append({
            'token': 'Overall Attention',
            'heatmap': heatmap_base64
        })
        
        logger.info(f"Successfully generated overall attention heatmap")
    except Exception as e:
        logger.error(f"Failed to process overall attention: {e}", exc_info=True)
    
    return results
