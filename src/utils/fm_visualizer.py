from pathlib import Path
import torch
from captum.attr import LayerGradCam
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np

class FeatureMapVisualizer:
    """Class for extracting and visualizing feature maps from each layer of a ResNet model."""
    
    def __init__(self, model, writer, visualization_dir):
        """Initialize the feature map visualizer.
        
        Args:
            model: The ResNet model to visualize
            writer: TensorBoard SummaryWriter instance
            visualization_dir: Directory to save visualizations
        """
        self.model = model
        self.writer = writer
        self.visualization_dir = Path(visualization_dir)
        self.feature_maps = {}
        self.hook_handles = []
        
        # Ensure directory exists
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        # Register hooks for all layers to capture feature maps
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks only to major layer outputs to capture feature maps."""
        # Clear previous hooks if any
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
        # Hook function to save feature maps
        def hook_fn(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()
            return hook
        
        # Register hooks only to major layers outputs
        # Initial processing (after the first convolution block)
        self.hook_handles.append(self.model.maxpool.register_forward_hook(hook_fn('initial_block')))
        
        # Main residual layers - only register hooks at the layer level (not individual blocks)
        self.hook_handles.append(self.model.layer1.register_forward_hook(hook_fn('layer1')))
        self.hook_handles.append(self.model.layer2.register_forward_hook(hook_fn('layer2')))
        self.hook_handles.append(self.model.layer3.register_forward_hook(hook_fn('layer3')))
        self.hook_handles.append(self.model.layer4.register_forward_hook(hook_fn('layer4')))
    
    def _get_gradcam_for_layer(self, img, label, layer_module):
        """Get GradCAM for a specific layer.
        
        Args:
            img: Input image tensor
            label: Target label
            layer_module: Layer module to compute GradCAM for
            
        Returns:
            Processed heatmap tensor
        """
        cam = LayerGradCam(self.model, layer_module)
        attr = cam.attribute(img.unsqueeze(0), target=label)
        heat = attr.squeeze(0).mean(0)
        if heat.shape != (224, 224):  # Resize if needed
            heat = TF.resize(heat.unsqueeze(0), [224, 224]).squeeze(0)
        return heat
    
    def _visualize_feature_map(self, feature_map, layer_name, max_features=8):
        """Visualize a feature map.
        
        Args:
            feature_map: Feature map tensor [C, H, W]
            layer_name: Name of the layer
            max_features: Maximum number of feature channels to visualize
            
        Returns:
            Matplotlib figure with feature map visualization
        """
        # Get the first batch and select subset of channels if there are too many
        if len(feature_map.shape) == 4:  # [B, C, H, W]
            feature_map = feature_map[0]  # Take first batch [C, H, W]
        
        # For layer4 and later stages where we might have too many features
        if feature_map.shape[0] > max_features:
            # Choose evenly spaced features
            step = feature_map.shape[0] // max_features
            indices = list(range(0, feature_map.shape[0], step))[:max_features]
            feature_map = feature_map[indices]
            
        num_features = feature_map.shape[0]
        
        # Calculate grid dimensions - try to keep it close to square
        grid_width = int(np.ceil(np.sqrt(num_features)))
        grid_height = int(np.ceil(num_features / grid_width))
        
        # Create figure with black background
        fig, axes = plt.subplots(grid_height, grid_width, figsize=(16, 16), 
                                 facecolor='black', dpi=150)
        fig.suptitle(f'Feature Maps - {layer_name}', fontsize=18, color='white')
        
        # Flatten axes for easy indexing
        if hasattr(axes, 'flatten'):
            axes = axes.flatten()
        elif not isinstance(axes, np.ndarray):  # If there's only one subplot
            axes = np.array([axes])
        
        # Plot each feature channel
        for i in range(grid_width * grid_height):
            if i < num_features:
                # Get feature map and normalize for visualization
                fm = feature_map[i].cpu().numpy()
                if fm.size == 0:
                    continue
                
                # Normalize for better visualization
                if fm.max() > fm.min():
                    fm = (fm - fm.min()) / (fm.max() - fm.min())
                
                # Use a more visually distinctive colormap
                if layer_name == 'layer1':
                    cmap = 'viridis' 
                elif layer_name == 'layer2':
                    cmap = 'plasma'
                elif layer_name == 'layer3':
                    cmap = 'inferno'
                elif layer_name == 'layer4':
                    cmap = 'magma'
                else:
                    cmap = 'cividis'
                    
                ax = axes[i]
                ax.imshow(fm, cmap=cmap)
                ax.set_title(f'{i}', fontsize=8, color='white')
                ax.set_facecolor('black')
                
                # Add a thin white border around each feature map
                for spine in ax.spines.values():
                    spine.set_color('white')
                    spine.set_linewidth(0.5)
            else:
                # Hide unused subplots
                axes[i].axis('off')
                axes[i].set_facecolor('black')
            
            # Turn off axis for all subplots
            if i < num_features:
                axes[i].set_xticks([])
                axes[i].set_yticks([])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, wspace=0.02, hspace=0.3)  # Tighter spacing
        
        return fig
    
    def visualize_all_layers(self, img, label, epoch, save=True):
        """Run a forward pass and visualize feature maps from all layers.
        
        Args:
            img: Input image tensor [C, H, W]
            label: Target class label
            epoch: Current epoch number (for saving)
            save: Whether to save visualizations to disk
            
        Returns:
            Dictionary mapping layer names to feature map tensors
        """
        # Clear previous feature maps
        self.feature_maps = {}
        
        # Ensure model is in eval mode and run forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(img.unsqueeze(0))
        
        # Create directory for this epoch
        if save:
            epoch_dir = self.visualization_dir / f'epoch_{epoch}'
            epoch_dir.mkdir(exist_ok=True)
        
        # Visualize original image
        img_np = img.cpu().numpy().transpose((1, 2, 0))
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np)
        ax.set_title('Original Image')
        ax.axis('off')
        
        if save:
            fig.savefig(epoch_dir / 'original_image.png', bbox_inches='tight')
        
        # Log to tensorboard
        self.writer.add_image('Original_Image', img.unsqueeze(0), epoch, dataformats='NCHW')
        plt.close(fig)
        
        # Process each feature map for the main layers
        main_layers = ['initial_block', 'layer1', 'layer2', 'layer3', 'layer4']
        for layer_name in main_layers:
            if layer_name not in self.feature_maps:
                continue
                
            feature_map = self.feature_maps[layer_name]
            
            # Skip if feature map is empty or not a proper tensor
            if not isinstance(feature_map, torch.Tensor) or feature_map.numel() == 0:
                continue
            
            # Create visualization - we'll skip fc and avgpool since they don't have spatial dimensions
            if layer_name not in ['fc', 'avgpool']:
                # Create feature map visualization
                fig = self._visualize_feature_map(feature_map, layer_name)
                
                # Save visualization
                if save:
                    # Create a high-quality PNG with black background
                    fig.savefig(epoch_dir / f'{layer_name}_feature_map.png', 
                               bbox_inches='tight', facecolor='black', dpi=200)
                
                # Log to tensorboard
                self.writer.add_figure(f'FeatureMap/{layer_name}', fig, epoch)
                plt.close(fig)
            
            # For main residual layers, also compute GradCAM
            if layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                try:
                    # Get the layer module
                    layer_module = getattr(self.model, layer_name)
                    
                    # For layer modules, we'll use the last block's last convolution
                    last_block = layer_module[-1]  # Get the last block
                    module = last_block.conv3  # Use the last convolution
                    
                    # Get GradCAM
                    heat = self._get_gradcam_for_layer(img, label, module)
                    heat_np = heat.cpu().numpy()
                    
                    # Visualize GradCAM
                    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='black')
                    
                    # Original image
                    axes[0].imshow(img_np)
                    axes[0].set_title('Original Image', color='white')
                    axes[0].axis('off')
                    
                    # GradCAM
                    im = axes[1].imshow(heat_np, cmap='jet')
                    axes[1].set_title(f'GradCAM - {layer_name}', color='white')
                    axes[1].axis('off')
                    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                    
                    if save:
                        fig.savefig(epoch_dir / f'{layer_name}_gradcam.png', 
                                  bbox_inches='tight', facecolor='black')
                    
                    # Overlay
                    fig_overlay, ax = plt.subplots(figsize=(8, 8), facecolor='black')
                    ax.imshow(img_np)
                    ax.imshow(heat_np, cmap='jet', alpha=0.5)
                    ax.set_title(f'GradCAM Overlay - {layer_name}', color='white')
                    ax.axis('off')
                    
                    if save:
                        fig_overlay.savefig(epoch_dir / f'{layer_name}_gradcam_overlay.png', 
                                          bbox_inches='tight', facecolor='black')
                    
                    # Log to tensorboard
                    self.writer.add_figure(f'GradCAM/{layer_name}', fig, epoch)
                    self.writer.add_figure(f'GradCAM_Overlay/{layer_name}', fig_overlay, epoch)
                    
                    plt.close(fig)
                    plt.close(fig_overlay)
                    
                except Exception as e:
                    print(f"Error generating GradCAM for {layer_name}: {e}")
        
        return self.feature_maps
    
    def cleanup(self):
        """Remove all hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []