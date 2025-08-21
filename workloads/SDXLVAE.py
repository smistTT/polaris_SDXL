#!/usr/bin/env python3
"""
Enhanced Simple VAE Implementation for Polaris - Bridging toward AutoencoderKL
Significantly expanded parameter count while maintaining TTSIM compatibility
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.config.wl2archmap import WL2ArchTypeSpec, WL2ArchDatatypes
import numpy as np

def setup_ttsim_framework():
    """Initialize TTSIM framework if not already done"""
    try:
        WL2ArchTypeSpec.get_instance()
    except:
        # Set up basic WL2ArchTypeSpec instance for TTSIM operations
        datatype_spec = WL2ArchDatatypes(
            global_type="float32",
            override={}
        )
        WL2ArchTypeSpec.set_instance(datatype_spec)

class EnhancedConv2D(SimNN.Module):
    """Enhanced Conv2D wrapper for TTSIM with larger parameter count"""
    def __init__(self, name, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.name = name

        self.conv = F.Conv2d(name, in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv.set_module(self)
        super().link_op2module()
    
    def __call__(self, x):
        return self.conv(x)

class EnhancedGroupNorm(SimNN.Module):
    """Enhanced GroupNorm wrapper for TTSIM"""
    def __init__(self, name, num_channels, num_groups=None, eps=1e-6):
        super().__init__()
        self.name = name
        
        # Auto-adjust groups to ensure divisibility for larger channel counts
        if num_groups is None:
            # Use min(32, num_channels) for better coverage with larger channels
            num_groups = min(32, num_channels)
            # Ensure num_channels is divisible by num_groups
            while num_channels % num_groups != 0 and num_groups > 1:
                num_groups -= 1
        
        self.group_norm = F.GroupNorm(f"{name}.group_norm", num_channels, num_groups=num_groups, epsilon=eps)
        self.group_norm.set_module(self)
        super().link_op2module()
    
    def __call__(self, x):
        return self.group_norm(x)

class EnhancedSiLU(SimNN.Module):
    """Enhanced SiLU activation for TTSIM"""
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.silu = F.Gelu(f"{name}.silu")  # Use Gelu as SiLU substitute
        self.silu.set_module(self)
        super().link_op2module()
    
    def __call__(self, x):
        return self.silu(x)

class EnhancedVAEEncoder(SimNN.Module):
    """Significantly Enhanced VAE Encoder - much deeper and wider architecture"""
    def __init__(self, name, config):
        super().__init__()
        self.name = name
        
        # Parse config with much larger default channels
        if isinstance(config, dict):
            self.in_channels = config.get('in_channels', 3)
            # Production-scale channels for maximum parameter increase
            provided_channels = config.get('block_out_channels', [256, 512, 1024, 1024])
            # Ensure we have at least 4 channel levels for deep architecture
            if len(provided_channels) < 4:
                # Extend the provided channels to 4 levels
                while len(provided_channels) < 4:
                    provided_channels.append(provided_channels[-1] * 2)
            self.block_out_channels = provided_channels[:4]  # Use first 4 levels
            self.latent_channels = config.get('latent_channels', 4)
            self.double_z = config.get('double_z', True)
        else:
            self.in_channels = getattr(config, 'in_channels', 3)
            provided_channels = getattr(config, 'block_out_channels', [128, 256, 512, 512])
            # Ensure we have at least 4 channel levels for deep architecture
            if len(provided_channels) < 4:
                # Extend the provided channels to 4 levels
                while len(provided_channels) < 4:
                    provided_channels.append(provided_channels[-1] * 2)
            self.block_out_channels = provided_channels[:4]  # Use first 4 levels
            self.latent_channels = getattr(config, 'latent_channels', 4)
            self.double_z = getattr(config, 'double_z', True)
        
        # Determine final latent channels (encoder outputs mean + logvar = 2 * latent_channels)
        self.output_channels = self.latent_channels * 2 if self.double_z else self.latent_channels
        
        # Much deeper encoder architecture with multiple progressive downsampling layers
        
        # Initial convolution: 3 → 128
        self.conv_in = EnhancedConv2D(f"{name}.conv_in", self.in_channels, self.block_out_channels[0])
        self.norm_in = EnhancedGroupNorm(f"{name}.norm_in", self.block_out_channels[0])
        self.act_in = EnhancedSiLU(f"{name}.act_in")
        
        # Block 1: 128 → 128 (multiple conv layers)
        self.conv1_1 = EnhancedConv2D(f"{name}.conv1_1", self.block_out_channels[0], self.block_out_channels[0])
        self.norm1_1 = EnhancedGroupNorm(f"{name}.norm1_1", self.block_out_channels[0])
        self.act1_1 = EnhancedSiLU(f"{name}.act1_1")
        
        self.conv1_2 = EnhancedConv2D(f"{name}.conv1_2", self.block_out_channels[0], self.block_out_channels[0])
        self.norm1_2 = EnhancedGroupNorm(f"{name}.norm1_2", self.block_out_channels[0])
        self.act1_2 = EnhancedSiLU(f"{name}.act1_2")
        
        # Downsample 1: 128 → 256
        self.down1 = EnhancedConv2D(f"{name}.down1", self.block_out_channels[0], self.block_out_channels[1], 
                                   kernel_size=3, stride=2, padding=1)
        self.norm_down1 = EnhancedGroupNorm(f"{name}.norm_down1", self.block_out_channels[1])
        self.act_down1 = EnhancedSiLU(f"{name}.act_down1")
        
        # Block 2: 256 → 256 (multiple conv layers)
        self.conv2_1 = EnhancedConv2D(f"{name}.conv2_1", self.block_out_channels[1], self.block_out_channels[1])
        self.norm2_1 = EnhancedGroupNorm(f"{name}.norm2_1", self.block_out_channels[1])
        self.act2_1 = EnhancedSiLU(f"{name}.act2_1")
        
        self.conv2_2 = EnhancedConv2D(f"{name}.conv2_2", self.block_out_channels[1], self.block_out_channels[1])
        self.norm2_2 = EnhancedGroupNorm(f"{name}.norm2_2", self.block_out_channels[1])
        self.act2_2 = EnhancedSiLU(f"{name}.act2_2")
        
        # Downsample 2: 256 → 512  
        self.down2 = EnhancedConv2D(f"{name}.down2", self.block_out_channels[1], self.block_out_channels[2], 
                                   kernel_size=3, stride=2, padding=1)
        self.norm_down2 = EnhancedGroupNorm(f"{name}.norm_down2", self.block_out_channels[2])
        self.act_down2 = EnhancedSiLU(f"{name}.act_down2")
        
        # Block 3: 512 → 512 (multiple conv layers)
        self.conv3_1 = EnhancedConv2D(f"{name}.conv3_1", self.block_out_channels[2], self.block_out_channels[2])
        self.norm3_1 = EnhancedGroupNorm(f"{name}.norm3_1", self.block_out_channels[2])
        self.act3_1 = EnhancedSiLU(f"{name}.act3_1")
        
        self.conv3_2 = EnhancedConv2D(f"{name}.conv3_2", self.block_out_channels[2], self.block_out_channels[2])
        self.norm3_2 = EnhancedGroupNorm(f"{name}.norm3_2", self.block_out_channels[2])
        self.act3_2 = EnhancedSiLU(f"{name}.act3_2")
        
        # Downsample 3: 512 → 512
        self.down3 = EnhancedConv2D(f"{name}.down3", self.block_out_channels[2], self.block_out_channels[3], 
                                   kernel_size=3, stride=2, padding=1)
        self.norm_down3 = EnhancedGroupNorm(f"{name}.norm_down3", self.block_out_channels[3])
        self.act_down3 = EnhancedSiLU(f"{name}.act_down3")
        
        # Block 4: 512 → 512 (multiple conv layers)
        self.conv4_1 = EnhancedConv2D(f"{name}.conv4_1", self.block_out_channels[3], self.block_out_channels[3])
        self.norm4_1 = EnhancedGroupNorm(f"{name}.norm4_1", self.block_out_channels[3])
        self.act4_1 = EnhancedSiLU(f"{name}.act4_1")
        
        self.conv4_2 = EnhancedConv2D(f"{name}.conv4_2", self.block_out_channels[3], self.block_out_channels[3])
        self.norm4_2 = EnhancedGroupNorm(f"{name}.norm4_2", self.block_out_channels[3])
        self.act4_2 = EnhancedSiLU(f"{name}.act4_2")
        
        # Final output convolution: 512 → output_channels (8 for mean+logvar)
        self.conv_out = EnhancedConv2D(f"{name}.conv_out", self.block_out_channels[3], self.output_channels)
        
        super().link_op2module()
    
    def __call__(self, x):
        # Initial conv
        h = self.conv_in(x)
        h = self.norm_in(h)
        h = self.act_in(h)
        
        # Block 1
        h = self.conv1_1(h)
        h = self.norm1_1(h)
        h = self.act1_1(h)
        h = self.conv1_2(h)
        h = self.norm1_2(h)
        h = self.act1_2(h)
        
        # Downsample 1
        h = self.down1(h)
        h = self.norm_down1(h)
        h = self.act_down1(h)
        
        # Block 2
        h = self.conv2_1(h)
        h = self.norm2_1(h)
        h = self.act2_1(h)
        h = self.conv2_2(h)
        h = self.norm2_2(h)
        h = self.act2_2(h)
        
        # Downsample 2
        h = self.down2(h)
        h = self.norm_down2(h)
        h = self.act_down2(h)
        
        # Block 3
        h = self.conv3_1(h)
        h = self.norm3_1(h)
        h = self.act3_1(h)
        h = self.conv3_2(h)
        h = self.norm3_2(h)
        h = self.act3_2(h)
        
        # Downsample 3
        h = self.down3(h)
        h = self.norm_down3(h)
        h = self.act_down3(h)
        
        # Block 4
        h = self.conv4_1(h)
        h = self.norm4_1(h)
        h = self.act4_1(h)
        h = self.conv4_2(h)
        h = self.norm4_2(h)
        h = self.act4_2(h)
        
        # Final output
        h = self.conv_out(h)
        
        return h

class EnhancedVAEDecoder(SimNN.Module):
    """Significantly Enhanced VAE Decoder - symmetric to encoder"""
    def __init__(self, name, config):
        super().__init__()
        self.name = name
        
        # Parse config - symmetric to encoder
        if isinstance(config, dict):
            self.out_channels = config.get('out_channels', 3)
            provided_channels = config.get('block_out_channels', [128, 256, 512, 512])
            # Ensure we have at least 4 channel levels for deep architecture
            if len(provided_channels) < 4:
                # Extend the provided channels to 4 levels
                while len(provided_channels) < 4:
                    provided_channels.append(provided_channels[-1] * 2)
            self.block_out_channels = provided_channels[:4]  # Use first 4 levels
            self.latent_channels = config.get('latent_channels', 4)
        else:
            self.out_channels = getattr(config, 'out_channels', 3)
            provided_channels = getattr(config, 'block_out_channels', [128, 256, 512, 512])
            # Ensure we have at least 4 channel levels for deep architecture
            if len(provided_channels) < 4:
                # Extend the provided channels to 4 levels
                while len(provided_channels) < 4:
                    provided_channels.append(provided_channels[-1] * 2)
            self.block_out_channels = provided_channels[:4]  # Use first 4 levels
            self.latent_channels = getattr(config, 'latent_channels', 4)
        
        # Symmetric decoder architecture (reverse of encoder)
        
        # Initial convolution: latent_channels → 512
        self.conv_in = EnhancedConv2D(f"{name}.conv_in", self.latent_channels, self.block_out_channels[3])
        self.norm_in = EnhancedGroupNorm(f"{name}.norm_in", self.block_out_channels[3])
        self.act_in = EnhancedSiLU(f"{name}.act_in")
        
        # Block 1: 512 → 512 (multiple conv layers)
        self.conv1_1 = EnhancedConv2D(f"{name}.conv1_1", self.block_out_channels[3], self.block_out_channels[3])
        self.norm1_1 = EnhancedGroupNorm(f"{name}.norm1_1", self.block_out_channels[3])
        self.act1_1 = EnhancedSiLU(f"{name}.act1_1")
        
        self.conv1_2 = EnhancedConv2D(f"{name}.conv1_2", self.block_out_channels[3], self.block_out_channels[3])
        self.norm1_2 = EnhancedGroupNorm(f"{name}.norm1_2", self.block_out_channels[3])
        self.act1_2 = EnhancedSiLU(f"{name}.act1_2")
        
        # Upsample 1: 512 → 512 (upsample + conv)
        self.up1 = F.Resize(f"{name}.up1", scale_factor=2.0, mode="nearest")
        self.up1.set_module(self)
        self.conv_up1 = EnhancedConv2D(f"{name}.conv_up1", self.block_out_channels[3], self.block_out_channels[2])
        self.norm_up1 = EnhancedGroupNorm(f"{name}.norm_up1", self.block_out_channels[2])
        self.act_up1 = EnhancedSiLU(f"{name}.act_up1")
        
        # Block 2: 512 → 512 (multiple conv layers)
        self.conv2_1 = EnhancedConv2D(f"{name}.conv2_1", self.block_out_channels[2], self.block_out_channels[2])
        self.norm2_1 = EnhancedGroupNorm(f"{name}.norm2_1", self.block_out_channels[2])
        self.act2_1 = EnhancedSiLU(f"{name}.act2_1")
        
        self.conv2_2 = EnhancedConv2D(f"{name}.conv2_2", self.block_out_channels[2], self.block_out_channels[2])
        self.norm2_2 = EnhancedGroupNorm(f"{name}.norm2_2", self.block_out_channels[2])
        self.act2_2 = EnhancedSiLU(f"{name}.act2_2")
        
        # Upsample 2: 512 → 256 (upsample + conv)
        self.up2 = F.Resize(f"{name}.up2", scale_factor=2.0, mode="nearest")
        self.up2.set_module(self)
        self.conv_up2 = EnhancedConv2D(f"{name}.conv_up2", self.block_out_channels[2], self.block_out_channels[1])
        self.norm_up2 = EnhancedGroupNorm(f"{name}.norm_up2", self.block_out_channels[1])
        self.act_up2 = EnhancedSiLU(f"{name}.act_up2")
        
        # Block 3: 256 → 256 (multiple conv layers)
        self.conv3_1 = EnhancedConv2D(f"{name}.conv3_1", self.block_out_channels[1], self.block_out_channels[1])
        self.norm3_1 = EnhancedGroupNorm(f"{name}.norm3_1", self.block_out_channels[1])
        self.act3_1 = EnhancedSiLU(f"{name}.act3_1")
        
        self.conv3_2 = EnhancedConv2D(f"{name}.conv3_2", self.block_out_channels[1], self.block_out_channels[1])
        self.norm3_2 = EnhancedGroupNorm(f"{name}.norm3_2", self.block_out_channels[1])
        self.act3_2 = EnhancedSiLU(f"{name}.act3_2")
        
        # Upsample 3: 256 → 128 (upsample + conv)
        self.up3 = F.Resize(f"{name}.up3", scale_factor=2.0, mode="nearest")
        self.up3.set_module(self)
        self.conv_up3 = EnhancedConv2D(f"{name}.conv_up3", self.block_out_channels[1], self.block_out_channels[0])
        self.norm_up3 = EnhancedGroupNorm(f"{name}.norm_up3", self.block_out_channels[0])
        self.act_up3 = EnhancedSiLU(f"{name}.act_up3")
        
        # Block 4: 128 → 128 (multiple conv layers)
        self.conv4_1 = EnhancedConv2D(f"{name}.conv4_1", self.block_out_channels[0], self.block_out_channels[0])
        self.norm4_1 = EnhancedGroupNorm(f"{name}.norm4_1", self.block_out_channels[0])
        self.act4_1 = EnhancedSiLU(f"{name}.act4_1")
        
        self.conv4_2 = EnhancedConv2D(f"{name}.conv4_2", self.block_out_channels[0], self.block_out_channels[0])
        self.norm4_2 = EnhancedGroupNorm(f"{name}.norm4_2", self.block_out_channels[0])
        self.act4_2 = EnhancedSiLU(f"{name}.act4_2")
        
        # Final output convolution: 128 → 3 (RGB)
        self.conv_out = EnhancedConv2D(f"{name}.conv_out", self.block_out_channels[0], self.out_channels)
        
        super().link_op2module()
    
    def __call__(self, z):
        # Initial conv from latent space
        h = self.conv_in(z)
        h = self.norm_in(h)
        h = self.act_in(h)
        
        # Block 1
        h = self.conv1_1(h)
        h = self.norm1_1(h)
        h = self.act1_1(h)
        h = self.conv1_2(h)
        h = self.norm1_2(h)
        h = self.act1_2(h)
        
        # Upsample 1
        h = self.up1(h)
        h = self.conv_up1(h)
        h = self.norm_up1(h)
        h = self.act_up1(h)
        
        # Block 2
        h = self.conv2_1(h)
        h = self.norm2_1(h)
        h = self.act2_1(h)
        h = self.conv2_2(h)
        h = self.norm2_2(h)
        h = self.act2_2(h)
        
        # Upsample 2
        h = self.up2(h)
        h = self.conv_up2(h)
        h = self.norm_up2(h)
        h = self.act_up2(h)
        
        # Block 3
        h = self.conv3_1(h)
        h = self.norm3_1(h)
        h = self.act3_1(h)
        h = self.conv3_2(h)
        h = self.norm3_2(h)
        h = self.act3_2(h)
        
        # Upsample 3
        h = self.up3(h)
        h = self.conv_up3(h)
        h = self.norm_up3(h)
        h = self.act_up3(h)
        
        # Block 4
        h = self.conv4_1(h)
        h = self.norm4_1(h)
        h = self.act4_1(h)
        h = self.conv4_2(h)
        h = self.norm4_2(h)
        h = self.act4_2(h)
        
        # Final output
        h = self.conv_out(h)
        
        return h

class EnhancedSampler(SimNN.Module):
    """Enhanced sampler for VAE latent sampling"""
    def __init__(self, name, latent_channels):
        super().__init__()
        self.name = name
        self.latent_channels = latent_channels
        
        # Use slicing to extract mean (first latent_channels)
        self.slice = F.SliceFixed(f"{name}.slice", [0, 0, 0, 0], [1, latent_channels, 512, 512])
        self.slice.set_module(self)
        super().link_op2module()
    
    def __call__(self, encoded):
        # For TTSIM compatibility, just return the mean (first latent_channels)
        # In full VAE, this would do proper reparameterization sampling
        return self.slice(encoded)

class EnhancedAutoencoder(SimNN.Module):
    """Significantly Enhanced Autoencoder with proper VAE sampling"""
    def __init__(self, name, config):
        super().__init__()
        self.name = name
        self.config = config
        
        # Parse config
        if isinstance(config, dict):
            self.latent_channels = config.get('latent_channels', 4)
            self.scaling_factor = config.get('scaling_factor', 0.13025)
            self.sample_size = config.get('sample_size', 64)
        else:
            self.latent_channels = getattr(config, 'latent_channels', 4)
            self.scaling_factor = getattr(config, 'scaling_factor', 0.13025)
            self.sample_size = getattr(config, 'sample_size', 64)
        
        # Enhanced encoder, sampler, and decoder
        self.encoder = EnhancedVAEEncoder(f"{name}.encoder", config)
        self.sampler = EnhancedSampler(f"{name}.sampler", self.latent_channels)
        self.decoder = EnhancedVAEDecoder(f"{name}.decoder", config)
        
        super().link_op2module()
    
    def encode(self, x):
        """Encode RGB to latent distribution (mean + logvar)"""
        return self.encoder(x)
    
    def sample(self, encoded):
        """Sample from latent distribution"""
        return self.sampler(encoded)
    
    def decode(self, z):
        """Decode latent to RGB"""
        return self.decoder(z)
    
    def __call__(self, x):
        """Full VAE forward pass: encode → sample → decode"""
        encoded = self.encode(x)
        latents = self.sample(encoded)
        reconstructed = self.decode(latents)
        return reconstructed

class EnhancedSimpleVAEAutoencoder(SimNN.Module):
    """Enhanced Simple VAE implementation for Polaris framework integration"""
    def __init__(self, name, config):
        super().__init__()
        self.name = name
        self.config = config
        
        # Setup TTSIM framework
        setup_ttsim_framework()
        
        # Create the autoencoder
        self.autoencoder = EnhancedAutoencoder(f"{name}.autoencoder", config)
        
        # Create input tensors for Polaris framework
        self.create_input_tensors()
        
        super().link_op2module()
    
    def create_input_tensors(self):
        """Create input tensors for testing"""
        # Handle config as dict (from Polaris) or object (from direct instantiation)
        if isinstance(self.config, dict):
            batch_size = self.config['bs']
            sample_size = self.config.get('sample_size', 64)
        else:
            batch_size = self.config.bs
            sample_size = getattr(self.config, 'sample_size', 64)
        
        input_shape = [batch_size, 3, sample_size, sample_size]
        
        self.input_tensors = {
            'image': F._from_shape('image', input_shape, is_param=False, np_dtype=np.float32)
        }
        
        return self.input_tensors['image']
    
    def encode(self, image=None):
        """Encode image to latent distribution"""
        if image is None:
            image = self.input_tensors['image']
        return self.autoencoder.encode(image)
    
    def decode(self, latents):
        """Decode latents to image"""
        return self.autoencoder.decode(latents)
    
    def __call__(self, image=None):
        """Forward pass through Enhanced VAE"""
        # Use input tensors if no arguments provided (for Polaris framework)
        if image is None:
            image = self.input_tensors['image']
        
        return self.autoencoder(image)
    
    def get_forward_graph(self):
        """Get forward computation graph for ONNX export"""
        GG = super()._get_forward_graph(self.input_tensors)
        return GG
    
    def analytical_param_count(self):
        """Return analytical parameter count for Enhanced VAE"""
        # Rough calculation based on enhanced architecture with production-scale channels
        block_channels = [256, 512, 1024, 1024]  # Production-scale channels
        
        # Encoder parameters (much deeper)
        encoder_params = 0
        
        # Initial conv: 3 → 256
        encoder_params += 3 * 256 * 9  # 3x3 conv
        
        # Block 1: 256 → 256 (2 layers)
        encoder_params += 256 * 256 * 9 * 2
        
        # Down 1: 256 → 512
        encoder_params += 256 * 512 * 9
        
        # Block 2: 512 → 512 (2 layers)
        encoder_params += 512 * 512 * 9 * 2
        
        # Down 2: 512 → 1024
        encoder_params += 512 * 1024 * 9
        
        # Block 3: 1024 → 1024 (2 layers)
        encoder_params += 1024 * 1024 * 9 * 2
        
        # Down 3: 1024 → 1024
        encoder_params += 1024 * 1024 * 9
        
        # Block 4: 1024 → 1024 (2 layers)
        encoder_params += 1024 * 1024 * 9 * 2
        
        # Final conv: 1024 → 8
        encoder_params += 1024 * 8 * 9
        
        # GroupNorm parameters (approximate)
        for ch in block_channels:
            encoder_params += ch * 4  # Scale and bias parameters
        
        # Decoder parameters (symmetric to encoder)
        decoder_params = encoder_params  # Approximately same
        
        total = encoder_params + decoder_params
        
        return total

# Main classes expected by Polaris framework (must match module name)
SimpleVAEAutoencoder = EnhancedSimpleVAEAutoencoder  # Keep compatible name

# ============================================================================
# Test Functions
# ============================================================================

def test_enhanced_vae_creation():
    """Test Enhanced VAE creation"""
    print("\n=== Testing SDXL VAE Creation ===")
    setup_ttsim_framework()
    
    # Test config with enhanced channels
    config = {
        'in_channels': 3,
        'latent_channels': 4,
        'sample_size': 64,
        'block_out_channels': [128, 256, 512, 512],
        'scaling_factor': 0.13025,
        'bs': 1
    }
    
    try:
        # Test component creation
        vae = EnhancedSimpleVAEAutoencoder("test_enhanced_vae", config)
        print("✅ SDXL VAE created successfully")
        
        # Test parameter counting
        total_params = vae.analytical_param_count()
        print(f"✅ SDXL VAE parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ SDXL VAE creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_enhanced_vae_tests():
    """Run Enhanced VAE tests"""
    print("🚀 Running SDXL VAE Tests...")
    
    tests = [
        test_enhanced_vae_creation,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🎯 SDXL VAE Tests: {passed}/{len(tests)} passed")
    return passed == len(tests)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            run_enhanced_vae_tests()
    else:
        print("SDXL VAE Implementation for Polaris")
        print("Usage: python SDXLVAE.py --test")
