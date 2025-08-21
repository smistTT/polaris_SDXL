#!/usr/bin/env python3
"""
SDXL Pipeline Integration - Complete Stable Diffusion XL Implementation
Integrates SDXL VAE + DualTextEncoder + SDXLUNet for full SDXL pipeline
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.config.wl2archmap import WL2ArchTypeSpec, WL2ArchDatatypes
import numpy as np

# Import our successful components with direct imports to avoid exec issues
try:
    from SDXLVAE import SimpleVAEAutoencoder  # Use SDXL VAE with production channels
    from DualTextEncoder import DualTextEncoderSDXL  
    from SDXLUNet import SDXLUNet
except ImportError:
    # Fallback: exec-based loading with proper __file__ context
    def load_component_module(module_name):
        """Dynamically load a module from the workloads directory"""
        module_path = os.path.join(os.path.dirname(__file__), f"{module_name}.py")
        if os.path.exists(module_path):
            spec = {'__file__': module_path}  # Fix: Provide __file__ context
            with open(module_path, 'r') as f:
                exec(f.read(), spec)
            return spec
        else:
            raise ImportError(f"Module {module_name} not found at {module_path}")

    # Load component modules
    SDXLVAE_module = load_component_module('SDXLVAE')
    DualTextEncoder_module = load_component_module('DualTextEncoder')
    SDXLUNet_module = load_component_module('SDXLUNet')
    
    # Extract classes from loaded modules
    SimpleVAEAutoencoder = SDXLVAE_module['SimpleVAEAutoencoder']
    DualTextEncoderSDXL = DualTextEncoder_module['DualTextEncoderSDXL']
    SDXLUNet = SDXLUNet_module['SDXLUNet']

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

class SDXLPipelineComplete(SimNN.Module):
    """
    Complete SDXL Pipeline integrating all three components
    Following proven SimpleVAE/SuperSimpleCLIP pattern for TTSIM compatibility
    """
    def __init__(self, name, config):
        super().__init__()
        self.name = name
        self.config = config
        
        # Setup TTSIM framework
        setup_ttsim_framework()
        
        # Parse configuration
        if isinstance(config, dict):
            # VAE configuration (SDXL VAE with production-scale channels)
            self.vae_config = {
                'in_channels': config.get('vae_in_channels', 3),
                'latent_channels': config.get('vae_latent_channels', 4),
                'sample_size': config.get('vae_sample_size', 512),  # Input image size
                'block_out_channels': config.get('vae_block_out_channels', [256, 512, 1024, 1024]),  # Production scale
                'scaling_factor': config.get('vae_scaling_factor', 0.13025),
                'bs': config.get('bs', 1)
            }
            
            # CLIP configuration (Dual Text Encoder)
            self.clip_config = {
                'vocab_size': config.get('clip_vocab_size', 49408),
                'max_seq_length': config.get('clip_max_seq_length', 77),
                # Text encoder 1 (CLIPTextModel - 768 dim)
                'hidden_size_1': config.get('clip_hidden_size', 768),  
                'intermediate_size_1': config.get('clip_intermediate_size', 3072),
                'num_attention_heads_1': config.get('clip_num_attention_heads', 12),
                # Text encoder 2 (CLIPTextModelWithProjection - 1280 dim)
                'hidden_size_2': config.get('clip_hidden_size_2', 1280),
                'intermediate_size_2': config.get('clip_intermediate_size_2', 5120),
                'num_attention_heads_2': config.get('clip_num_attention_heads_2', 20),
                'projection_dim': config.get('clip_projection_dim', 1280),
                'num_hidden_layers': config.get('clip_num_hidden_layers', 6),
                'max_position_embeddings': config.get('clip_max_position_embeddings', 77),
                'bs': config.get('bs', 1)
            }
            
            # UNet configuration (full SDXL spec)
            self.unet_config = {
                'sample_size': config.get('unet_sample_size', 128),  # Latent space size
                'in_channels': config.get('unet_in_channels', 4),
                'out_channels': config.get('unet_out_channels', 4),
                'cross_attention_dim': config.get('cross_attention_dim', 2048),
                'block_out_channels': config.get('block_out_channels', [320, 640, 1280, 1280]),
                'down_block_types': config.get('down_block_types', [
                    "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", 
                    "CrossAttnDownBlock2D", "DownBlock2D"
                ]),
                'up_block_types': config.get('up_block_types', [
                    "UpBlock2D", "CrossAttnUpBlock2D", 
                    "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"
                ]),
                'transformer_layers_per_block': config.get('transformer_layers_per_block', [1, 2, 10, 10]),
                'attention_head_dim': config.get('attention_head_dim', [5, 10, 20, 20]),
                'layers_per_block': config.get('layers_per_block', 2),
                'act_fn': config.get('act_fn', 'silu'),
                'norm_num_groups': config.get('norm_num_groups', 32),
                'addition_embed_type': config.get('addition_embed_type', 'text_time'),
                'addition_time_embed_dim': config.get('addition_time_embed_dim', 256),
                'projection_class_embeddings_input_dim': config.get('projection_class_embeddings_input_dim', 2816),
                'bs': config.get('bs', 1)
            }
            
            self.bs = config.get('bs', 1)
            self.num_inference_steps = config.get('num_inference_steps', 1)  # Simplified for TTSIM
        
        # Create the three main components
        self.vae = SimpleVAEAutoencoder(f"{name}.vae", self.vae_config)
        self.text_encoder = DualTextEncoderSDXL(f"{name}.text_encoder", self.clip_config)
        self.unet = SDXLUNet(f"{name}.unet", self.unet_config)
        
        # Create input tensors for the complete pipeline
        self.create_input_tensors()
        super().link_op2module()
    
    def create_input_tensors(self):
        """Create input tensors for complete SDXL pipeline"""
        # Text prompt input (token IDs)
        prompt_tensor = F._from_shape(
            f"{self.name}.prompt",
            shape=[self.bs, self.clip_config['max_seq_length']],
            is_param=False,
            is_const=False,
            np_dtype=np.int32
        )
        
        # Image input for VAE encoding
        image_tensor = F._from_shape(
            f"{self.name}.image",
            shape=[self.bs, self.vae_config['in_channels'], self.vae_config['sample_size'], self.vae_config['sample_size']],
            is_param=False,
            is_const=False,
            np_dtype=np.float32
        )
        
        # Timestep for diffusion process
        timestep_tensor = F._from_shape(
            f"{self.name}.timestep",
            shape=[self.bs],
            is_param=False,
            is_const=False,
            np_dtype=np.float32
        )
        
        # SDXL micro-conditioning inputs (simplified)
        time_ids_tensor = F._from_shape(
            f"{self.name}.time_ids",
            shape=[self.bs, 6],
            is_param=False,
            is_const=False,
            np_dtype=np.float32
        )
        
        # Note: Pooled embeddings are now computed by DualTextEncoder, not pre-created
        
        # Include sub-component input tensors for proper graph construction
        sub_component_tensors = {}
        
        # Add VAE input tensors
        if hasattr(self.vae, 'input_tensors'):
            for key, tensor in self.vae.input_tensors.items():
                sub_component_tensors[f'vae_{key}'] = tensor
        
        # Add CLIP input tensors (includes position_ids and causal_mask)
        if hasattr(self.text_encoder, 'input_tensors'):
            for key, tensor in self.text_encoder.input_tensors.items():
                sub_component_tensors[f'clip_{key}'] = tensor
        
        # Add UNet input tensors  
        if hasattr(self.unet, 'input_tensors'):
            for key, tensor in self.unet.input_tensors.items():
                sub_component_tensors[f'unet_{key}'] = tensor
        
        self.input_tensors = {
            'prompt': prompt_tensor,      # Text prompt tokens
            'image': image_tensor,        # Input image for img2img
            'timestep': timestep_tensor,  # Diffusion timestep
            'time_ids': time_ids_tensor,  # SDXL time conditioning
            **sub_component_tensors       # All sub-component tensors for graph construction
        }
    
    def __call__(self, prompt=None, image=None, timestep=None, time_ids=None, **kwargs):
        """
        Complete SDXL Pipeline Forward Pass
        
        Args:
            prompt: Text prompt tokens [B, 77]
            image: Input image [B, 3, 512, 512] 
            timestep: Diffusion timestep [B]
            time_ids: SDXL time conditioning [B, 6]
        
        Returns:
            Generated/denoised image [B, 3, 512, 512]
        """
        # Use input tensors if no arguments provided
        if prompt is None:
            prompt = self.input_tensors['prompt']
        if image is None:
            image = self.input_tensors['image']
        if timestep is None:
            timestep = self.input_tensors['timestep']
        if time_ids is None:
            time_ids = self.input_tensors['time_ids']
        
        # === SDXL Pipeline Forward Pass ===
        
        # 1. Text Encoding (Dual CLIP)
        # Call dual text encoder - returns both embeddings and pooled embeddings
        text_encoder_output = self.text_encoder(
            input_ids=prompt,
            position_ids=None  # Will use default pre-created tensor
        )
        
        # Extract both outputs from dual encoder
        text_embeddings = text_encoder_output['embeddings']      # [B, seq_len, 2048] - concatenated 768+1280
        pooled_embeddings = text_encoder_output['pooled_embeds'] # [B, 1280] - real computed embeddings
        
        # 2. Image Encoding (SDXL VAE Encoder)
        # Encode image to latent distribution parameters
        encoded = self.vae.encode(image)  # Returns encoded tensor (mean + logvar)
        # Sample from encoded distribution using autoencoder's sampler
        latents = self.vae.autoencoder.sample(encoded)  # [B, latent_channels, H/8, W/8]
        
        # 3. Diffusion Process (SDXL UNet)
        # Prepare SDXL conditioning inputs
        denoised_latents = self.unet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=text_embeddings,  # [B, 77, 768] - standard text conditioning
            text_embeds=pooled_embeddings,          # [B, 1280] - pooled text for micro-conditioning  
            time_ids=time_ids                       # [B, 6] - time-based conditioning
        )
        
        # 4. Image Decoding (SDXL VAE Decoder)
        generated_image = self.vae.decode(denoised_latents)  # [B, 3, H, W]
        
        return generated_image
    
    def get_forward_graph(self):
        """Get forward computation graph for ONNX export"""
        GG = super()._get_forward_graph(self.input_tensors)
        return GG
    
    def analytical_param_count(self, verbose=False):
        """Return analytical parameter count for the complete SDXL pipeline"""
        # Enable verbose output for sub-components if requested
        if verbose:
            self._verbose = True
            self.text_encoder._verbose = True
        
        vae_params = self.vae.analytical_param_count()
        clip_params = self.text_encoder.analytical_param_count()
        unet_params = self.unet.analytical_param_count()
        
        total_params = vae_params + clip_params + unet_params
        
        # Optional verbose output for debugging
        if verbose:
            print(f"📊 SDXL Pipeline Parameter Count:")
            print(f"   VAE:         {vae_params:,}")
            print(f"   CLIP:        {clip_params:,}")  
            print(f"   UNet:        {unet_params:,}")
            print(f"   Total:       {total_params:,}")
        
        return total_params

# Simplified pipeline for testing individual components
class SDXLPipelineSimplified(SimNN.Module):
    """
    Simplified SDXL Pipeline for component testing
    Tests each component individually before full integration
    """
    def __init__(self, name, config, component_type='vae'):
        super().__init__()
        self.name = name
        self.config = config
        self.component_type = component_type
        
        setup_ttsim_framework()
        
        if component_type == 'vae':
            self.component = SimpleVAEAutoencoder(f"{name}.vae", config)
            self.input_tensors = self.component.input_tensors
        elif component_type == 'clip':
            self.component = DualTextEncoderSDXL(f"{name}.clip", config)
            self.input_tensors = self.component.input_tensors
        elif component_type == 'unet':
            self.component = SDXLUNet(f"{name}.unet", config)
            self.component.create_input_tensors()
            self.input_tensors = self.component.input_tensors
        else:
            raise ValueError(f"Unknown component type: {component_type}")
        
        super().link_op2module()
    
    def __call__(self, **kwargs):
        return self.component(**kwargs)
    
    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG
    
    def analytical_param_count(self):
        return self.component.analytical_param_count()

# Main classes expected by Polaris framework (must match module name)
SDXLPipeline = SDXLPipelineComplete      # Complete integrated pipeline
SDXLComponent = SDXLPipelineSimplified   # Individual component testing

# For component testing, Polaris expects class names matching the workload names
sdxl_component_vae = SDXLPipelineSimplified
sdxl_component_clip = SDXLPipelineSimplified  
sdxl_component_unet = SDXLPipelineSimplified
sdxl_complete_pipeline = SDXLPipelineComplete

# ============================================================================
# Test Functions
# ============================================================================

def test_sdxl_pipeline_creation():
    """Test SDXL pipeline creation"""
    print("\n=== Testing SDXL Pipeline Creation ===")
    setup_ttsim_framework()
    
    # Test config
    config = {
        'bs': 1,
        'vae_sample_size': 512,
        'unet_sample_size': 64,
        'clip_max_seq_length': 77
    }
    
    try:
        # Test component creation
        pipeline = SDXLPipelineComplete("test_sdxl", config)
        print("✅ SDXL Pipeline created successfully")
        
        # Test parameter counting
        total_params = pipeline.analytical_param_count()
        print(f"✅ Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ SDXL Pipeline creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual SDXL components"""
    print("\n=== Testing Individual SDXL Components ===")
    
    components_config = {
        'vae': {
            'in_channels': 3,
            'latent_channels': 4,
            'sample_size': 64,
            'block_out_channels': [64, 128],
            'bs': 1
        },
        'clip': {
            'vocab_size': 1000,
            'max_seq_length': 32,
            'hidden_size': 256,
            'intermediate_size': 1024,
            'num_attention_heads': 4,
            'max_position_embeddings': 32,
            'model_type': 'standard',
            'bs': 1
        },
        'unet': {
            'sample_size': 32,
            'in_channels': 4,
            'out_channels': 4,
            'cross_attention_dim': 768,
            'block_out_channels': [128, 256, 512, 512],
            'down_block_types': ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
            'up_block_types': ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
            'transformer_layers_per_block': [1, 1, 2, 2],
            'attention_head_dim': [4, 8, 16, 16],
            'layers_per_block': 2,
            'act_fn': 'silu',
            'norm_num_groups': 32,
            'bs': 1
        }
    }
    
    results = {}
    for component_name, config in components_config.items():
        try:
            print(f"\n🔧 Testing {component_name.upper()} component...")
            component = SDXLPipelineSimplified(f"test_{component_name}", config, component_name)
            params = component.analytical_param_count()
            print(f"✅ {component_name.upper()}: {params:,} parameters")
            results[component_name] = True
            
        except Exception as e:
            print(f"❌ {component_name.upper()} failed: {e}")
            results[component_name] = False
    
    passed = sum(results.values())
    total = len(results)
    print(f"\n🎯 Component Tests: {passed}/{total} passed")
    return passed == total

def run_all_tests():
    """Run all SDXL pipeline tests"""
    print("🚀 Running SDXL Pipeline Integration Tests...")
    
    tests = [
        test_individual_components,
        test_sdxl_pipeline_creation,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🎯 SDXL Integration Tests: {passed}/{len(tests)} passed")
    return passed == len(tests)

# Export main classes for polaris.py
# Use SDXLPipelineComplete as the primary pipeline class
SDXLPipelineAutoencoder = SDXLPipelineComplete

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            run_all_tests()
    else:
        print("SDXL Pipeline Integration for Polaris")
        print("Usage: python SDXLPipeline.py --test")
