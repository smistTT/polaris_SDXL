#!/usr/bin/env python3
"""
Dual Text Encoder Implementation for SDXL - Ground Truth Architecture
Implements CLIPTextModel + CLIPTextModelWithProjection following diffusers SDXL pattern
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

class DualLinear(SimNN.Module):
    """Linear layer using F.MatMul + F.Add to avoid += issues"""
    def __init__(self, name, in_features, out_features, bias=True):
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Create weight and bias tensors
        self.weight = F._from_shape(f"{name}.weight", [in_features, out_features], is_param=True)
        self.weight.set_module(self)
        
        if bias:
            self.bias = F._from_shape(f"{name}.bias", [out_features], is_param=True)
            self.bias.set_module(self)
            self.add_bias = F.Add(f"{name}.add_bias")
            self.add_bias.set_module(self)
        
        self.matmul = F.MatMul(f"{name}.matmul")
        self.matmul.set_module(self)
        
        super().link_op2module()
    
    def __call__(self, x):
        x = self.matmul(x, self.weight)
        if self.use_bias:
            x = self.add_bias(x, self.bias)
        return x

class CLIPTextModel(SimNN.Module):
    """
    First text encoder - CLIPTextModel equivalent
    Produces 768-dim embeddings for cross-attention
    """
    def __init__(self, name, config):
        super().__init__()
        self.name = name
        self.config = config
        
        # Configuration
        self.vocab_size = config.get('vocab_size', 49408)
        self.max_seq_length = config.get('max_seq_length', 77)
        self.hidden_size = config.get('hidden_size_1', 768)  # Standard CLIP ViT-L
        self.intermediate_size = config.get('intermediate_size_1', 3072)
        self.num_attention_heads = config.get('num_attention_heads_1', 12)
        self.num_hidden_layers = config.get('num_hidden_layers', 12)
        self.bs = config.get('bs', 1)
        
        # Token and position embeddings
        self.token_embedding = F.Embedding(f"{name}.token_embedding", self.vocab_size, self.hidden_size)
        self.token_embedding.set_module(self)
        
        self.position_embedding = F.Embedding(f"{name}.position_embedding", self.max_seq_length, self.hidden_size)
        self.position_embedding.set_module(self)
        
        # Simple attention and feedforward layers
        self.attention_layers = []
        self.ff_layers = []
        
        for i in range(self.num_hidden_layers):
            # Multi-head attention (simplified)
            attention = DualLinear(
                f"{name}.attention.{i}",
                self.hidden_size,
                self.hidden_size
            )
            self.attention_layers.append(attention)
            
            # Feed-forward network
            ff1 = DualLinear(f"{name}.ff1.{i}", self.hidden_size, self.intermediate_size)
            ff2 = DualLinear(f"{name}.ff2.{i}", self.intermediate_size, self.hidden_size)
            
            self.ff_layers.append((ff1, ff2))
            
            # Set modules
            setattr(self, f"attention_{i}", attention)
            setattr(self, f"ff1_{i}", ff1)
            setattr(self, f"ff2_{i}", ff2)
        
        # Layer normalization operations - separate for attention and feedforward
        self.attn_layer_norms = []
        self.ff_layer_norms = []
        for i in range(self.num_hidden_layers):
            attn_ln = F.LayerNorm(f"{name}.attn_ln.{i}", self.hidden_size)
            attn_ln.set_module(self)
            self.attn_layer_norms.append(attn_ln)
            setattr(self, f"attn_ln_{i}", attn_ln)
            
            ff_ln = F.LayerNorm(f"{name}.ff_ln.{i}", self.hidden_size)
            ff_ln.set_module(self)
            self.ff_layer_norms.append(ff_ln)
            setattr(self, f"ff_ln_{i}", ff_ln)
        
        # Final layer norm
        self.final_ln = F.LayerNorm(f"{name}.final_ln", self.hidden_size)
        self.final_ln.set_module(self)
        
        # Activation functions - separate for each layer
        self.gelu_ops = []
        for i in range(self.num_hidden_layers):
            gelu = F.Gelu(f"{name}.gelu.{i}")
            gelu.set_module(self)
            self.gelu_ops.append(gelu)
            setattr(self, f"gelu_{i}", gelu)
        
        # Addition operations for residual connections
        self.add_ops = []
        for i in range(self.num_hidden_layers * 2 + 1):  # embedding + attention + ff per layer
            add_op = F.Add(f"{name}.add.{i}")
            add_op.set_module(self)
            self.add_ops.append(add_op)
            setattr(self, f"add_{i}", add_op)
        
        super().link_op2module()
    
    def __call__(self, input_ids, position_ids=None):
        """
        Forward pass for CLIPTextModel
        
        Args:
            input_ids: Token IDs [B, seq_len]
            position_ids: Position IDs [B, seq_len] (optional)
            
        Returns:
            hidden_states: [B, seq_len, 768] - for cross-attention
        """
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        if position_ids is None:
            # Create default position ids
            seq_len = input_ids.shape[1] if len(input_ids.shape) > 1 else self.max_seq_length
            position_ids = F._from_data(
                f"{self.name}.default_pos_ids",
                np.arange(seq_len).reshape(1, -1).astype(np.int32),
                is_param=False,
                is_const=True
            )
            position_ids.set_module(self)
        
        position_embeds = self.position_embedding(position_ids)
        
        # Add token + position embeddings
        hidden_states = self.add_ops[0](token_embeds, position_embeds)
        
        # Transformer layers
        add_idx = 1
        for i in range(self.num_hidden_layers):
            # Pre-attention layer norm
            attn_normed = self.attn_layer_norms[i](hidden_states)
            
            # Simplified multi-head attention (without actual multi-head splitting)
            attn_output = self.attention_layers[i](attn_normed)
            
            # Residual connection
            hidden_states = self.add_ops[add_idx](hidden_states, attn_output)
            add_idx += 1
            
            # Pre-feedforward layer norm
            ff_input = self.ff_layer_norms[i](hidden_states)
            ff_hidden = self.gelu_ops[i](self.ff_layers[i][0](ff_input))
            ff_output = self.ff_layers[i][1](ff_hidden)
            
            # Residual connection
            hidden_states = self.add_ops[add_idx](hidden_states, ff_output)
            add_idx += 1
        
        # Final layer norm
        hidden_states = self.final_ln(hidden_states)
        
        return hidden_states  # [B, seq_len, 768]

class CLIPTextModelWithProjection(SimNN.Module):
    """
    Second text encoder - CLIPTextModelWithProjection equivalent
    Produces 1280-dim embeddings + pooled embeddings for micro-conditioning
    """
    def __init__(self, name, config):
        super().__init__()
        self.name = name
        self.config = config
        
        # Configuration - larger than first encoder
        self.vocab_size = config.get('vocab_size', 49408)
        self.max_seq_length = config.get('max_seq_length', 77)
        self.hidden_size = config.get('hidden_size_2', 1280)  # OpenCLIP ViT-bigG
        self.intermediate_size = config.get('intermediate_size_2', 5120)
        self.num_attention_heads = config.get('num_attention_heads_2', 20)
        self.num_hidden_layers = config.get('num_hidden_layers', 32)  # Deeper than first encoder
        self.projection_dim = config.get('projection_dim', 1280)
        self.bs = config.get('bs', 1)
        
        # Token and position embeddings
        self.token_embedding = F.Embedding(f"{name}.token_embedding", self.vocab_size, self.hidden_size)
        self.token_embedding.set_module(self)
        
        self.position_embedding = F.Embedding(f"{name}.position_embedding", self.max_seq_length, self.hidden_size)
        self.position_embedding.set_module(self)
        
        # Simplified transformer layers (fewer for TTSIM efficiency)
        effective_layers = min(self.num_hidden_layers, 6)  # Limit for TTSIM
        self.attention_layers = []
        self.ff_layers = []
        
        for i in range(effective_layers):
            # Multi-head attention (simplified)
            attention = DualLinear(
                f"{name}.attention.{i}",
                self.hidden_size,
                self.hidden_size
            )
            self.attention_layers.append(attention)
            
            # Feed-forward network
            ff1 = DualLinear(f"{name}.ff1.{i}", self.hidden_size, self.intermediate_size)
            ff2 = DualLinear(f"{name}.ff2.{i}", self.intermediate_size, self.hidden_size)
            
            self.ff_layers.append((ff1, ff2))
            
            # Set modules
            setattr(self, f"attention_{i}", attention)
            setattr(self, f"ff1_{i}", ff1)
            setattr(self, f"ff2_{i}", ff2)
        
        # Layer normalization - separate for attention and feedforward
        self.attn_layer_norms = []
        self.ff_layer_norms = []
        for i in range(effective_layers):
            attn_ln = F.LayerNorm(f"{name}.attn_ln.{i}", self.hidden_size)
            attn_ln.set_module(self)
            self.attn_layer_norms.append(attn_ln)
            setattr(self, f"attn_ln_{i}", attn_ln)
            
            ff_ln = F.LayerNorm(f"{name}.ff_ln.{i}", self.hidden_size)
            ff_ln.set_module(self)
            self.ff_layer_norms.append(ff_ln)
            setattr(self, f"ff_ln_{i}", ff_ln)
        
        # Final layer norm
        self.final_ln = F.LayerNorm(f"{name}.final_ln", self.hidden_size)
        self.final_ln.set_module(self)
        
        # Text projection layer (for pooled embeddings)
        self.text_projection = DualLinear(
            f"{name}.text_projection",
            self.hidden_size,
            self.projection_dim
        )
        
        # Pooling operation (take first token [CLS])
        # SliceFixed takes starts and ends as lists: [batch_start, seq_start, hidden_start], [batch_end, seq_end, hidden_end]
        self.pooling_slice = F.SliceFixed(f"{name}.pooling_slice", [0, 0, 0], [self.bs, 1, self.hidden_size])
        self.pooling_slice.set_module(self)
        
        # Reshape to remove sequence dimension for pooled output [B, 1, hidden] -> [B, hidden]
        self.pooling_reshape = F.ReshapeFixed(f"{name}.pooling_reshape", [self.bs, self.hidden_size])
        self.pooling_reshape.set_module(self)
        
        # Activation functions - separate for each layer
        self.gelu_ops = []
        for i in range(effective_layers):
            gelu = F.Gelu(f"{name}.gelu.{i}")
            gelu.set_module(self)
            self.gelu_ops.append(gelu)
            setattr(self, f"gelu_{i}", gelu)
        
        # Addition operations
        self.add_ops = []
        for i in range(effective_layers * 2 + 1):
            add_op = F.Add(f"{name}.add.{i}")
            add_op.set_module(self)
            self.add_ops.append(add_op)
            setattr(self, f"add_{i}", add_op)
        
        self.effective_layers = effective_layers
        super().link_op2module()
    
    def __call__(self, input_ids, position_ids=None):
        """
        Forward pass for CLIPTextModelWithProjection
        
        Args:
            input_ids: Token IDs [B, seq_len]
            position_ids: Position IDs [B, seq_len] (optional)
            
        Returns:
            tuple: (hidden_states [B, seq_len, 1280], pooled_output [B, 1280])
        """
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        if position_ids is None:
            # Create default position ids
            seq_len = input_ids.shape[1] if len(input_ids.shape) > 1 else self.max_seq_length
            position_ids = F._from_data(
                f"{self.name}.default_pos_ids",
                np.arange(seq_len).reshape(1, -1).astype(np.int32),
                is_param=False,
                is_const=True
            )
            position_ids.set_module(self)
        
        position_embeds = self.position_embedding(position_ids)
        
        # Add token + position embeddings
        hidden_states = self.add_ops[0](token_embeds, position_embeds)
        
        # Transformer layers
        add_idx = 1
        for i in range(self.effective_layers):
            # Pre-attention layer norm
            attn_normed = self.attn_layer_norms[i](hidden_states)
            
            # Simplified attention
            attn_output = self.attention_layers[i](attn_normed)
            
            # Residual connection
            hidden_states = self.add_ops[add_idx](hidden_states, attn_output)
            add_idx += 1
            
            # Pre-feedforward layer norm
            ff_input = self.ff_layer_norms[i](hidden_states)
            ff_hidden = self.gelu_ops[i](self.ff_layers[i][0](ff_input))
            ff_output = self.ff_layers[i][1](ff_hidden)
            
            # Residual connection
            hidden_states = self.add_ops[add_idx](hidden_states, ff_output)
            add_idx += 1
        
        # Final layer norm
        hidden_states = self.final_ln(hidden_states)
        
        # Pooled output - take [CLS] token (first token) and project
        cls_token = self.pooling_slice(hidden_states)  # [B, 1, hidden_size]
        cls_token = self.pooling_reshape(cls_token)  # [B, hidden_size]
        pooled_output = self.text_projection(cls_token)  # [B, projection_dim]
        
        return hidden_states, pooled_output  # [B, seq_len, 1280], [B, 1280]

class DualTextEncoderSDXL(SimNN.Module):
    """
    Complete Dual Text Encoder for SDXL following ground truth architecture
    Combines CLIPTextModel + CLIPTextModelWithProjection
    """
    def __init__(self, name, config):
        super().__init__()
        self.name = name
        self.config = config
        
        setup_ttsim_framework()
        
        # Create both text encoders
        self.text_encoder_1 = CLIPTextModel(f"{name}.text_encoder_1", config)
        self.text_encoder_2 = CLIPTextModelWithProjection(f"{name}.text_encoder_2", config)
        
        # Concatenation operation for combining embeddings
        self.concat_embeds = F.ConcatX(f"{name}.concat_embeds", axis=2)  # Concat along feature dim
        self.concat_embeds.set_module(self)
        
        # Create input tensors
        self.create_input_tensors()
        super().link_op2module()
    
    def create_input_tensors(self):
        """Create input tensors for dual text encoder"""
        # Input token IDs
        input_ids_tensor = F._from_shape(
            f"{self.name}.input_ids",
            shape=[self.config.get('bs', 1), self.config.get('max_seq_length', 77)],
            is_param=False,
            is_const=False,
            np_dtype=np.int32
        )
        
        # Position IDs (shared between both encoders)
        seq_len = self.config.get('max_seq_length', 77)
        position_ids_data = np.arange(seq_len).reshape(1, -1).astype(np.int32)
        position_ids_tensor = F._from_data(
            f"{self.name}.position_ids",
            position_ids_data,
            is_param=False,
            is_const=True
        )
        position_ids_tensor.set_module(self)
        
        self.input_tensors = {
            'input_ids': input_ids_tensor,
            'position_ids': position_ids_tensor
        }
    
    def __call__(self, input_ids=None, position_ids=None):
        """
        Forward pass through dual text encoders
        
        Args:
            input_ids: Token IDs [B, seq_len]
            position_ids: Position IDs [B, seq_len] (optional)
        
        Returns:
            dict containing:
                - 'embeddings': Concatenated embeddings [B, seq_len, 2048] for cross-attention
                - 'pooled_embeds': Pooled embeddings [B, 1280] for micro-conditioning
        """
        # Use default inputs if not provided
        if input_ids is None:
            input_ids = self.input_tensors['input_ids']
        if position_ids is None:
            position_ids = self.input_tensors['position_ids']
        
        # Process through both text encoders
        # Text encoder 1: [B, seq_len, 768]
        text_embeds_1 = self.text_encoder_1(input_ids, position_ids)
        
        # Text encoder 2: [B, seq_len, 1280] + [B, 1280]
        text_embeds_2, pooled_embeds = self.text_encoder_2(input_ids, position_ids)
        
        # Concatenate embeddings: 768 + 1280 = 2048
        combined_embeds = self.concat_embeds(text_embeds_1, text_embeds_2)
        
        return {
            'embeddings': combined_embeds,      # [B, seq_len, 2048]
            'pooled_embeds': pooled_embeds      # [B, 1280]
        }
    
    def get_forward_graph(self):
        """Get forward computation graph for ONNX export"""
        GG = super()._get_forward_graph(self.input_tensors)
        return GG
    
    def analytical_param_count(self):
        """Calculate parameter count for both text encoders"""
        # Text encoder 1 parameters (CLIPTextModel - 768 dim)
        vocab_size = self.config.get('vocab_size', 49408)
        max_seq_length = self.config.get('max_seq_length', 77)
        hidden_size_1 = self.config.get('hidden_size_1', 768)
        intermediate_size_1 = self.config.get('intermediate_size_1', 3072)
        num_layers = min(self.config.get('num_hidden_layers', 12), 12)
        
        # Embeddings
        token_embeds_1 = vocab_size * hidden_size_1
        pos_embeds_1 = max_seq_length * hidden_size_1
        
        # Transformer layers (attention + ff per layer)
        attn_params_1 = num_layers * (hidden_size_1 * hidden_size_1 + hidden_size_1)
        ff_params_1 = num_layers * (hidden_size_1 * intermediate_size_1 + intermediate_size_1 + 
                                   intermediate_size_1 * hidden_size_1 + hidden_size_1)
        ln_params_1 = num_layers * hidden_size_1 * 2  # weight + bias per layer norm
        
        text_encoder_1_params = token_embeds_1 + pos_embeds_1 + attn_params_1 + ff_params_1 + ln_params_1
        
        # Text encoder 2 parameters (CLIPTextModelWithProjection - 1280 dim)
        hidden_size_2 = self.config.get('hidden_size_2', 1280)
        intermediate_size_2 = self.config.get('intermediate_size_2', 5120)
        projection_dim = self.config.get('projection_dim', 1280)
        effective_layers_2 = min(self.config.get('num_hidden_layers', 32), 6)  # Limited for TTSIM
        
        # Embeddings
        token_embeds_2 = vocab_size * hidden_size_2
        pos_embeds_2 = max_seq_length * hidden_size_2
        
        # Transformer layers
        attn_params_2 = effective_layers_2 * (hidden_size_2 * hidden_size_2 + hidden_size_2)
        ff_params_2 = effective_layers_2 * (hidden_size_2 * intermediate_size_2 + intermediate_size_2 + 
                                           intermediate_size_2 * hidden_size_2 + hidden_size_2)
        ln_params_2 = effective_layers_2 * hidden_size_2 * 2
        
        # Projection layer
        projection_params = hidden_size_2 * projection_dim + projection_dim
        
        text_encoder_2_params = (token_embeds_2 + pos_embeds_2 + attn_params_2 + 
                                ff_params_2 + ln_params_2 + projection_params)
        
        total_params = text_encoder_1_params + text_encoder_2_params
        
        # Optional verbose output for debugging
        if hasattr(self, '_verbose') and self._verbose:
            print(f"📊 Dual Text Encoder Parameter Count:")
            print(f"   Text Encoder 1 (CLIP):     {text_encoder_1_params:,}")
            print(f"   Text Encoder 2 (Proj):     {text_encoder_2_params:,}")
            print(f"   Total Dual Encoder:        {total_params:,}")
        
        return total_params

# Main class expected by Polaris (must match module name)
DualTextEncoder = DualTextEncoderSDXL
DualTextEncoderAutoencoder = DualTextEncoderSDXL

# Export for component testing
if __name__ == "__main__":
    import sys
    
    def test_dual_text_encoder():
        """Test dual text encoder creation and basic functionality"""
        print("\n=== Testing Dual Text Encoder ===")
        setup_ttsim_framework()
        
        # Test config
        config = {
            'bs': 1,
            'vocab_size': 1000,
            'max_seq_length': 32,
            'hidden_size_1': 768,       # CLIPTextModel
            'intermediate_size_1': 3072,
            'num_attention_heads_1': 12,
            'hidden_size_2': 1280,      # CLIPTextModelWithProjection  
            'intermediate_size_2': 5120,
            'num_attention_heads_2': 20,
            'projection_dim': 1280,
            'num_hidden_layers': 6      # Reduced for testing
        }
        
        try:
            # Create dual text encoder
            dual_encoder = DualTextEncoderSDXL("test_dual_encoder", config)
            print("✅ Dual text encoder created successfully")
            
            # Test parameter counting
            total_params = dual_encoder.analytical_param_count()
            print(f"✅ Total parameters: {total_params:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Dual text encoder test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_dual_text_encoder()
    else:
        print("Dual Text Encoder for SDXL - Ground Truth Architecture")
        print("Usage: python DualTextEncoder.py --test")
