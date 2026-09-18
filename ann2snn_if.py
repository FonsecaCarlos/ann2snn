"""
ann2snn_if.py - Conversão ANN-to-SNN Híbrida do Backbone Convolucional da Track 1
Mouse vs. AI 2025 (HCMUS_TheFangs - NeurIPS 2025)

Características:
- Abordagem Híbrida: Converte o backbone convolucional (Conv1 + Conv2) para SNN,
  preservando o cabeçote contínuo (Dense + 3 blocos MoE/GLU).
- Codificação por taxa de disparo (Rate Coding) com T passos temporais.
- Modelo de neurônio: IF (Integrate-and-Fire).
- Canal duplo de disparos (pulsos positivos e negativos) para tratar ativações LeakyReLU.
- Biblioteca principal: SpikingJelly (spikingjelly.activation_based) com suporte a surrogate gradients.
- Calibração de limiares (Threshold Balancing) e ajuste fino via Gradientes Substitutos (Surrogate Gradients / BPTT).
"""

import os
import sys
import math
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tentativa de importação do SpikingJelly
try:
    from spikingjelly.activation_based import neuron, surrogate, functional, layer
    SPIKINGJELLY_AVAILABLE = True
except ImportError:
    SPIKINGJELLY_AVAILABLE = False
    # Implementação de fallback para surrogate gradient ATan e IFNode caso spikingjelly não esteja instalado
    class ATanSurrogate(torch.autograd.Function):
        alpha = 2.0
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return (x >= 0.0).float()

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            alpha = ATanSurrogate.alpha
            grad_x = (alpha / 2.0) / (1.0 + (math.pi / 2.0 * alpha * x).pow(2)) * grad_output
            return grad_x

    class FallbackSurrogate:
        @staticmethod
        def __call__(x):
            return ATanSurrogate.apply(x)


# ==============================================================================
# 1. COMPONENTES DO CABEÇOTE ANN ORIGINAL (Track 1 NatureVisualEncoder MoE/GLU)
# ==============================================================================

class Swish(nn.Module):
    """Função de ativação Swish / SiLU (x * sigmoid(x))"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class AttentionExpert(nn.Module):
    """Expert baseado em Gated Linear Unit (GLU) e atenção com projeção de saída"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, variant_id: int = 0):
        super().__init__()
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enhanced_features = self.feature_transform(x)
        gate_values = self.gate(enhanced_features)
        attended_features = enhanced_features * gate_values
        return self.output_proj(attended_features)


class QFormerExpert(nn.Module):
    """Expert estilo QFormer com queries aprendíveis"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, variant_id: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = 4
        self.queries = nn.Parameter(torch.randn(self.num_queries, hidden_dim) * 0.02)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            Swish()
        )
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * self.num_queries, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        input_features = self.input_proj(x).unsqueeze(1)
        queries = self.queries.unsqueeze(0).repeat(batch_size, 1, 1)
        q = self.q_proj(queries)
        k = self.k_proj(input_features)
        v = self.v_proj(input_features)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, v)
        queries_updated = self.norm(queries + attended)
        flattened = queries_updated.reshape(batch_size, self.num_queries * self.hidden_dim)
        return self.output_proj(flattened)


class MLPExpert(nn.Module):
    """Expert MLP padrão com conexões residuais"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, variant_id: int = 0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SimpleDiverseGate(nn.Module):
    """Mecanismo de roteamento (gate) com temperatura aprendível para MoE"""
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            Swish(),
            nn.Linear(input_dim // 2, num_experts)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x) / torch.clamp(self.temperature, min=0.1, max=5.0)
        weights = F.softmax(logits, dim=-1)
        mean_usage = weights.mean(dim=0)
        target_usage = 1.0 / self.num_experts
        balance_loss = torch.sum((mean_usage - target_usage) ** 2) * self.num_experts
        return weights, balance_loss


class AttentionQFormerMoELayer(nn.Module):
    """Camada de Mixture-of-Experts (MoE) com 2 experts especializados"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.experts = nn.ModuleList([
            MLPExpert(input_dim, hidden_dim, output_dim, 0),
            AttentionExpert(input_dim, hidden_dim, output_dim, 1)
        ])
        self.gate = SimpleDiverseGate(input_dim, len(self.experts))
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights, balance_loss = self.gate(x)
        output = torch.zeros(x.shape[0], weights.shape[-1], device=x.device)
        for i, expert in enumerate(self.experts):
            w = weights[:, i].unsqueeze(-1)
            output = output + w * expert(x)
        output = output + self.residual_proj(x)
        return output, balance_loss


# ==============================================================================
# 2. NEURÔNIO IF DE CANAL DUPLO (DUAL-CHANNEL IF NODE)
# ==============================================================================

class DualChannelIFNode(nn.Module):
    """
    Neurônio Integrate-and-Fire (IF) com Canal Duplo para tratar LeakyReLU.

    Motivação:
    O LeakyReLU produz valores contínuos positivos (f(x) = x, x >= 0) e negativos
    (f(x) = alpha * x, x < 0). Neurônios SNN biológicos convencionais só disparam
    pulsos positivos (0 ou 1).

    Este módulo opera com dois canais de integração paralelos:
    1. Canal Positivo (S+): Integra excitações x+ = ReLU(x) e gera disparos positivos (+1).
    2. Canal Negativo (S-): Integra inibições x- = ReLU(-x) e gera disparos negativos (-1).
    
    A saída efetiva combinada ao longo do tempo é:
        S_eff[t] = S+[t] - alpha * S-[t]
    
    Compatível nativamente com SpikingJelly e Gradientes Substitutos (Surrogate Gradients).
    """
    def __init__(
        self,
        v_threshold: float = 1.0,
        negative_slope: float = 0.01,
        surrogate_fn: Optional[Any] = None,
        step_mode: str = 'm'
    ):
        super().__init__()
        self.v_threshold = float(v_threshold)
        self.negative_slope = float(negative_slope)
        self.step_mode = step_mode

        if SPIKINGJELLY_AVAILABLE:
            if surrogate_fn is None:
                surrogate_fn = surrogate.ATan(alpha=2.0)
            # Canal Positivo
            self.node_pos = neuron.IFNode(
                v_threshold=self.v_threshold,
                v_reset=0.0,
                surrogate_function=surrogate_fn,
                step_mode=step_mode
            )
            # Canal Negativo
            self.node_neg = neuron.IFNode(
                v_threshold=self.v_threshold,
                v_reset=0.0,
                surrogate_function=surrogate_fn,
                step_mode=step_mode
            )
        else:
            self.node_pos = None
            self.node_neg = None
            self.surrogate_fn = FallbackSurrogate()
            self.v_pos = 0.0
            self.v_neg = 0.0

    def reset(self):
        """Reinicia os potenciais de membrana de ambos os canais."""
        if SPIKINGJELLY_AVAILABLE:
            functional.reset_net(self.node_pos)
            functional.reset_net(self.node_neg)
        else:
            self.v_pos = 0.0
            self.v_neg = 0.0

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executa a passagem temporal do canal duplo.
        
        Args:
            x_seq (torch.Tensor): Tensor de entrada ao longo do tempo.
                                  Formato: [T, B, C, H, W].

        Returns:
            s_eff (torch.Tensor): Pulso efetivo combinado [T, B, C, H, W] = S+ - alpha * S-
            s_pos (torch.Tensor): Pulsos positivos [T, B, C, H, W]
            s_neg (torch.Tensor): Pulsos negativos [T, B, C, H, W]
        """
        # Decomposição em corrente positiva e negativa
        x_pos = F.relu(x_seq)
        x_neg = F.relu(-x_seq)

        if SPIKINGJELLY_AVAILABLE:
            s_pos = self.node_pos(x_pos)
            s_neg = self.node_neg(x_neg)
        else:
            # Fallback nativo PyTorch com surrogate gradient ATan
            T, B = x_seq.shape[0], x_seq.shape[1]
            s_pos_list, s_neg_list = [], []
            v_p = torch.zeros_like(x_pos[0])
            v_n = torch.zeros_like(x_neg[0])

            for t in range(T):
                v_p = v_p + x_pos[t]
                v_n = v_n + x_neg[t]

                sp = self.surrogate_fn(v_p - self.v_threshold)
                sn = self.surrogate_fn(v_n - self.v_threshold)

                # Reset por subtração (soft reset)
                v_p = v_p - sp * self.v_threshold
                v_n = v_n - sn * self.v_threshold

                s_pos_list.append(sp)
                s_neg_list.append(sn)

            s_pos = torch.stack(s_pos_list, dim=0)
            s_neg = torch.stack(s_neg_list, dim=0)

        # Combinação linear considerando o negative_slope do LeakyReLU
        s_eff = s_pos - self.negative_slope * s_neg
        return s_eff, s_pos, s_neg


# ==============================================================================
# 3. BACKBONE CONVOLUCIONAL SNN (SpikingConvBackbone)
# ==============================================================================

class SpikingConvBackbone(nn.Module):
    """
    Backbone Convolucional SNN correspondente às camadas conv_layers do NatureVisualEncoder.

    Arquitetura:
    - Conv1: Conv2d(3, 16, kernel_size=8, stride=4)
    - IF1:   DualChannelIFNode (Canais + e -)
    - Conv2: Conv2d(16, 32, kernel_size=4, stride=2)
    - IF2:   DualChannelIFNode (Canais + e -)
    - Decodificação temporal por taxa média (Rate Decoding)
    """
    def __init__(
        self,
        initial_channels: int = 3,
        T: int = 16,
        v_threshold1: float = 1.0,
        v_threshold2: float = 1.0,
        negative_slope: float = 0.01,
        surrogate_fn: Optional[Any] = None
    ):
        super().__init__()
        self.T = T
        self.initial_channels = initial_channels

        self.conv1 = nn.Conv2d(initial_channels, 16, kernel_size=8, stride=4)
        self.if1 = DualChannelIFNode(
            v_threshold=v_threshold1,
            negative_slope=negative_slope,
            surrogate_fn=surrogate_fn,
            step_mode='m'
        )

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.if2 = DualChannelIFNode(
            v_threshold=v_threshold2,
            negative_slope=negative_slope,
            surrogate_fn=surrogate_fn,
            step_mode='m'
        )

        # Fatores de escala aprendíveis/calibráveis pós-taxa de disparo
        self.scale1 = nn.Parameter(torch.tensor(v_threshold1), requires_grad=True)
        self.scale2 = nn.Parameter(torch.tensor(v_threshold2), requires_grad=True)

        # Tamanho do achatamento final: 32 * 9 * 9 = 2592 para entrada 84x84
        self.final_flat = 32 * 9 * 9

    def reset(self):
        """Reinicia todos os neurônios da SNN."""
        self.if1.reset()
        self.if2.reset()

    def forward(self, visual_obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Executa a passagem para a entrada visual usando Rate Coding ao longo de T passos.

        Args:
            visual_obs (torch.Tensor): Imagens de entrada [B, C, H, W] ou [B, H, W, C].

        Returns:
            flat_features (torch.Tensor): Saída contínua decodificada [B, final_flat].
            spikes_info (dict): Dicionário com estatísticas e tensores de disparos dos canais.
        """
        # Ajusta shape para [B, C, H, W] se vier no formato Unity [B, H, W, C]
        if visual_obs.ndim == 4 and visual_obs.shape[-1] == self.initial_channels:
            visual_obs = visual_obs.permute(0, 3, 1, 2)

        B, C, H, W = visual_obs.shape
        T = self.T

        # Codificação por taxa constante ao longo de T passos (Rate Coding)
        # Formato: [T, B, C, H, W]
        x_seq = visual_obs.unsqueeze(0).repeat(T, 1, 1, 1, 1)

        # --- Camada 1: Conv1 ---
        # Achatamos T e B para aplicar Conv2d padrão paralelamente no tempo
        x_flat = x_seq.view(T * B, C, H, W)
        c1_out = self.conv1(x_flat)
        _, C1, H1, W1 = c1_out.shape
        c1_seq = c1_out.view(T, B, C1, H1, W1)

        # Neurônio IF Canal Duplo 1
        s_eff1, s_pos1, s_neg1 = self.if1(c1_seq)

        # --- Camada 2: Conv2 ---
        # O sinal transmitido para a segunda camada é escalado pelo pulso efetivo
        in2_flat = (s_eff1 * self.scale1).view(T * B, C1, H1, W1)
        c2_out = self.conv2(in2_flat)
        _, C2, H2, W2 = c2_out.shape
        c2_seq = c2_out.view(T, B, C2, H2, W2)

        # Neurônio IF Canal Duplo 2
        s_eff2, s_pos2, s_neg2 = self.if2(c2_seq)

        # --- Decodificação Temporal (Rate Decoding) ---
        # Média temporal dos pulsos efetivos ao longo de T passos
        # S_mean: [B, C2, H2, W2]
        s_mean = s_eff2.mean(dim=0) * self.scale2
        flat_features = s_mean.reshape(B, -1)

        spikes_info = {
            "s_pos1_rate": s_pos1.detach().mean().item(),
            "s_neg1_rate": s_neg1.detach().mean().item(),
            "s_pos2_rate": s_pos2.detach().mean().item(),
            "s_neg2_rate": s_neg2.detach().mean().item(),
            "s_eff1": s_eff1,
            "s_eff2": s_eff2
        }

        return flat_features, spikes_info


# ==============================================================================
# 4. MODELO HÍBRIDO COMPLETO (HybridNatureVisualEncoder)
# ==============================================================================

class HybridNatureVisualEncoder(nn.Module):
    """
    Modelo Híbrido:
    - Backbone Convolucional: SNN (SpikingConvBackbone com IF de canal duplo e rate coding)
    - Cabeçote de Decisão: ANN Contínua (Camada Densa Linear + 3 Blocos MoE/GLU + LayerNorm)
    """
    def __init__(
        self,
        height: int = 84,
        width: int = 84,
        initial_channels: int = 3,
        output_size: int = 256,
        T: int = 16,
        v_threshold: float = 1.0,
        negative_slope: float = 0.01,
        surrogate_fn: Optional[Any] = None
    ):
        super().__init__()
        self.output_size = output_size
        self.final_flat = 32 * 9 * 9  # Para 84x84: 2592

        # 1. Backbone SNN
        self.spiking_backbone = SpikingConvBackbone(
            initial_channels=initial_channels,
            T=T,
            v_threshold1=v_threshold,
            v_threshold2=v_threshold,
            negative_slope=negative_slope,
            surrogate_fn=surrogate_fn
        )

        # 2. Camada Densa Contínua (ANN)
        self.dense = nn.Sequential(
            nn.Linear(self.final_flat, output_size),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

        # 3. Três Blocos MoE/GLU Contínuos (ANN)
        self.moe_layer1 = AttentionQFormerMoELayer(output_size, output_size, output_size)
        self.moe_layer2 = AttentionQFormerMoELayer(output_size, output_size, output_size)
        self.moe_layer3 = AttentionQFormerMoELayer(output_size, output_size, output_size)

        # 4. Normalizações por Camada (LayerNorm)
        self.norm1 = nn.LayerNorm(output_size)
        self.norm2 = nn.LayerNorm(output_size)
        self.norm3 = nn.LayerNorm(output_size)

        self.moe_loss = 0.0

    def reset_snn(self):
        """Reinicia estados temporais da SNN."""
        self.spiking_backbone.reset()

    def forward(self, visual_obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Passagem direta do modelo híbrido.
        """
        # Passagem pelo backbone SNN
        flat_snn, spikes_info = self.spiking_backbone(visual_obs)

        # Cabeçote Contínuo (ANN)
        x = self.dense(flat_snn)

        x1, loss1 = self.moe_layer1(x)
        x1 = self.norm1(x1)

        x2, loss2 = self.moe_layer2(x1)
        x2 = self.norm2(x2)

        x3, loss3 = self.moe_layer3(x2)
        x3 = self.norm3(x3)

        if self.training:
            self.moe_loss = loss1 + loss2 + loss3

        return x3, spikes_info


# ==============================================================================
# 5. CARREGAMENTO DOS PESOS PRÉ-TREINADOS (.pt)
# ==============================================================================

def load_checkpoint_weights(
    hybrid_model: HybridNatureVisualEncoder,
    checkpoint_path: str,
    device: str = "cpu"
) -> Dict[str, int]:
    """
    Carrega os pesos do checkpoint original (My Behavior-249992.pt) para o modelo híbrido.
    
    Retorna contagem de tensores carregados e ignorados.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint não encontrado em: {checkpoint_path}")

    print(f"[*] Carregando pesos de: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Identificar se os pesos estão em 'model', 'state_dict' ou raiz
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    matched_keys = 0
    missing_keys = []

    model_dict = hybrid_model.state_dict()

    # Mapeamento de chaves do modelo original para o modelo híbrido
    # conv_layers.0 -> spiking_backbone.conv1
    # conv_layers.2 -> spiking_backbone.conv2
    # dense.* -> dense.*
    # moe_layer*.* -> moe_layer*.*
    # norm*.* -> norm*.*

    prefix_map = {
        "conv_layers.0.": "spiking_backbone.conv1.",
        "conv_layers.2.": "spiking_backbone.conv2.",
    }

    adapted_dict = {}
    for k, v in state_dict.items():
        # Remove prefixos comuns de políticas do ML-Agents caso existam
        clean_k = k
        for p in ["visual_encoder.", "_visual_encoder.", "policy.", "network."]:
            if clean_k.startswith(p):
                clean_k = clean_k[len(p):]

        # Substitui prefixos de convolução para o backbone SNN
        target_k = clean_k
        for src_p, dst_p in prefix_map.items():
            if clean_k.startswith(src_p):
                target_k = clean_k.replace(src_p, dst_p)
                break

        if target_k in model_dict:
            if model_dict[target_k].shape == v.shape:
                adapted_dict[target_k] = v
                matched_keys += 1
            else:
                print(f"[!] Incompatibilidade de shape para {target_k}: {model_dict[target_k].shape} vs {v.shape}")
        else:
            missing_keys.append(k)

    model_dict.update(adapted_dict)
    hybrid_model.load_state_dict(model_dict)
    print(f"[✓] {matched_keys} tensores de parâmetros carregados com sucesso!")
    return {"matched": matched_keys, "total_model_keys": len(model_dict)}


# ==============================================================================
# 6. CALIBRAÇÃO DE LIMIARES (THRESHOLD BALANCING)
# ==============================================================================

def calibrate_thresholds(
    spiking_backbone: SpikingConvBackbone,
    sample_images: torch.Tensor,
    percentile: float = 99.9
):
    """
    Calibra os limiares V_th e escalas do backbone SNN usando ativações reais ou de calibração.
    Evita tanto o silêncio neuronal quanto a saturação precoce.
    """
    print(f"[*] Calibrando limiares com {sample_images.shape[0]} amostras (Percentil: {percentile}%)...")
    spiking_backbone.eval()

    with torch.no_grad():
        if sample_images.ndim == 4 and sample_images.shape[-1] == spiking_backbone.initial_channels:
            sample_images = sample_images.permute(0, 3, 1, 2)

        # Camada 1: Saída conv1
        c1 = spiking_backbone.conv1(sample_images)
        c1_abs = torch.abs(c1)
        # Calcula valor de referência para limiar baseado no percentil
        k1 = int((percentile / 100.0) * c1_abs.numel())
        v1_th = float(torch.kthvalue(c1_abs.flatten(), k1).values.item())
        v1_th = max(v1_th, 1e-3)

        # Atualiza neurônio 1
        spiking_backbone.if1.v_threshold = v1_th
        if SPIKINGJELLY_AVAILABLE and spiking_backbone.if1.node_pos is not None:
            spiking_backbone.if1.node_pos.v_threshold = v1_th
            spiking_backbone.if1.node_neg.v_threshold = v1_th
        spiking_backbone.scale1.data.fill_(v1_th)

        # Camada 2: Saída conv2
        c2 = spiking_backbone.conv2(F.leaky_relu(c1, negative_slope=spiking_backbone.if1.negative_slope))
        c2_abs = torch.abs(c2)
        k2 = int((percentile / 100.0) * c2_abs.numel())
        v2_th = float(torch.kthvalue(c2_abs.flatten(), k2).values.item())
        v2_th = max(v2_th, 1e-3)

        # Atualiza neurônio 2
        spiking_backbone.if2.v_threshold = v2_th
        if SPIKINGJELLY_AVAILABLE and spiking_backbone.if2.node_pos is not None:
            spiking_backbone.if2.node_pos.v_threshold = v2_th
            spiking_backbone.if2.node_neg.v_threshold = v2_th
        spiking_backbone.scale2.data.fill_(v2_th)

    print(f"[✓] Limiares calibrados: V_th1 = {v1_th:.4f}, V_th2 = {v2_th:.4f}")


# ==============================================================================
# 7. AJUSTE FINO COM SURROGATE GRADIENTS (BPTT FEATURE DISTILLATION)
# ==============================================================================

def finetune_surrogate_gradients(
    hybrid_model: HybridNatureVisualEncoder,
    ann_conv_layers: nn.Sequential,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 5,
    lr: float = 1e-4,
    device: str = "cpu"
) -> list:
    """
    Realiza o ajuste fino do backbone SNN usando Gradientes Substitutos (BPTT).
    
    A função de perda alinha as representações do backbone SNN com as representações
    do backbone ANN original (Destilação de Características / MSE Loss), preservando
    o poder de representação original sem esquecimento catastrófico.
    """
    print(f"\n[*] Iniciando Ajuste Fino com Gradientes Substitutos ({num_epochs} épocas, lr={lr})...")

    hybrid_model.to(device)
    ann_conv_layers.to(device)
    ann_conv_layers.eval()

    # Otimizamos apenas os parâmetros do backbone SNN e escalas
    optimizer = torch.optim.Adam(hybrid_model.spiking_backbone.parameters(), lr=lr)
    loss_history = []

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        batches = 0

        for batch_idx, batch_data in enumerate(dataloader):
            images = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
            images = images.to(device)

            if images.ndim == 4 and images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)

            # 1. Alvo da ANN contínua original
            with torch.no_grad():
                ann_features = ann_conv_layers(images).reshape(images.shape[0], -1)

            # 2. Saída do backbone SNN híbrido
            hybrid_model.reset_snn()
            snn_features, spikes_info = hybrid_model.spiking_backbone(images)

            # 3. Perda de Alinhamento (MSE)
            loss = F.mse_loss(snn_features, ann_features)

            optimizer.zero_grad()
            loss.backward()

            # Gradientes substitutos permitem a propagação através dos disparos
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(batches, 1)
        loss_history.append(avg_loss)
        print(f"  [Época {epoch:02d}/{num_epochs:02d}] Perda MSE (SNN vs ANN): {avg_loss:.6f} | "
              f"Taxas de Disparo: Camada 1 (+:{spikes_info['s_pos1_rate']:.3f}, -:{spikes_info['s_neg1_rate']:.3f}) | "
              f"Camada 2 (+:{spikes_info['s_pos2_rate']:.3f}, -:{spikes_info['s_neg2_rate']:.3f})")

    print("[✓] Ajuste fino com Surrogate Gradients concluído com sucesso!")
    return loss_history


# ==============================================================================
# 8. DEMONSTRAÇÃO E VERIFICAÇÃO (MAIN)
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DEMO: Conversão ANN-to-SNN Híbrida com IF de Canal Duplo e SpikingJelly")
    print(f"SpikingJelly instalado: {SPIKINGJELLY_AVAILABLE}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo de execução: {device}")

    # 1. Instanciação do modelo híbrido
    T_timesteps = 16
    hybrid_net = HybridNatureVisualEncoder(
        height=84,
        width=84,
        initial_channels=3,
        output_size=256,
        T=T_timesteps,
        v_threshold=1.0,
        negative_slope=0.01
    ).to(device)

    # 2. Criação do backbone ANN de referência para comparação
    ann_conv = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=8, stride=4),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Conv2d(16, 32, kernel_size=4, stride=2),
        nn.LeakyReLU(negative_slope=0.01)
    ).to(device)

    # 3. Tentativa de carregamento do checkpoint oficial da Track 1
    checkpoint_file = os.path.join(
        os.path.dirname(__file__), "..", "track1_simplecnn_glu_norm", "checkpoint", "My Behavior-249992.pt"
    )

    if os.path.exists(checkpoint_file):
        try:
            load_checkpoint_weights(hybrid_net, checkpoint_file, device=device)
            # Copiar os pesos convolucionais também para a ANN de referência
            ann_conv[0].weight.data.copy_(hybrid_net.spiking_backbone.conv1.weight.data)
            ann_conv[0].bias.data.copy_(hybrid_net.spiking_backbone.conv1.bias.data)
            ann_conv[2].weight.data.copy_(hybrid_net.spiking_backbone.conv2.weight.data)
            ann_conv[2].bias.data.copy_(hybrid_net.spiking_backbone.conv2.bias.data)
        except Exception as e:
            print(f"[!] Aviso: Não foi possível carregar o checkpoint: {e}")
    else:
        print(f"[!] Checkpoint não encontrado em {checkpoint_file}. Usando pesos de teste.")

    # 4. Amostras sintéticas para calibração e teste (8 amostras, 84x84x3)
    sample_imgs = torch.randn(8, 3, 84, 84, device=device)

    # 5. Calibração dos limiares
    calibrate_thresholds(hybrid_net.spiking_backbone, sample_imgs)

    # 6. Passagem direta e teste de inferência
    hybrid_net.reset_snn()
    with torch.no_grad():
        out, info = hybrid_net(sample_imgs)
        print(f"\n[✓] Forward Pass realizado com sucesso!")
        print(f"    Shape da saída final do encoder: {out.shape} (Esperado: [8, 256])")
        print(f"    Disparos Camada 1: Canal Positivo: {info['s_pos1_rate']:.4f} | Canal Negativo: {info['s_neg1_rate']:.4f}")
        print(f"    Disparos Camada 2: Canal Positivo: {info['s_pos2_rate']:.4f} | Canal Negativo: {info['s_neg2_rate']:.4f}")

    # 7. Simulação de Fine-Tuning com Surrogate Gradients
    dataset = torch.utils.data.TensorDataset(torch.randn(32, 3, 84, 84))
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    finetune_surrogate_gradients(hybrid_net, ann_conv, loader, num_epochs=3, lr=1e-4, device=device)

    print("\n" + "=" * 70)
    print("Módulo ann2snn/ann2snn_if.py pronto para uso e integração!")
    print("=" * 70)

