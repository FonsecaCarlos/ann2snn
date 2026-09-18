# Módulo ANN2SNN: Conversão Híbrida para Redes Neurais de Disparos (SNN)

Este diretório contém a implementação da conversão do modelo vencedor da Track 1 (*Mouse vs. AI 2025* — Equipe HCMUS_TheFangs) para uma Rede Neural de Disparos (*Spiking Neural Network* - SNN).

Agradecemos a colaboração de Phu-Hoa, Chi-Nguyen, Duy Minh, Phu Quy, Trung Kiet, and Team HCMUS_TheFangs, que nos forneceram seu modelo campeão.

---

## 1. Visão Geral da Abordagem Híbrida

Para preservar a capacidade representacional do modelo que alcançou 95.40% de score na competição e contornar a complexidade de multiplicação de disparos em blocos de atenção/MoE, adotamos uma **arquitetura híbrida**:

* **Backbone Convolucional (SNN):**
  * As 2 camadas convolucionais (`conv1` e `conv2`) são convertidas para neurônios de disparo **Integrate-and-Fire (IF)**.
  * Utiliza **codificação por taxa de disparo (*Rate Coding*)** simulada ao longo de $T$ passos temporais (*time-steps*).
  * **Canal Duplo de Disparos (*Dual-Channel*):** Desenvolvido especificamente para tratar o `LeakyReLU` original.
    * Canal Positivo ($S^+$): integra ativações positivas e emite pulsos $+1$.
    * Canal Negativo ($S^-$): integra ativações negativas e emite pulsos ponderados pelo $\alpha = 0.01$ do LeakyReLU.
    * Sinal efetivo combinado: $S_{\text{eff}}[t] = S^+[t] - \alpha \cdot S^-[t]$.
  * Decodificação temporal por taxa média (*Rate Decoding*): $\bar{S} = \frac{1}{T} \sum_{t=1}^T S_{\text{eff}}[t] \times \text{scale}$.

* **Cabeçote de Decisão (ANN):**
  * Mantém a camada densa linear de 256 unidades e os 3 blocos empilhados de **Mixture-of-Experts (MoE) / Gated Linear Unit (GLU)** contínuos com `LayerNorm`.

---

## 2. Estrutura dos Arquivos

```
ann2snn/
├── ann2snn_if.py           # Implementação do modelo híbrido, canal duplo IF, calibração e fine-tuning
├── requirements_snn.txt    # Dependências necessárias (PyTorch, SpikingJelly)
└── README.md               # Este documento explicativo
```

---

## 3. Como Executar

### Pré-requisitos
No seu ambiente virtual ou conda (ex: o ambiente `mouse` do projeto):

Leia o arquivo @walkthrough.md antes de prosseguir.

```bash
pip install -r ann2snn/requirements_snn.txt
```

### Execução do Script
Para executar a verificação completa (carregamento dos pesos do checkpoint `My Behavior-249992.pt`, calibração de limiares, forward pass e demonstração do ajuste fino com Surrogate Gradients):

```bash
python ann2snn/ann2snn_if.py
```

---

## 4. Etapas Principais Implementadas no Código

1. **`DualChannelIFNode`:** Gerencia simultaneamente os disparos positivos e negativos via `spikingjelly.activation_based.neuron.IFNode`.
2. **`load_checkpoint_weights`:** Mapeia automaticamente as chaves de convolução `conv_layers.0.*` e `conv_layers.2.*` e o cabeçote MoE do arquivo `My Behavior-249992.pt` para o modelo híbrido.
3. **`calibrate_thresholds`:** Calibra os limiares $V_{\text{th}}$ de cada camada usando o percentil 99.9% das ativações para balancear a taxa de disparo e evitar saturação.
4. **`finetune_surrogate_gradients`:** Executa o ajuste fino via *Backpropagation Through Time (BPTT)* com o gradiente substituto `surrogate.ATan`, minimizando o erro quadrático médio (MSE) entre as características extraídas pela SNN e pela ANN original.

