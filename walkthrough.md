# Walkthrough: Conversão ANN-to-SNN Híbrida (Backbone Convolucional)

## 1. O que foi implementado

### Diretório [`ann2snn/`](file:///home/carlos_fonseca/mestrado/MouseVsAI2025_HCMUS_TheFangs_release/ann2snn/)
* **[`ann2snn_if.py`](file:///home/carlos_fonseca/mestrado/MouseVsAI2025_HCMUS_TheFangs_release/ann2snn/ann2snn_if.py)**:
  * **`DualChannelIFNode`**: Neurônio Integrate-and-Fire de **canal duplo** (positivo e negativo). Trata o comportamento de ativações negativas do `LeakyReLU` original ($\alpha = 0.01$) emitindo pulsos positivos $S^+$ e pulsos inibitórios/negativos $S^-$, combinando-os em $S_{\text{eff}} = S^+ - \alpha \cdot S^-$.
  * **`SpikingConvBackbone`**: Converte as camadas convolucionais (`Conv1` e `Conv2`) em SNN com **codificação por taxa de disparo (*Rate Coding*)** em $T$ passos temporais (*time-steps*).
  * **`HybridNatureVisualEncoder`**: Modelo híbrido integrando o backbone SNN ao cabeçote contínuo (Dense + 3 blocos Mixture-of-Experts / GLU + LayerNorm).
  * **`load_checkpoint_weights`**: Mecanismo de mapeamento e injeção direta dos pesos pré-treinados contidos no checkpoint [`My Behavior-249992.pt`](file:///home/carlos_fonseca/mestrado/MouseVsAI2025_HCMUS_TheFangs_release/track1_simplecnn_glu_norm/checkpoint/My%20Behavior-249992.pt).
  * **`calibrate_thresholds`**: Calibração de limiares por percentil (99.9%) das ativações para evitar saturação precoce ou silêncio dos neurônios de disparo.
  * **`finetune_surrogate_gradients`**: Rotina de ajuste fino via BPTT (*Backpropagation Through Time*) utilizando **Surrogate Gradients** (`surrogate.ATan` do SpikingJelly) para alinhar as representações da SNN com as representações da ANN original (destilação por MSE).
* **[`requirements_snn.txt`](file:///home/carlos_fonseca/mestrado/MouseVsAI2025_HCMUS_TheFangs_release/ann2snn/requirements_snn.txt)**: Relação das dependências necessárias (`torch`, `torchvision`, `spikingjelly`, `numpy`).
* **[`README.md`](file:///home/carlos_fonseca/mestrado/MouseVsAI2025_HCMUS_TheFangs_release/ann2snn/README.md)**: Documentação de uso, arquitetura e comandos de execução.

---

## 2. Como Executar no seu Ambiente

Para executar a demonstração completa (carregamento do checkpoint, calibração e fine-tuning):

```bash
# 1. Ative seu ambiente conda ou venv onde o PyTorch está instalado
 conda activate mouse

# Se precisar criar
# Cria o ambiente (substitua "snn_env" pelo nome que preferir) com Python 3.11
conda create -n snn_env python=3.11 -y

# Ativa o ambiente
conda activate snn_env

# Instalar o PyTorch
# A instalação varia conforme a disponibilidade de GPU (NVIDIA CUDA) ou CPU.
# Com suporte a GPU (NVIDIA / CUDA):
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Apenas CPU (sem GPU dedicada):
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Se der erro o comando para instalar a versão para GPU tente os comando abaixo. Como o pacote é muito grande pode dar erro de espaço insuficiente
# Os comandos abaixo limpam o cache e criam uma pasta temporária para realizar a instalação

pip cache purge
conda clean --all -y

#E depois executei:

mkdir -p ~/pip_tmp
TMPDIR=~/pip_tmp pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

# Testar a instalação
# Verifique se o PyTorch foi instalado corretamente e se ele reconhece a sua GPU:
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"

# Se você instalou a versão para GPU e tiver os drivers configurados, CUDA Available retornará True. Caso tenha instalado a versão para CPU, retornará False.

# 2. Instale o SpikingJelly
pip install spikingjelly

# 3. Execute o script de conversão
python ann2snn/ann2snn_if.py
```
