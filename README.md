# Implementação do FixMatch: Aprendizado Semisupervisionado no CIFAR-10

Este repositório contém um Jupyter Notebook com a implementação completa do algoritmo **FixMatch**, focado na classificação de imagens do dataset CIFAR-10 utilizando técnicas de Aprendizado Semisupervisionado (SSL). O projeto demonstra como melhorar a performance do modelo utilizando dados não rotulados através de consistência de predições entre diferentes aumentações.

O projeto segue o paradigma do FixMatch, que combina duas abordagens principais:
1.  **Pseudo-Labeling:** O modelo atua como um "professor" para si mesmo, gerando rótulos para dados não rotulados quando a confiança da predição em uma imagem (com aumentação fraca) excede um limiar pré-definido (ex: 0.95).
2.  **Consistency Regularization:** O modelo é treinado para que sua predição em uma versão **fortemente aumentada** da mesma imagem corresponda ao pseudo-rótulo gerado.

### Principais Componentes:
* **Arquitetura:** ResNet18 modificada para o CIFAR-10 (ajuste na primeira camada convolucional para suportar inputs $32 \times 32$).
* **Data Augmentation:**
    * *Weak (Fraca):* Random Crop e Random Horizontal Flip.
    * *Strong (Forte):* RandAugment e RandomErasing (Cutout).
* **Split de Dados:** Funções auxiliares para dividir o dataset de treino em subconjuntos rotulados (simulando escassez de labels) e não rotulados.
* **Pipeline de Treino:** Classe `FixMatchTrainer` personalizada que gerencia as perdas supervisionada ($L_s$) e não-supervisionada ($L_u$).


## Estrutura do Código

O notebook está organizado da seguinte forma:
1.  **Carregamento de Dados:** Download e normalização do CIFAR-10.
2.  **Pre-processamento SSL:** Separação balanceada de índices para dados rotulados e não rotulados.
3.  **Definição do Modelo:** Implementação da ResNet18 customizada.
4.  **Aumentação de Dados:** Definição das pipelines de transformação *Weak* e *Strong*.
5.  **Classe Trainer:** Loop de treinamento com cálculo de *masked cross-entropy loss* para os dados não rotulados.
6.  **Visualização:** Funções para inspecionar pseudo-rótulos e comparar históricos de treinamento.
