# Simulação de Máquina Térmica Quântica com PennyLane

Este projeto implementa simulações de uma máquina térmica quântica e otimização de parâmetros usando aprendizado de máquina quântico. O código foi refatorado a partir de um notebook Jupyter para uma estrutura de projeto mais modular e organizada.

## Estrutura do Projeto

```
.
├── README.md                # Este arquivo
├── requirements.txt         # Dependências do Python
├── main.ipynb               # Notebook principal para executar as simulações
├── RO_18.ipynb              # O notebook original (mantido para referência)
└── src/
    ├── __init__.py
    ├── circuits.py          # Classes para a construção de circuitos quânticos
    ├── optimization.py      # Classe para otimização e aprendizado de máquina
    └── building_blocks.py   # Funções auxiliares (Hamiltonianos, portas, etc.)
```

## Instalação

1.  **Clone o repositório (se estiver no GitHub):**
    ```bash
    git clone <url-do-seu-repositorio>
    cd <nome-do-repositorio>
    ```

2.  **Crie um ambiente virtual (recomendado):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    O notebook original também usava `apt-get` para instalar o `texlive-latex-extra` para a renderização de fontes com LaTeX no Matplotlib. Se você precisar dessa funcionalidade, pode ser necessário instalar pacotes LaTeX no seu sistema.
    ```bash
    pip install -r requirements.txt
    ```

## Como Usar

O ponto de entrada principal para executar as simulações é o notebook `main.ipynb`.

1.  Abra o Jupyter Notebook ou JupyterLab:
    ```bash
    jupyter notebook
    ```
2.  Navegue e abra o arquivo `main.ipynb`.
3.  O notebook irá guiá-lo através da importação dos módulos, configuração dos parâmetros e execução da simulação de otimização.

**Nota Importante:** O código no módulo `src/building_blocks.py` contém uma função placeholder chamada `compute_average_work`. A implementação original desta função estava em um arquivo externo (`qhe_cycle_qtd_quantities.py`) que não fazia parte do notebook. Você precisará fornecer a implementação correta para que a otimização funcione.

## Código Fonte

-   **`src/circuits.py`**: Contém as classes `Quantum_Circuits_Emulation` e `Quantum_Circuits`, que são responsáveis por construir e executar os circuitos quânticos usando PennyLane.
-   **`src/optimization.py`**: Contém a classe `QuantumMachineLearning`, que implementa o otimizador Adam para encontrar os parâmetros ótimos do circuito para maximizar a ergotropia.
-   **`src/building_blocks.py`**: Fornece todas as funções auxiliares, como a criação de Hamiltonianos (Heisenberg XX, XY, XYZ), blocos de circuito para termalização e correlação, e as funções que aplicam a evolução temporal.
