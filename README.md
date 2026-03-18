# Simulação de Máquinas Térmicas Quânticas

Este projeto implementa um sistema de simulação de motores e máquinas térmicas quânticas. A base de código original procedimental foi **completamente refatorada** para seguir princípios estritos de Orientação a Objetos (OO), dividida em responsabilidades únicas, visando facilitar a pesquisa e desenvolvimento. 

A arquitetura usa o [PennyLane](https://pennylane.ai/) para a construção dos circuitos quânticos parametrizados e o [QuTiP](https://qutip.org/) para a manipulação dos estados quânticos (Matrizes de Densidade e Vetores de Estado).

---

## 🏗️ Arquitetura Orientada a Objetos

O código foi modulado da seguinte forma para maximizar a reusabilidade e clareza:

### `src/core/topologies.py`
Gerencia como os qubits estão fisicamente/logicamente conectados.
- Fornece a interface `Topology` e implementações de padrões de conexão (`LinearTopology`, `RingTopology`, `StarTopology`, `CompleteTopology`, entre outras). Cada classe entrega uma lista de pares interagentes pelo método genérico `.get_pairs()`.

### `src/core/operations.py`
Encapsula as lógicas das portas quânticas paramétricas.
- Fornece a interface `QuantumOperation` com o método abstrato `.apply()`.
- **Termalização**: `ThermalizationOperation`.
- **Correlações**: `IsotropicCorrelation`, `ParametricCorrelation`, `ImaginaryParametricCorrelation`.
- **Evolução de Extração (Work)**: Evoluções temporais decompostas de Trotter (ex. `TrotterizedHeisenbergXX`, `TrotterizedHeisenbergXYZ`).

### `src/core/hamiltonians.py`
Construtores unificados de observáveis para testes analíticos ou extração de gradientes.
- `HeisenbergXX`, `HeisenbergXY`, `IsotropicHeisenberg` formam Hamiltonianos customizados no PennyLane sob a interface abstrata `ObservableBuilder`.

### `src/core/quantum_system.py`
Controlador central do PennyLane (`QuantumThermalMachine`).
- Compila instâncias flexíveis de `QuantumOperation` nos QNodes internos (`__initial_state_circuit`, `__final_state_circuit`, `__swap_circuit`), evitando passagem solta de funções e variáveis no estado global.
- Providencia utilitários para desenho visual (`draw`) e transpiladores de hardware da IBM (`compiled_ibm`).

### `src/core/thermodynamics.py`
Calculadora de propriedades termodinâmicas sem conservação de estado desnecessária.
- Expõe a métrica principal da classe `ThermodynamicsCalculator` com método `.compute_average_work()` para resolver os gradientes.

### `src/optimization/ergotropy_optimizer.py`
Implementação orientada a objetos do Otimizador Quântico (`ErgotropyOptimizer`).
- Realiza a otimização de gradiente local baseada no algorítmo estocástico Adam. As derivadas de evolução da ergotropia quântica (Work) são calculadas por intermédio da métrica de diferenças finitas aproximadas (epsilon).

---

## 🧪 Testes Unitários

Seguindo as regras de estabilidade do código, todo o sistema possui cobertura local de testes via `pytest`. 

### Como Rodar

Crie e ative um ambiente virtual (venv), e então instale as dependências:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pennylane qutip pytest pytest-mock
```

A estrutura de testes encontra-se em `tests/`. Para rodar a suite completa e garantir a eficácia do refatoramento:
```bash
PYTHONPATH=. pytest tests/ -v
```

Os testes incluem:
1. Validação autônoma de todas construções topológicas geométricas.
2. Integridade dos transformadores de correlação, termalização e trabalho Trotterizados.
3. Compatibilidade das portas de matriz do Hamiltoniano final e QuTiP.
4. Otimizador estocástico Adam convergindo corretamente para estados minimizadores de gradiente sem erro na conversão dos Shapes da API QuTiP para matrizes de PennyLane.
