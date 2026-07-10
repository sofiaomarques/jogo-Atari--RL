# Neon Intercept RL

Um jogo 2D estilo Atari feito para treinar modelos de aprendizado por reforco.

O agente controla uma nave, desvia de meteoros, coleta energia e pode atirar. A interface principal e um ambiente compativel com Gymnasium:

- `obs_mode="pixels"`: observacao 84x84 em escala de cinza, como entrada classica de Atari.
- `obs_mode="rgb"`: frame RGB bruto de 160x210.
- `obs_mode="state"`: vetor numerico compacto para treinos rapidos em CPU.

## Acoes

| Valor | Acao |
| --- | --- |
| 0 | parado |
| 1 | esquerda |
| 2 | direita |
| 3 | cima |
| 4 | baixo |
| 5 | tiro |
| 6 | esquerda + tiro |
| 7 | direita + tiro |

## Recompensas

- `+0.01` por passo sobrevivido.
- `+0.10` por meteoro evitado.
- `+1.5` por coletar energia.
- `+2.0` por destruir meteoro.
- `+3.0` por destruir inimigo.
- `-5.0` por colidir com meteoro.
- `-6.0` por colidir com inimigo.
- `-10.0` quando todas as vidas acabam.
- `+5.0` quando o agente chega ao limite de passos do episodio.

## Instalar dependencias

Para treino completo:

```bash
python3 -m pip install -r requirements.txt
```

Para um teste sem instalar Gymnasium, o ambiente tambem roda com o fallback interno de espacos, desde que NumPy esteja instalado.

## Rodar agente aleatorio

```bash
python3 scripts/random_agent.py --episodes 3
```

Com janela visual:

```bash
python3 scripts/random_agent.py --episodes 1 --render
```

## Jogar manualmente

```bash
python3 scripts/play.py
```

Controles:

- setas: mover
- espaco: atirar
- `R`: reiniciar episodio

Esse modo usa `tkinter`, que normalmente ja vem com o Python no macOS. Nao precisa instalar `pygame`.

## Treinar PPO

Treino rapido com observacao por estado:

```bash
python3 scripts/train_ppo.py --obs state --timesteps 100000
```

Treino visual com pixels:

```bash
python3 scripts/train_ppo.py --obs pixels --timesteps 500000
```

O modelo sera salvo em `models/neon_intercept_ppo.zip`.

## Usar em codigo proprio

```python
from atari_rl_game import AtariDodgeEnv

env = AtariDodgeEnv(obs_mode="pixels")
obs, info = env.reset(seed=123)

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

## Testes

```bash
python3 -m unittest discover
```
