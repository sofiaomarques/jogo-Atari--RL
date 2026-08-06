# Atari Paddle RL

Um jogo estilo Atari, inspirado em Breakout, em que um agente aprende sozinho a mover uma raquete para nao deixar a bola cair e destruir os blocos no topo da tela.

O objetivo do modelo e maximizar pontos. Cada vez que a raquete rebate a bola ou a bola destroi um bloco, o agente ganha pontos e recompensa. Quando deixa a bola cair, perde vida e recebe uma penalidade grande. Ao limpar todos os blocos, uma nova fase com blocos aparece.

## Como rodar

Entre na pasta certa:

```bash
cd "/Users/sofiadeoliveiramarques/projetos pessoais/jogo atari- aprendizadoo por reforco"
```

Instale a dependencia basica:

```bash
python3 -m pip install -r requirements.txt
```

## Treinar o agente sozinho

Este comando treina uma politica por Q-learning:

```bash
python3 scripts/train_q_learning.py --episodes 1000
```

Durante o treino, o terminal mostra a media de pontos e de rebatidas. Se quiser que ele aprenda melhor, aumente os episodios:

```bash
python3 scripts/train_q_learning.py --episodes 3000
```

A politica aprendida fica salva em:

```text
models/q_policy.npz
```

## Ver o agente treinado jogando

Depois de treinar, rode:

```bash
python3 scripts/watch_trained_agent.py
```

Ele abre uma janela e o agente joga sozinho usando a politica salva.

## Jogar manualmente

```bash
python3 scripts/play.py
```

Controles:

- seta esquerda: mover para esquerda
- seta direita: mover para direita
- `R`: reiniciar

## Ambiente de aprendizado por reforco

O ambiente principal e `AtariDodgeEnv`, compativel com Gymnasium quando Gymnasium esta instalado, mas tambem funciona sem ele usando os espacos internos do projeto.

Acoes:

| Valor | Acao |
| --- | --- |
| 0 | parado |
| 1 | esquerda |
| 2 | direita |

Observacao `state`:

- posicao da raquete
- posicao da bola
- velocidade da bola
- distancia ate o ponto onde a bola deve chegar
- vidas restantes
- sequencia de rebatidas

Recompensas principais:

- recompensa pequena por sobreviver;
- recompensa por se aproximar do ponto de queda da bola;
- recompensa alta ao rebater a bola e destruir blocos;
- penalidade alta ao deixar a bola cair.

## Testes

```bash
python3 -m unittest discover
```

## PPO opcional

O projeto ainda tem um script opcional para Stable-Baselines3:

```bash
python3 -m pip install gymnasium stable-baselines3
python3 scripts/train_ppo.py --obs state --timesteps 100000
```
PRA TESTAR!!!!!

cd "/Users/sofiadeoliveiramarques/projetos pessoais/jogo atari- aprendizadoo por reforco"
python3 scripts/watch_trained_agent.py

Mas para o seu caso, comece por `train_q_learning.py`, que e mais simples e ja salva uma politica treinada.
