# Reinforcement Learning for Playing Tetris with Prescribed Initial Configuration and Limited Moves

## Table of Contents

- [Introduction](#introduction)
- [Game Rules](#game-rules)
- [Objective](#objective)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Tetris is a well-known video game that has inspired various efforts to train reinforcement learning agents to play it effectively. This project, however, introduces a unique twist by adopting a variation of Tetris known as Tetris-piclim (Tetris with Prescribed Initial Configuration and Limited Moves). Tetris-piclim emulates a playing rule found in the Tetris mobile app created by PLAYSTUDIO.

In Tetris-piclim, the game begins with the Tetris board pre-filled with a prescribed initial configuration, and the player's goal is to clear a specific number of lines within a limited number of moves. This variation introduces new challenges compared to classic Tetris, as players must strategize based on the given configuration and move constraints.

## Game Rules

The basic rules of Tetris-piclim are as follows:

1. The game is played on a 20x10 grid.
2. The board starts with a prescribed initial configuration.
3. The player must clear a specified number of lines (L) within a limited number of moves (M).
4. Players can control the falling Tetriminos using the standard Tetris controls (e.g., moving left/right, rotating).
5. The game ends when the player successfully clears the required number of lines, runs out of moves, or fills the board with Tetriminos.

## Objective

The main objective of this project is to develop a reinforcement learning (RL) agent capable of maximizing the winning probability in Tetris-piclim. Achieving this goal requires the agent to make optimal decisions in terms of clearing lines, managing moves, and dealing with the given initial configuration.

## Getting Started

To get started with this project, follow the steps below:

### Installation

1. Clone the repository:

   ```bash
   git clone 
