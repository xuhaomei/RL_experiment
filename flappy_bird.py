# coding=utf-8
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from agents.flappy_bird_agent import Agent

if __name__ == "__main__":
    # 训练次数
    episodes = 20000
    # 实例化游戏对象
    game = FlappyBird()
    # 类似游戏的一个接口，可以为我们提供一些功能
    p = PLE(game, fps=30, display_screen=False)
    # 初始化
    p.init()
    # 实例化Agent，将动作集传进去
    agent = Agent(p.getActionSet())
    max_score = 0

    for episode in range(episodes):
        # 重置游戏
        p.reset_game()
        # 获得状态
        state = agent.get_state(game.getGameState())
        agent.update_greedy()
        while True:
            # 获得最佳动作
            action = agent.get_best_action(state)
            # 然后执行动作获得奖励
            reward = agent.act(p, action)
            # 获得执行动作之后的状态
            next_state = agent.get_state(game.getGameState())
            # 更新q-table
            agent.update_q_table(state, action, next_state, reward)
            # 获得当前分数
            current_score = p.score()
            state = next_state
            if p.game_over():
                max_score = max(current_score, max_score)
                print('Episodes: %s, Current score: %s, Max score: %s' % (episode, current_score, max_score))
                # 保存q-table
                if current_score > 300:
                    np.save("{}_{}.npy".format(current_score, episode), agent.q_table)
                break
