from cactpot_env import CactPot
import numpy as np

def main():
    env = CactPot()
    state = env.reset()
    done = False

    while not done:
        env.render()
        if state['phase'] == 0:
            print("Reveal mode: Choose a square to reveal (1-9)")
        else:
            print("Line selection mode: Choose a line (1-8)")

        try:
            action = int(input("Enter your action: ")) - 1
            state, reward, done, _ = env.step(action)
        except ValueError:
            print("Invalid input. Please enter a number.")

        if done:
            print("Game Over! Your reward is:", reward)
            line_sum = np.sum(env.selected_line)
            print("Sum of selected line:", line_sum)
            env.render(show_true_grid=True)

if __name__ == "__main__":
    main()