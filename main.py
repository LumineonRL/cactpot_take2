import numpy as np
from cactpot_env import CactPot

def main():
    env = CactPot()
    obs = env.reset()

    while True:
        # Render the current state of the environment
        env.render()

        # Extract phase information from the observation
        phase = obs[-1]  # Assuming the phase is the last element of the observation

        if phase == 0:
            # Collect action from the user for reveal phase
            action = int(input("Choose a number to reveal (1-9): ")) - 1
        else:
            # Collect action from the user for line selection phase
            action = int(input("Choose a line (1-8): ")) - 1

        # Step through the environment with the selected action
        obs, reward, done, _ = env.step(action)

        if done:
            print("Game Over! Your reward is:", reward)
            line_sum = np.sum(env.true_grid[env.selected_line])
            print("Sum of selected line:", line_sum)
            env.render(show_true_grid=True)
            break

if __name__ == "__main__":
    main()