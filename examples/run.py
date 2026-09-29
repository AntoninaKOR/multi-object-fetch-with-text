import gym
from tabulate import tabulate
import tqdm
import multi_object_fetch
from multi_object_fetch.language import LanguageWrapper
from multi_object_fetch.utils.gui import GUI
from multi_object_fetch.utils.parser import MOFParser


# Parse environment arguments.
parser = MOFParser()
parser.add_argument("--policy", type=str, choices=["random", "user"], default="random",
                    help="The policy used to control the robot. 'user' for interactive keyboard control, and 'random' for a random policy.")
parser.add_argument("--no_language", action="store_true",
                    help="Disable HomeGrid-style parallel token streaming.")
parser.add_argument("--repeat_task_every", type=int, default=20,
                    help="Re-stream the instruction every N steps once the previous stream finished.")
parser.add_argument("--no_viz", action="store_true",
                    help="Disable on-frame instruction overlay.")
args = parser.parse_args()
env = gym.make(f'{args.environment}{args.task}_{args.num_distractors}Distractors_{args.reward_type}-v1')
if not args.no_language:
    env = LanguageWrapper(
        env,
        repeat_task_every=args.repeat_task_every,
        visualize=not args.no_viz,
    )
env.render(mode='rgb_array')

gui = GUI(fps=env.metadata['video.frames_per_second'])

if args.policy == "user":
    print("Running GUI:")
    print(tabulate([['W', 'forward'], ['S', 'backward'], ['A', 'left'], ['D', 'right'], ['shift', 'down'], ['space', 'up'], ['E', 'open gripper'], ['Q', 'close gripper']], headers=['Key', 'Action']) + "\n")
elif args.policy == "random":
    print("Running random policy:")

max_episode_steps = env.spec.max_episode_steps if env.spec is not None else 100

episode = 0
while True:
    obs, done = env.reset(), False
    print(f"Instruction: {getattr(env, 'instruction', None)}")
    episode += 1
    pbar = tqdm.tqdm(range(max_episode_steps))
    pbar.set_description(f"Episode {episode}")
    for step in pbar:
        gui.display(env.render(mode='rgb_array'))
        from PIL import Image
        im = Image.fromarray(env.render(mode='rgb_array'))
        im.save(f"{args.environment}{args.task}.png")

        if args.policy == "random":
            action = env.action_space.sample()
        elif args.policy == "user":
            action = gui.get_action()
        obs, reward, done, info = env.step(action)
        postfix = {"reward": reward}
        if "log_language_info" in info:
            postfix["lang"] = info.get("log_stream_prefix", info["log_language_info"])
            postfix["tok"] = info.get("log_token_piece", "")
            postfix["token"] = int(obs["token"])
        pbar.set_postfix(**postfix)
        if done:
            break
