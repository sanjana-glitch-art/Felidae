from env.cat_env import CatEnv
from gymnasium.utils.env_checker import check_env
check_env(CatEnv())
print("Environment is valid!")
