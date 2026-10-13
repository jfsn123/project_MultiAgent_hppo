import sys
sys.path.append('D:\\project_MultiAgent_hppo')

from python_scripts.PPO.episode.PPO_episoid_0 import PPO_episoid_0
from python_scripts.PPO.test.full_climb_test import full_climb_test
from python_scripts.PPO.test.decision_test import decision_models_test
from python_scripts.PPO.test.ground_catch_test import ground_catch_models_test
from python_scripts.PPO.test.tai_test import tai_models_test
from python_scripts.PPO.test.upper_catch_test import upper_grasp_test
from python_scripts.PPO.utils.Project_config import path_list
from controller import Robot

# Training modes:
# - "full": decision + ground grasp + stepping + upper grasp
# - "no_upper": decision + ground grasp + stepping
# - "catch_only": only train ground grasp
# - "tai_only": only train stepping, loading decision/catch models
# - "upper_only": only train upper grasp, loading decision/catch/tai models
# Test modes:
# - "test_full_climb": test full climb flow
# - "test_decision_batch": batch-test decision models
# - "test_catch_batch": batch-test ground catch models
# - "test_tai_batch": batch-test stepping models
# - "test_upper_grasp_only": test upper grasp only

<<<<<<< HEAD
TRAIN_MODE = "full"
=======
TRAIN_MODE = "test_full_climb"
>>>>>>> 4b6b461 (first commit)


def main():
    robot = Robot()

    if TRAIN_MODE == "test_full_climb":
        print("Test mode: full climb flow (100 cycles)")
        results = full_climb_test(robot)
        return

    if TRAIN_MODE == "test_decision_batch":
        print("Test mode: batch decision-model success rate")
        results = decision_models_test(robot)
        print(f"Batch test finished, model count: {len(results)}")
        return

    if TRAIN_MODE == "test_catch_batch":
        print("Test mode: batch ground-catch success rate")
        results = ground_catch_models_test(robot)
        print(f"Batch test finished, model count: {len(results)}")
        return

    if TRAIN_MODE == "test_tai_batch":
        print("Test mode: batch stepping-model success rate")
        results = tai_models_test(robot)
        print(f"Batch test finished, model count: {len(results)}")
        return

    if TRAIN_MODE == "test_upper_grasp_only":
        print("Test mode: upper grasp only")
        results = upper_grasp_test(robot)
        print(f"Batch test finished, model count: {len(results)}")
        return
        
    train_upper = (TRAIN_MODE in ["full", "upper_only"])
    train_decision = (TRAIN_MODE in ["full", "no_upper"])
    train_catch = (TRAIN_MODE in ["full", "no_upper", "catch_only"])
    train_tai = (TRAIN_MODE in ["full", "no_upper", "tai_only"])

    if TRAIN_MODE == "full":
        print("Train mode: decision + ground grasp + stepping + upper grasp")
    elif TRAIN_MODE == "no_upper":
        print("Train mode: decision + ground grasp + stepping")
    elif TRAIN_MODE == "catch_only":
        print("Train mode: ground grasp only")
    elif TRAIN_MODE == "tai_only":
        print("Train mode: stepping only")
    elif TRAIN_MODE == "upper_only":
        print("Train mode: upper grasp only")
    else:
        raise ValueError(f"Unknown TRAIN_MODE: {TRAIN_MODE}")

    PPO_episoid_0(
        robot=robot,
        decision_model_path=None,
        catch_model_path=None,
        tai_model_path=None,
        upper_model_path=None,
        train_upper=train_upper,
        train_decision=train_decision,
        train_catch=train_catch,
        train_tai=train_tai,
    )


if __name__ == '__main__':
    print("_________")
    with open(path_list['resetFlag'], 'r+') as file:
        file.write('1')
    main()
