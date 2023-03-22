# to change the default working directory in PyCharm

import os

print(os.getcwd())


def main():
    # MY EXPERIMENT
    from src.mclip import gen_from_text, ae_from_cmu, calc_cos_sim, action_classify
    # gen_from_text()
    # ae_from_cmu()
    calc_cos_sim()
    # action_classify()

    # CLASSIFICATION
    # from src.utils import action_classifier
    # action_classifier.main()

    # GENERATION
    # import src.visualize.text2motion as t2m
    # t2m.main()


if __name__ == '__main__':
    main()
