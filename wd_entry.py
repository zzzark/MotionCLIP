# to change the default working directory in PyCharm

import os

print(os.getcwd())


def main():
    from src.mclip import gen_from_text, ae_from_cmu, calc_cos_sim
    # gen_from_text()
    # ae_from_cmu()
    calc_cos_sim()

    # import src.visualize.text2motion as t2m
    # t2m.main()


if __name__ == '__main__':
    main()
