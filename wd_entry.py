# to change the default working directory in PyCharm

import os
import src.visualize.text2motion as t2m

print(os.getcwd())


def main():
    # Text-to-Motion
    t2m.main()


if __name__ == '__main__':
    main()
