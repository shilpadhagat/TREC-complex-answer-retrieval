from multiprocessing import Process
from Test_Class import Test


def func1():
    t_object = Test(0, 11)
    t_object.run_ranking()


def func2():
    t_object = Test(11, 21)
    t_object.run_ranking()


def func3():
    t_object = Test(21, 31)
    t_object.run_ranking()


def func4():
    t_object = Test(31, 41)
    t_object.run_ranking()


def func5():
    t_object = Test(41, 51)
    t_object.run_ranking()


def func6():
    t_object = Test(51, 61)
    t_object.run_ranking()


def func7():
    t_object = Test(61, 71)
    t_object.run_ranking()


def func8():
    t_object = Test(71, 81)
    t_object.run_ranking()


def func9():
    t_object = Test(81, 91)
    t_object.run_ranking()


def func10():
    t_object = Test(91, 98)
    t_object.run_ranking()

if __name__ == '__main__':
    p1 = Process(target=func1)
    p1.start()
    p2 = Process(target=func2)
    p2.start()
    p3 = Process(target=func3)
    p3.start()
    p4 = Process(target=func4)
    p4.start()
    p5 = Process(target=func5)
    p5.start()
    p6 = Process(target=func6)
    p6.start()
    p7 = Process(target=func7)
    p7.start()
    p8 = Process(target=func8)
    p8.start()
    p9 = Process(target=func9)
    p9.start()
    p10 = Process(target=func10)
    p10.start()
