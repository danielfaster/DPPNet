# encoding: utf-8
'''=================================================
@Author   : JunBin Yuan
@Date     ：2020/03/06
@Version  ：v1
@Desc   ：calculate the remaining training time
demo:
@time_recoder
def train_val(epoch,total_epoch):
    # train
    train(train_loader, model, optimizer, epoch, save_path)
    # test
    val(test_loader, model, epoch, save_path)

=================================================='''
import datetime
import time


def time_recoder(func):
    time_recorder = TimeUtil()

    def wrapper(*args, **kwargs):
        epoch = args[0]
        total_epoch = args[1]
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_recorder.displayTime(epoch, total_epoch, (end_time - start_time))
        return result

    return wrapper


class TimeUtil:
    def __init__(self, logger=None):
        self.last_10_times = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.total_time = 0
        self.logger = logger

    def my_print(self, str):
        if self.logger is None:
            print(str)
        else:
            self.logger.info(str)

    def displayTime(self, epoch, total_epoch, last_time):
        index = epoch % 10
        self.last_10_times[index] = last_time
        self.total_time += last_time
        last_10_total_time = sum(st for st in self.last_10_times)
        is_first = False
        zero_times = 0
        for i in self.last_10_times:
            if i == 0:
                is_first = True
                zero_times += 1
        if is_first:
            ave_10_time = last_10_total_time / (10 - zero_times)
        elif epoch < 10:
            ave_10_time = last_10_total_time / (epoch + 1)
        else:
            ave_10_time = last_10_total_time / 10

        remain_times = total_epoch - epoch - 1
        if remain_times == 0 or remain_times < 0:
            self.my_print(
                "Training finished. Elapsed: {:.2f} s, average over last 10 iterations: {:.2f} s".format(
                    self.total_time, ave_10_time))
        else:
            remain_time = ave_10_time * remain_times
            remain_time_hour = remain_time // (60 * 60)
            remain_time_min = remain_time % 3600 // 60
            remain_time_sec = remain_time % 3600 % 60
            estimated_finish_time = (
                datetime.datetime.now() + datetime.timedelta(seconds=remain_time)
            ).strftime("%Y-%m-%d %H:%M")

            self.my_print(
                "Elapsed: {:.2f} s, avg last 10 iterations: {:.2f} s, "
                "remaining: {:.0f}H {:.0f}M {:.2f}S, estimated finish: {}".format(
                    self.total_time, ave_10_time,
                    remain_time_hour, remain_time_min, remain_time_sec,
                    estimated_finish_time))

            return estimated_finish_time
